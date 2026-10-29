import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from spamosaic.architectures.wlgcn import WLGCN
from spamosaic.loss import PairedAlignmentLoss
from spamosaic.topola import reference_topola_features
from spamosaic.train_utils import set_seeds
from spamosaic.utils import mclust_R
from spamosaic import metrics

EXP = Path(__file__).resolve().parents[1]
CHECKS = {0, 1, 5, 10, 25, 50, 100, 200, 300}

def forward(models, graph, splits):
    embeddings, reconstructions = {}, {}
    for m, model in models.items():
        z, r = model(graph["attribute"][m], graph["edge"][m], graph["edgeW"][m])
        embeddings[m] = torch.split(z, splits[m])
        reconstructions[m] = torch.split(r, splits[m])
    return embeddings, reconstructions

def reconstruction(recs, graph, splits, modalities):
    return torch.stack([
        F.mse_loss(recs[m][i], torch.split(graph["attribute"][m], splits[m])[i])
        for i in (1, 2) for m in modalities if splits[m][i] > 0
    ]).mean()

def pair_features(z, modalities, ids):
    return torch.cat([z[m][0][ids] for m in modalities])

def grad_vector(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=True)
    return torch.cat([g.reshape(-1) for g in gradients])

def reference_ids(cfg, q, train, epoch, seed, spatial_order):
    if cfg["reference"] == "query":
        return q
    extra = train[~torch.isin(train, q)]
    if cfg["reference"] == "random":
        extra = extra[torch.randperm(len(extra), generator=torch.Generator().manual_seed(seed + 70000 + epoch))]
    if cfg["reference"] == "spatial":
        extra = spatial_order[~torch.isin(spatial_order, q)]
    count = len(train) if cfg["reference_size"] == 0 else min(cfg["reference_size"], len(train))
    return torch.cat([q, extra[:count-len(q)]])

def score(z, modalities, q, r, cfg):
    e = pair_features(z, modalities, q)
    lam = cfg["lambda"] * (len(r) / 256 if cfg["size_scaled"] else 1)
    if cfg["score"] == "dot":
        return e @ e.T
    f = reference_topola_features(e, pair_features(z, modalities, r), lam)
    return f @ f.T

def retrieval(scores):
    n = scores.shape[0] // 2
    cross = scores[:n, n:]
    values = {}
    for direction, x in (("ab", cross), ("ba", cross.T)):
        rank = 1 + (x > x.diagonal()[:, None]).sum(1)
        values[direction + "_r1"] = (rank == 1).float().mean().item()
        values[direction + "_r10"] = (rank <= 10).float().mean().item()
        values[direction + "_mrr"] = rank.float().reciprocal().mean().item()
    mask = ~torch.eye(n, dtype=torch.bool, device=scores.device)
    margin = (cross.diagonal()[:, None] - cross)[mask]
    for p, val in zip((.05, .5, .95), torch.quantile(margin, torch.tensor([.05, .5, .95], device=scores.device))):
        values[f"margin_q{p}"] = val.item()
    return values

def evaluate(models, graph, splits, modalities, train, valid, cfg, epoch, run, template, temperature, alpha):
    for model in models.values():
        model.eval()
    with torch.no_grad():
        z, recs = forward(models, graph, splits)
        ref = pair_features(z, modalities, train)
        evaluation_lambda = cfg["lambda"] * (len(train)/256 if cfg["size_scaled"] else 1)
        query = pair_features(z, modalities, valid)
        diag = {"epoch": epoch, "temperature": temperature, "alpha": alpha, "evaluation_reference_size":len(train), "evaluation_lambda":evaluation_lambda}
        for name, f in (("cosine", query), ("kernel", reference_topola_features(query, ref, evaluation_lambda))):
            diag.update({name + "_" + k: v for k, v in retrieval(f @ f.T).items()})
        eig = torch.linalg.eigvalsh(ref.double().T @ ref.double())
        probability = eig / eig.sum()
        diag.update(eig_min=eig[0].item(), eig_median=eig.median().item(), eig_max=eig[-1].item(),
                    effective_rank=torch.exp(-(probability * probability.log()).sum()).item(),
                    shrink_min=(eig[0]/(eig[0]+cfg["lambda"])).item(),
                    shrink_max=(eig[-1]/(eig[-1]+cfg["lambda"])).item())
        fixed = train[:min(256, len(train))]
        probe = pair_features(z, modalities, fixed)
        dot = probe @ probe.T
        samples = []
        ranks = []
        for j in range(4):
            rest = train[~torch.isin(train, fixed)]
            order = torch.randperm(len(rest), generator=torch.Generator().manual_seed(90000+j))
            ids = torch.cat([fixed, rest[order[:min(768, len(rest))]]])
            f = reference_topola_features(probe, pair_features(z, modalities, ids), cfg["lambda"])
            s = f @ f.T
            samples.append(s)
            ranks.append(retrieval(s)["ab_mrr"])
        diag["reference_score_std"] = torch.stack(samples).std(0).mean().item()
        diag["reference_mrr_std"] = float(np.std(ranks))
        norms = torch.cat([z[m][i].norm(dim=1) for m in modalities for i in range(3) if splits[m][i]])
        diag["single_norm_mean"] = norms.mean().item()
        diag["bridge_mean_norm"] = ((z[modalities[0]][0]+z[modalities[1]][0])/2).norm(dim=1).mean().item()
        if epoch in (100, 300) or epoch == cfg["epochs"]:
            rows = []
            raw = [torch.stack([z[m][i] for m in modalities if splits[m][i]]).mean(0) for i in range(3)]
            all_raw = torch.cat(raw)
            transformed = reference_topola_features(all_raw, ref, evaluation_lambda)
            for representation, emb in (("raw", all_raw), ("reference", transformed)):
                ad = template.copy()
                ad.obsm["merged_emb"] = emb.cpu().numpy()
                if cfg["dataset"] == "Simulations":
                    ad.obs["cluster"] = KMeans(n_clusters=5, random_state=0, n_init=10).fit_predict(ad.obsm["merged_emb"]).astype(str)
                else:
                    mclust_R(ad, 7, used_obsm="merged_emb")
                    ad.obs["cluster"] = ad.obs["mclust"].astype(str)
                boundaries = np.cumsum([0] + [len(x) for x in raw])
                sections = [ad[boundaries[i]:boundaries[i+1]].copy() for i in range(3)]
                if cfg["dataset"] == "Simulations":
                    rows.append(("ARI", "all", adjusted_rand_score(ad.obs["gt"].astype(str), ad.obs["cluster"])))
                    rows.extend(("ARI", f"slice{i+1}", adjusted_rand_score(s.obs["gt"].astype(str), s.obs["cluster"])) for i,s in enumerate(sections))
                    rows.append(("F1", "slice2-slice3", metrics.LabTransfer(sections[1], sections[2], "merged_emb", "gt", knn=10)))
                for i, s in enumerate(sections):
                    rows.extend([("PAS", f"slice{i+1}", metrics.compute_PAS2(s, "cluster")),
                                 ("CHAOS", f"slice{i+1}", metrics.compute_CHAOS2(s, "cluster"))])
                rows.append(("iLISI", "all", metrics.iLISI(ad, "src", "merged_emb")))
                pd.DataFrame(rows, columns=["metric", "scope", "value"]).to_csv(run / f"metrics_e{epoch}_{representation}.csv", index=False)
                rows = []
            np.savez_compressed(run / f"embeddings_e{epoch}.npz",
                                **{f"m{j}_s{i}": z[m][i].cpu().numpy() for j,m in enumerate(modalities) for i in range(3)})
            torch.save({m: model.state_dict() for m,model in models.items()}, run / f"model_e{epoch}.pt")
        return diag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    cfg = json.loads(args.job.read_text())
    torch.set_num_threads(4)
    torch.cuda.set_device(args.device)
    set_seeds(cfg["seed"])
    cache = EXP / "results/cache" / f'{cfg["dataset"]}_rep{cfg["rep"]}'
    data = torch.load(cache / "inputs.pt", weights_only=False)
    template = sc.read_h5ad(cache / "template.h5ad")
    modalities, splits = data["modalities"], data["splits"]
    graph = {k: {m: x.to(args.device) for m,x in v.items()} for k,v in data["graph"].items()}
    run = EXP / "results/runs" / cfg["id"]
    run.mkdir(parents=True, exist_ok=False)
    (run / "config.json").write_text(json.dumps(cfg, indent=2))
    n = splits[modalities[0]][0]
    order = torch.randperm(n, generator=torch.Generator().manual_seed(1701))
    n_valid = max(2, int(n*.2)) if cfg["holdout"] else 0
    valid = order[:n_valid] if n_valid else order[:min(256, n)]
    train = order[n_valid:]
    np.savez(run / "split_indices.npz", train=train.numpy(), validation=valid.numpy())
    coords = np.asarray(template.obsm["spatial"][:n])[train.numpy()]
    # Farthest-point traversal gives a deterministic spatial coverage order.
    coverage = [0]
    distance = np.full(len(train), np.inf)
    for j in range(len(train)-1):
        distance = np.minimum(distance, ((coords-coords[coverage[-1]])**2).sum(1))
        distance[coverage] = -1
        coverage.append(int(distance.argmax()))
    spatial_order = train[torch.tensor(coverage)]
    models = {m: WLGCN(graph["attribute"][m].shape[1], 32, K=2, hidden_size=128,
                        dropout=.2, dec_l=1, slope=.2).to(args.device) for m in modalities}
    optimizers = {m: torch.optim.Adam(model.parameters(), lr=.01, weight_decay=5e-4) for m,model in models.items()}
    parameters = [p for model in models.values() for name,p in model.named_parameters() if not name.startswith("decoder.")]
    temperature = cfg["temperature"]
    alpha = cfg["alpha"]
    q = train[:cfg["query_size"]]
    r = reference_ids(cfg, q, train, 0, cfg["seed"], spatial_order)
    criterion = PairedAlignmentLoss(len(q), cfg["objective"], cfg["negatives"]).to(args.device)
    for model in models.values(): model.eval()
    z, recs = forward(models, graph, splits)
    initial_scores = score(z, modalities, q, r, cfg)
    e = pair_features(z, modalities, q)
    baseline = PairedAlignmentLoss(len(q), "cl", "all").to(args.device)
    dot = e @ e.T
    base_temperature = .1 if cfg["dataset"] == "Simulations" else .01
    if cfg["calibration"] in ("temperature", "both"):
        gap = initial_scores - initial_scores.gather(1, criterion.positive[:,None])
        dot_gap = dot - dot.gather(1, criterion.positive[:,None])
        temperature = base_temperature * (gap[criterion.negative_mask].std()/dot_gap[criterion.negative_mask].std()).item() * cfg["temperature_multiplier"]
    if cfg["calibration"] == "both":
        base_grad = grad_vector(baseline(dot/base_temperature), parameters)
        target_grad = grad_vector(criterion(initial_scores/temperature), parameters)
        alpha = (base_grad.norm()/target_grad.norm()).item() * cfg["alpha"]
    (run / "calibration.json").write_text(json.dumps({"temperature":temperature,"alpha":alpha},indent=2))
    del z, recs, initial_scores, e, dot
    history, diagnostics, gradients = [], [], []
    started = time.time()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        diagnostics.append(evaluate(models,graph,splits,modalities,train,valid,cfg,0,run,template,temperature,alpha))
    for epoch in range(1, cfg["epochs"]+1):
        for model in models.values(): model.train()
        for optimizer in optimizers.values(): optimizer.zero_grad()
        z, recs = forward(models, graph, splits)
        rec = reconstruction(recs, graph, splits, modalities)
        if cfg["fixed_query"]:
            batches = [q]
        else:
            permutation = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(cfg["seed"]+epoch+10000))]
            batches = list(torch.tensor_split(permutation, math.ceil(len(permutation)/cfg["query_size"])))
            if len(batches[-1]) == 1:
                batches[-2] = torch.cat([batches[-2],batches[-1]])
                batches.pop()
        # Normalize per anchor, including the final shorter batch.
        terms = []
        for ids in batches:
            refs = reference_ids(cfg,ids,train,epoch,cfg["seed"],spatial_order)
            loss = PairedAlignmentLoss(len(ids),cfg["objective"],cfg["negatives"]).to(args.device)(
                score(z,modalities,ids,refs,cfg)/temperature)
            terms.append(loss * len(ids)/sum(len(x) for x in batches))
        align = torch.stack(terms).sum()
        if epoch in CHECKS or epoch == cfg["epochs"]:
            ga, gr = grad_vector(align,parameters), grad_vector(rec,parameters)
            gradients.append({"epoch":epoch,"alignment_norm":ga.norm().item(),"reconstruction_norm":gr.norm().item(),
                              "weighted_ratio":(alpha*ga.norm()/gr.norm()).item(),
                              "cosine":F.cosine_similarity(ga,gr,dim=0).item()})
            pd.DataFrame(gradients).to_csv(run/"gradients.csv",index=False)
        total = alpha*align+rec
        if not torch.isfinite(total):
            raise FloatingPointError(f"Nonfinite objective at epoch {epoch}")
        total.backward()
        for optimizer in optimizers.values(): optimizer.step()
        history.append({"epoch":epoch,"alignment":align.item(),"reconstruction":rec.item(),
                        "total":total.item(),"elapsed_seconds":time.time()-started})
        pd.DataFrame(history).to_csv(run/"losses.csv",index=False)
        if epoch in CHECKS or epoch == cfg["epochs"]:
            numpy_state, python_state = np.random.get_state(), random.getstate()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                diagnostics.append(evaluate(models,graph,splits,modalities,train,valid,cfg,epoch,run,template,temperature,alpha))
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            pd.DataFrame(diagnostics).to_csv(run/"diagnostics.csv",index=False)
            print(cfg["id"],epoch,history[-1],flush=True)
    (run/"complete.json").write_text(json.dumps({"elapsed_seconds":time.time()-started,"epochs":cfg["epochs"]}))
    print("COMPLETE",cfg["id"],flush=True)

if __name__ == "__main__":
    main()
