import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import torch
from spamosaic.framework import SpaMosaic
from spamosaic.preprocessing import RNA_preprocess, ADT_preprocess, Epigenome_preprocess
from spamosaic.train_utils import set_seeds

PARAMS = {
    "Simulations": {"input_key": "X_pca_be", "T": .1, "n_cluster": 5},
    "MB-H3K4me3": {"input_key": "dimred_bc", "T": .01, "n_cluster": 7},
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=PARAMS, required=True)
    parser.add_argument("--rep", default="1")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.data_root = Path("/data/home/jiboya/SpaMosaic/data/processed")
    args.sim_setting = "rna-0.1-12-adt-2.1"
    args.sim_rep = args.rep
    dataset = args.dataset
    seed = 1234
    if dataset == 'Simulations':
        data_dir = args.data_root / dataset / args.sim_setting / args.sim_rep
        layout = {'rna': ['ad_rna1', 'ad_rna2', None], 'adt': ['ad_adt1', None, 'ad_adt2']}
    else:
        data_dir = args.data_root / dataset
        layout = {'rna': ['S1/ad_rna', 'S2/ad_rna', None],
                  'H3K4me3': ['S1/ad_h3k4me3', None, 'S3/ad_h3k4me3']}
    raw = {}
    for names in layout.values():
        for i, name in enumerate(names):
            if name is None:
                continue
            ad = sc.read_h5ad(data_dir / (name + '.h5ad'))
            if dataset == 'Simulations':
                ad.obs['src'] = ['mult', 'rna', 'adt'][i]
                ad.X = ad.layers['counts'].copy()
            else:
                ad.X = scipy.sparse.csr_matrix(ad.X, dtype=np.float32)
            raw[name] = ad
            print(f'Loaded {dataset}/{name}: {ad.shape}', flush=True)
    inputs = {m: [raw[n] if n is not None else None for n in names] for m, names in layout.items()}
    modalities = list(inputs)
    if not inputs[modalities[0]][0].obs_names.equals(inputs[modalities[1]][0].obs_names):
        raise ValueError('Bridge modalities must have identical spot names in the same order')
    representatives = [inputs[modalities[0]][0], inputs[modalities[0]][1], inputs[modalities[1]][2]]
    if not pd.Index(np.concatenate([ad.obs_names for ad in representatives])).is_unique:
        raise ValueError('Spot names must be unique across slices')
    if dataset == 'Simulations':
        for ad in representatives:
            if ad.obs['gt'].isna().any():
                raise ValueError('Simulation ground-truth labels must not be missing')
    else:
        print('MB-H3K4me3: no gt in notebook inputs; evaluating PAS, CHAOS, iLISI only.', flush=True)


    set_seeds(seed)
    if dataset == 'Simulations':
        RNA_preprocess(inputs['rna'], batch_corr=True, favor='scanpy', n_hvg=None,
                       lognorm=False, scale=True, batch_key='src', key=PARAMS[dataset]['input_key'])
        ADT_preprocess(inputs['adt'], batch_corr=True, favor='scanpy', lognorm=False,
                       scale=True, batch_key='src', key=PARAMS[dataset]['input_key'])
    else:
        RNA_preprocess(inputs['rna'], batch_corr=True, n_hvg=5000,
                       batch_key='src', key=PARAMS[dataset]['input_key'])
        Epigenome_preprocess(inputs['H3K4me3'], batch_corr=True, n_peak=50000,
                             batch_key='src', key=PARAMS[dataset]['input_key'])
    model = SpaMosaic(modBatch_dict=inputs, input_key=PARAMS[dataset]['input_key'],
                      batch_key='src', intra_knns=10, inter_knn_base=10,
                      w_g=.8, seed=seed, device=args.device)

    output = Path(__file__).resolve().parents[1] / "results" / "cache" / f"{dataset}_rep{args.rep}"
    output.mkdir(parents=True, exist_ok=False)
    graph = {key: {m: x.cpu() for m, x in model.mod_graphs[key].items()}
             for key in ("edge", "edgeW", "attribute")}
    torch.save({"graph": graph, "splits": model.mod_batch_split,
                "modalities": modalities, "dataset": dataset, "rep": args.rep},
               output / "inputs.pt")
    sections = [sc.AnnData(np.zeros((ad.n_obs, 1), dtype=np.float32), obs=ad.obs.copy(),
                          obsm={"spatial": ad.obsm["spatial"].copy()}) for ad in representatives]
    sc.concat(sections).write_h5ad(output / "template.h5ad")
    (output / "provenance.json").write_text(json.dumps({
        "preprocess_seed": seed, "dataset": dataset, "rep": args.rep,
        "bridge_names": inputs[modalities[0]][0].obs_names.tolist(),
        "params": PARAMS[dataset]}, indent=2))
    print("PREPARED", output, flush=True)

if __name__ == "__main__":
    main()
