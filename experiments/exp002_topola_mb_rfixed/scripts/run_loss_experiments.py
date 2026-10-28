#!/usr/bin/env python
"""Four-way loss ablation using the datasets and metrics in run_demo.ipynb.

Examples (run in the server's SpaMosaic environment):
    python run_loss_experiments.py --smoke-test --device cuda:0
    python run_loss_experiments.py --datasets Simulations --device cuda:0
    python run_loss_experiments.py --datasets MB-H3K4me3 --device cuda:0
    python run_loss_experiments.py --seeds 1234 2024 2025 --temperatures .01 .1

Default BPR uses the same temperature as CL. --bpr-scaling paper instead uses
T=1 for dot BPR and T=lambda_topola for TopoLa BPR+, following paper Eq. (7).
The encoder's existing weight decay is retained; there is no ID embedding penalty.
For >2 modalities, a single joint TopoLa matrix is used for all modality pairs.
"""

import argparse
import gc
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

ROOT = Path(__file__).resolve().parent
VARIANTS = {
    'dot_cl': ('dot', 'adapted'),
    'topola_cl': ('topola', 'adapted'),
    'dot_bpr': ('dot', 'bpr'),
    'topola_bpr': ('topola', 'bpr'),
}
PARAMS = {
    'Simulations': {'input_key': 'X_pca_be', 'T': .1, 'n_cluster': 5},
    'MB-H3K4me3': {'input_key': 'dimred_bc', 'T': .01, 'n_cluster': 7},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', choices=PARAMS, default=list(PARAMS))
    parser.add_argument('--data-root', type=Path, default=ROOT / 'data' / 'processed')
    parser.add_argument('--sim-setting', default='rna-0.1-12-adt-2.1')
    parser.add_argument('--sim-rep', choices=['1', '2', '3'], default='1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1234])
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--temperatures', nargs='+', type=float, help='Default: notebook temperature for each dataset')
    parser.add_argument('--topola-lambdas', nargs='+', type=float, default=[10.0])
    parser.add_argument('--bpr-scaling', choices=['temperature', 'paper'], default='temperature')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--mini-batch-size', type=int, default=1024)
    parser.add_argument('--use-mini-thr', type=int, default=8000)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'loss_experiments' / time.strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--smoke-test', action='store_true', help='Synthetic gradients and short WLGCN training; no data or output files')
    args = parser.parse_args()
    for value in (args.temperatures or []) + args.topola_lambdas + [args.lr]:
        if not math.isfinite(value) or value <= 0:
            parser.error('Temperatures, lambda, and learning rate must be finite and positive')
    if args.epochs < 1 or args.mini_batch_size < 2 or args.use_mini_thr < 2:
        parser.error('epochs must be >=1; mini-batch-size and use-mini-thr must be >=2')

    # Keep --help usable without importing the scientific stack or R dependencies.
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse
    import torch
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    from spamosaic.framework import SpaMosaic
    from spamosaic.preprocessing import RNA_preprocess, ADT_preprocess, Epigenome_preprocess
    from spamosaic.train_utils import set_seeds
    from spamosaic.utils import mclust_R, split_adata_ob

    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        parser.error('Use cpu or cuda:N; the existing SpaMosaic framework does not support MPS')
    torch.empty(0, device=device)  # Fail at the requested device instead of falling back to CPU.
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if args.smoke_test:
        smoke_test(device)
        return

    sys.path.insert(0, str(ROOT / 'reproduce/code/compute_metrics'))
    import metrics

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / 'config.json').open('w') as f:
        json.dump(vars(args), f, default=str, indent=2)
    results = []
    for dataset in args.datasets:
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

        for seed in args.seeds:
            # These functions write obsm, leaving the count matrices available for the next seed.
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
            # Reuse this graph and the exact post-construction RNG state for every variant.
            random_state = random.getstate()
            numpy_state = np.random.get_state()
            torch_state = torch.get_rng_state()
            cuda_states = torch.cuda.get_rng_state_all() if device.type == 'cuda' else []
            configurations = []
            for variant in args.variants:
                score_type, loss_type = VARIANTS[variant]
                lambdas = args.topola_lambdas if score_type == 'topola' else [10.0]
                for regularization in lambdas:
                    temperatures = args.temperatures or [PARAMS[dataset]['T']]
                    if loss_type == 'bpr' and args.bpr_scaling == 'paper':
                        temperatures = [regularization if score_type == 'topola' else 1.0]
                    configurations.extend((variant, regularization, t) for t in temperatures)
            for run_index, (variant, regularization, temperature) in enumerate(configurations):
                score_type, loss_type = VARIANTS[variant]
                random.setstate(random_state)
                np.random.set_state(numpy_state)
                torch.set_rng_state(torch_state)
                if device.type == 'cuda':
                    torch.cuda.set_rng_state_all(cuda_states)
                run_dir = args.output_dir / dataset / f'seed{seed}' / f'{run_index:03d}_{variant}_T{temperature:g}_lambda{regularization:g}'
                run_dir.mkdir(parents=True)
                print(f'Running {run_dir}', flush=True)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                model.train(net='wlgcn', lr=args.lr, T=temperature, n_epochs=args.epochs,
                            use_mini_thr=args.use_mini_thr, mini_batch_size=args.mini_batch_size,
                            loss_type=loss_type, score_type=score_type, lambda_topola=regularization)
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                training_seconds = time.perf_counter() - started
                pd.DataFrame({'alignment_sum': model.loss_cl, 'reconstruction_sum': model.loss_rec}).to_csv(run_dir / 'losses.csv', index=False)
                with torch.no_grad():
                    sections = model.infer_emb(inputs)
                mosaic = sc.concat(sections)
                mosaic.write_h5ad(run_dir / 'embedding.h5ad')
                if dataset == 'Simulations':
                    mosaic.obs['cluster'] = KMeans(n_clusters=PARAMS[dataset]['n_cluster'], random_state=0).fit_predict(mosaic.obsm['merged_emb']).astype(str)
                else:
                    mclust_R(mosaic, PARAMS[dataset]['n_cluster'], used_obsm='merged_emb')
                    mosaic.obs['cluster'] = mosaic.obs['mclust'].astype(str)
                split_adata_ob(sections, mosaic, 'obs', 'cluster')
                scores = []
                if dataset == 'Simulations':
                    scores.append(('ARI', 'all', adjusted_rand_score(mosaic.obs['gt'].astype(str), mosaic.obs['cluster'])))
                    for i, ad in enumerate(sections):
                        scores.append(('ARI', f'slice{i+1}', adjusted_rand_score(ad.obs['gt'].astype(str), ad.obs['cluster'])))
                    scores.append(('F1', 'slice2-slice3', metrics.LabTransfer(sections[1], sections[2], 'merged_emb', 'gt', knn=10)))
                for i, ad in enumerate(sections):
                    scores.append(('PAS', f'slice{i+1}', metrics.compute_PAS2(ad, 'cluster')))
                    scores.append(('CHAOS', f'slice{i+1}', metrics.compute_CHAOS2(ad, 'cluster')))
                scores.append(('iLISI', 'all', metrics.iLISI(mosaic, 'src', 'merged_emb')))
                mosaic.write_h5ad(run_dir / 'embedding.h5ad')
                for metric, scope, value in scores:
                    results.append(dict(dataset=dataset, seed=seed, variant=variant, temperature=temperature,
                                        lambda_topola=regularization if score_type == 'topola' else None,
                                        bpr_scaling=args.bpr_scaling, epochs=args.epochs,
                                        metric=metric, scope=scope, value=value, training_seconds=training_seconds))
                pd.DataFrame(scores, columns=['metric', 'scope', 'value']).to_csv(run_dir / 'metrics.csv', index=False)
                pd.DataFrame(results).to_csv(args.output_dir / 'metrics.csv', index=False)
                pd.DataFrame(results).groupby(
                    ['dataset', 'variant', 'temperature', 'lambda_topola', 'bpr_scaling', 'epochs', 'metric', 'scope'],
                    dropna=False,
                )['value'].agg(['mean', 'std', 'count']).to_csv(args.output_dir / 'summary.csv')
                print(pd.DataFrame(scores, columns=['metric', 'scope', 'value']).to_string(index=False), flush=True)
                del model.mod_model, sections, mosaic
                gc.collect()
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            del model
        del inputs, raw, representatives
        gc.collect()
    print(f'Results: {args.output_dir / "metrics.csv"}', flush=True)


def smoke_test(device):
    """Check QR scores, ranking orientation, gradients, and actual multi-modal WLGCN training."""
    import torch
    import torch.nn.functional as F
    from spamosaic.architectures.wlgcn import WLGCN
    from spamosaic.loss import CL_loss, BPR_loss
    from spamosaic.train_utils import set_seeds, train_model
    from topola_D_and_S import full_topola_distance

    set_seeds(1234)
    features = torch.randn(8, 3, dtype=torch.float64, device=device, requires_grad=True)
    gram = features @ features.T
    expected = torch.linalg.solve(gram + 10 * torch.eye(8, dtype=torch.float64, device=device), gram)
    torch.testing.assert_close(full_topola_distance(features, 10), expected)
    torch.autograd.gradcheck(lambda x: full_topola_distance(x, 10), (features,))
    for n_mods in (2, 3):
        scores = torch.randn(n_mods * 4, n_mods * 4, dtype=torch.float64, device=device)
        scores = (scores + scores.T) / 2
        terms = []
        for a, b in itertools.combinations(range(n_mods), 2):
            for i in range(4):
                for j in range(4):
                    if i != j:
                        terms.append(F.softplus(scores[a*4+i, b*4+j] - scores[a*4+i, b*4+i]))
                        terms.append(F.softplus(scores[b*4+i, a*4+j] - scores[b*4+i, a*4+i]))
        torch.testing.assert_close(BPR_loss(4, n_mods).to(device)(scores), torch.stack(terms).mean())
        modalities = [str(i) for i in range(n_mods)]
        splits = {m: [8] + [6 if j == i else 0 for j in range(n_mods)] for i, m in enumerate(modalities)}
        attributes = {m: torch.randn(14, 5, device=device) for m in modalities}
        row = torch.arange(14, device=device)
        neighbor = torch.cat(((torch.arange(8, device=device) + 1) % 8,
                              8 + (torch.arange(6, device=device) + 1) % 6))
        edges = {m: torch.stack((torch.cat((row, neighbor)), torch.cat((neighbor, row)))) for m in modalities}
        weights = {m: torch.ones(28, device=device) for m in modalities}
        edge_classes = {m: torch.cat((torch.zeros(8, device=device), torch.full((6,), i+1, device=device))).repeat(2)
                        for i, m in enumerate(modalities)}
        meta = {0: (True, 8, 4, 2, n_mods, modalities)}
        meta.update({i+1: (False, [m]) for i, m in enumerate(modalities)})
        cases = list(VARIANTS.values()) + ([('dot', 'ce')] if n_mods == 2 else [])
        for score_type, loss_type in cases:
            set_seeds(1234)
            models = {m: WLGCN(5, 3, K=1, hidden_size=8, dropout=0).to(device) for m in modalities}
            before = models['0'].fc2.weight.detach().clone()
            optimizers = {m: torch.optim.Adam(models[m].parameters(), lr=.01) for m in modalities}
            criterion = ({0: (BPR_loss(4, n_mods) if loss_type == 'bpr' else CL_loss(4, n_mods)).to(device)}
                         if loss_type != 'ce' else torch.nn.CrossEntropyLoss())
            models, alignment, reconstruction = train_model(
                models, optimizers, criterion, torch.nn.MSELoss(), loss_type, 0.,
                attributes, edges, weights, edge_classes, meta, splits, .1, 2, device,
                score_type=score_type, lambda_topola=10.)
            if not torch.isfinite(torch.tensor(alignment + reconstruction)).all():
                raise RuntimeError(f'Nonfinite loss: {n_mods}, {score_type}, {loss_type}')
            for model in models.values():
                for parameter in model.parameters():
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError('Nonfinite parameter gradient')
            if torch.equal(before, models['0'].fc2.weight):
                raise RuntimeError('Encoder did not update')
            print(f'PASS: {n_mods} modalities, {score_type}, {loss_type}', flush=True)
    print('PASS: QR equivalence, gradcheck, BPR reference, and all short training runs', flush=True)


if __name__ == '__main__':
    main()
