#!/usr/bin/env python
"""exp003 diagnostic: measure the eigenvalue scale of H = E E^T on real
training batches, so the TopoLa lambda grid can be placed relative to the
embedding scale.

Motivation: paper appendix E searched lambda in {1e-3 .. 1e-7} with AdaGCL
embeddings. Here rows of E are L2-normalized (WLGCN output, d=out_dim=32),
so trace(H)=n and the nonzero eigenvalues of H equal the eigenvalues of the
32x32 matrix E^T E. If lambda << eig_min, D_lambda is essentially the
projection onto span(E) and all such lambda give nearly identical scores;
the informative range is lambda comparable to the eigenvalues.

Method: run the reference topola_cl training (lambda=10, same dynamics as
exp001) after monkeypatching spamosaic.train_utils.full_topola_distance to
record the exact spectrum of E^T E at every kernel invocation (bridge
mini-batches: n=2*1024 rows mostly; last MB mini-batch smaller).

Output: results/lambda_scale.csv (one row per kernel call) plus a printed
summary with suggested grid anchors.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import torch

import spamosaic.train_utils as tu
from spamosaic.framework import SpaMosaic
from spamosaic.preprocessing import RNA_preprocess, ADT_preprocess, Epigenome_preprocess
from spamosaic.topola import full_topola_distance as original_full_topola
from spamosaic.train_utils import set_seeds

PARAMS = {
    'Simulations': {'input_key': 'X_pca_be', 'T': .1, 'n_cluster': 5},
    'MB-H3K4me3': {'input_key': 'dimred_bc', 'T': .01, 'n_cluster': 7},
}


def load_inputs(dataset, data_root):
    if dataset == 'Simulations':
        data_dir = data_root / dataset / 'rna-0.1-12-adt-2.1' / '1'
        layout = {'rna': ['ad_rna1', 'ad_rna2', None], 'adt': ['ad_adt1', None, 'ad_adt2']}
    else:
        data_dir = data_root / dataset
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
    return {m: [raw[n] if n is not None else None for n in names] for m, names in layout.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--datasets', nargs='+', choices=PARAMS, default=list(PARAMS))
    parser.add_argument('--data-root', type=Path, default=Path(__file__).resolve().parents[3] / 'data' / 'processed')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'results/lambda_scale.csv')
    args = parser.parse_args()

    records = []

    def patched_full_topola(features, regularization):
        with torch.no_grad():
            e = features.detach().cpu().to(torch.float64)
            gram = e.T @ e                      # d x d, PSD; eigs = nonzero eigs of H = EE^T
            eig = torch.linalg.eigvalsh(gram).clamp_min(0)
            eig = eig[eig > 1e-10]
            records.append(dict(n=int(e.shape[0]), d=int(e.shape[1]),
                                trace=float(eig.sum()), mean_eig_n_over_d=float(e.shape[0] / e.shape[1]),
                                eig_min=float(eig.min()), eig_median=float(eig.median()),
                                eig_max=float(eig.max()), lambda_in_call=float(regularization)))
        return original_full_topola(features, regularization)

    tu.full_topola_distance = patched_full_topola

    frames = []
    for dataset in args.datasets:
        set_seeds(1234)
        inputs = load_inputs(dataset, args.data_root)
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
        records.clear()
        model = SpaMosaic(modBatch_dict=inputs, input_key=PARAMS[dataset]['input_key'],
                          batch_key='src', intra_knns=10, inter_knn_base=10,
                          w_g=.8, seed=1234, device=args.device)
        model.train(net='wlgcn', lr=.01, T=PARAMS[dataset]['T'], n_epochs=args.epochs,
                    use_mini_thr=8000, mini_batch_size=1024,
                    loss_type='adapted', score_type='topola', lambda_topola=10.0)
        df = pd.DataFrame(records)
        df.insert(0, 'call', range(1, len(df) + 1))
        df.insert(0, 'dataset', dataset)
        frames.append(df)
        rho = df['eig_max']
        print(f'\n=== {dataset}: {len(df)} kernel calls, n={df.n.iloc[0]}, d={df.d.iloc[0]} ===')
        print(f'rho=eig_max: median={rho.median():.1f} min={rho.min():.1f} max={rho.max():.1f}')
        print(f'eig_median: median={df.eig_median.median():.1f}  eig_min: median={df.eig_min.median():.2f}')
        print(f'sanity trace vs n: {df.trace.iloc[0]:.1f} vs {df.n.iloc[0]}')
        rho_med = float(rho.median())
        for lam in (1e-7, 1e-5, 1e-3, 1e-1, 1, 10, 100):
            print(f'  lambda={lam:>8g}: lambda/rho={lam / rho_med:.4g}  '
                  f'{"~exact projection (lambda<<eig_min)" if lam < float(df.eig_min.median()) else "regularization regime"}')
        del model
        torch.cuda.empty_cache()

    out = pd.concat(frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f'\nWrote {args.output}')


if __name__ == '__main__':
    main()
