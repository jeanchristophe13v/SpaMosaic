#!/usr/bin/env python
"""exp008: 训练 dot / topola_cal 并保存嵌入,供 analyze_exp008.py 做排序、MNN 与几何诊断。

派生自 exp005 的 run_exp005.py:训练流程、StableCL_loss、校准 TopoLa 完全相同;
新增 S1-S2-S2 两个数据集;不做聚类(诊断不需要 ARI),只算 iLISI 作为与 exp005 的自检。

用法(仓库根目录):
    python experiments/exp008_topola_rank_diag/scripts/run_exp008.py \
        --device cuda:5 --datasets Lymph-S1-S2-S2 Simulations
"""
import argparse
import gc
import json
import os
import sys
import time
from os.path import abspath, dirname, join

REPO_ROOT = abspath(join(dirname(__file__), '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
METRICS_DIR = join(REPO_ROOT, 'reproduce/code/compute_metrics')
if METRICS_DIR not in sys.path:
    sys.path.insert(0, METRICS_DIR)

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('R_HOME', join(sys.prefix, 'lib', 'R'))

import numpy as np
import pandas as pd
import scanpy as sc
import torch

import spamosaic.framework as fw
import spamosaic.train_utils as tu
from spamosaic.framework import SpaMosaic
from spamosaic.loss import CL_loss
from spamosaic.preprocessing import ADT_preprocess, RNA_preprocess
from spamosaic.topola import full_topola_distance as _raw_topola
from spamosaic.train_utils import set_seeds
import metrics as repro_metrics


class StableCL_loss(CL_loss):
    """与 CL_loss 数学等价、数值稳定(同 exp005)。"""

    def forward(self, simi):
        positives = (simi * self.pos_mask).sum(dim=1) / self.pos_mask.sum(dim=1)
        negatives = torch.logsumexp(simi.masked_fill(self.negatives_mask == 0, -torch.inf), dim=1)
        if self.bias:
            row_max = simi.max(dim=1).values.detach()
            negatives = torch.logaddexp(negatives, torch.log(torch.tensor(float(self.bias), device=simi.device)) + row_max)
        return -(positives - negatives).mean()


fw.CL_loss = StableCL_loss
print('[setup] framework.CL_loss -> StableCL_loss (logsumexp)', flush=True)

PARAMS = {
    'Simulations':     dict(input_key='X_pca_be', T=0.1,  gt_key='gt'),
    'Lymph-S1-S2-S3':  dict(input_key='dimred_bc', T=0.01, gt_key='lab'),
    'Tonsil-S1-S2-S3': dict(input_key='dimred_bc', T=0.01, gt_key='lab'),
    'Lymph-S1-S2-S2':  dict(input_key='dimred_bc', T=0.01, gt_key='lab'),
    'Tonsil-S1-S2-S2': dict(input_key='dimred_bc', T=0.01, gt_key='lab'),
}
LAYOUTS = {
    'sim':  {'rna': ['ad_rna1', 'ad_rna2', None], 'adt': ['ad_adt1', None, 'ad_adt2']},
    's123': {'rna': ['S1/ad_rna', 'S2/ad_rna', None], 'adt': ['S1/ad_adt', None, 'S3/ad_adt']},
    's122': {'rna': ['S1/ad_rna', 'S2-rna/ad_rna', None], 'adt': ['S1/ad_adt', None, 'S2-adt/ad_adt']},
}
VARIANTS = ('dot', 'topola_cal')


def layout_of(name):
    if name == 'Simulations':
        return 'sim'
    return 's122' if name.endswith('S2-S2') else 's123'


def load_inputs(name, args):
    """与 exp005 一致的加载 + 预处理;S1-S2-S2 沿用真实数据的同一套参数。"""
    kind = layout_of(name)
    layout = LAYOUTS[kind]
    if kind == 'sim':
        data_dir = join(REPO_ROOT, 'data/processed/Simulations', args.sim_setting, args.sim_rep)
    else:
        data_dir = join(REPO_ROOT, 'data/processed', name)

    raw = {}
    for names in layout.values():
        for n in names:
            if n is not None and n not in raw:
                raw[n] = sc.read_h5ad(join(data_dir, n + '.h5ad'))
                print(f'  loaded {n}: {raw[n].shape}', flush=True)

    if kind == 'sim':
        src_labels = ['mult', 'rna', 'adt']
        for names in layout.values():
            for i, n in enumerate(names):
                if n is not None:
                    raw[n].obs['src'] = src_labels[i]
                    raw[n].X = raw[n].layers['counts'].copy()

    input_dict = {m: [raw[n] if n is not None else None for n in names] for m, names in layout.items()}

    P = PARAMS[name]
    if kind == 'sim':
        RNA_preprocess(input_dict['rna'], batch_corr=True, favor='scanpy', n_hvg=None,
                       lognorm=False, scale=True, batch_key='src', key=P['input_key'])
        ADT_preprocess(input_dict['adt'], batch_corr=True, favor='scanpy',
                       lognorm=False, scale=True, batch_key='src', key=P['input_key'])
    else:
        RNA_preprocess(input_dict['rna'], batch_corr=True, favor='scanpy', n_hvg=5000,
                       batch_key='src', key=P['input_key'])
        ADT_preprocess(input_dict['adt'], batch_corr=True, batch_key='src', key=P['input_key'])
    for m, ads in input_dict.items():
        print(f'  {m}: {[ad.obsm[P["input_key"]].shape if ad is not None else None for ad in ads]}', flush=True)
    return input_dict


def save_meta(dataset, input_dict, out_dir):
    """每个数据集一次:输入特征、obs 名、标签、src,供分析脚本使用。"""
    P = PARAMS[dataset]
    meta = {}
    for m, ads in input_dict.items():
        for bi, ad in enumerate(ads):
            if ad is None:
                continue
            key = f'{m}_b{bi}'
            meta[f'feat_{key}'] = np.asarray(ad.obsm[P['input_key']], dtype=np.float32)
            meta[f'names_{key}'] = np.asarray(ad.obs_names, dtype=str)
            meta[f'lab_{key}'] = ad.obs[P['gt_key']].astype(str).values.astype(str)
            meta[f'src_{key}'] = ad.obs['src'].astype(str).values.astype(str)
    path = join(out_dir, 'cache', dataset)
    os.makedirs(path, exist_ok=True)
    np.savez(join(path, 'meta.npz'), **meta)


def run_arm(dataset, variant, input_dict, args):
    P = PARAMS[dataset]
    set_seeds(args.seed)

    if variant == 'topola_cal':
        def calibrated(features, regularization):
            D = _raw_topola(features, regularization)
            return D / D.diagonal().mean().detach()
        tu.full_topola_distance = calibrated

    try:
        model = SpaMosaic(
            modBatch_dict=input_dict, input_key=P['input_key'],
            batch_key='src', intra_knns=10, inter_knn_base=10, w_g=0.8,
            seed=args.seed, device=args.device,
        )
        kwargs = dict(score_type='topola', lambda_topola=args.lambda_topola) if variant != 'dot' else {}
        started = time.time()
        model.train(net='wlgcn', lr=0.01, T=P['T'], n_epochs=args.epochs, **kwargs)
        train_seconds = time.time() - started

        ad_embs = model.infer_emb(input_dict, emb_key='emb', final_latent_key='merged_emb')
        dump = {}
        for m, ads in input_dict.items():
            for bi, ad in enumerate(ads):
                if ad is not None:
                    dump[f'emb_{m}_b{bi}'] = np.asarray(ad.obsm['emb'], dtype=np.float32)
        for m, pairs in model.mod_mnn_set.items():
            arr = np.asarray(sorted(pairs), dtype=str).reshape(-1, 2)
            dump[f'mnn_{m}'] = arr

        ad_mosaic = sc.concat(ad_embs)
        ad_lisi = sc.AnnData(np.zeros((ad_mosaic.n_obs, 2)), obs=ad_mosaic.obs[['src']].copy())
        ad_lisi.obsm['emb'] = ad_mosaic.obsm['merged_emb']
        ilisi = float(repro_metrics.iLISI(ad_lisi, 'src', 'emb'))

        run_dir = join(args.out_dir, 'runs', dataset, variant, f'seed{args.seed}')
        os.makedirs(run_dir, exist_ok=True)
        np.savez(join(run_dir, 'emb.npz'), **dump)
        return ilisi, train_seconds, {m: len(p) for m, p in model.mod_mnn_set.items()}
    finally:
        if variant == 'topola_cal':
            tu.full_topola_distance = _raw_topola


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=list(PARAMS), default=list(PARAMS))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', nargs='+', type=int, default=[1234, 2024, 2025, 2026, 2027])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lambda-topola', type=float, default=10.0)
    parser.add_argument('--sim-setting', default='rna-0.1-12-adt-2.1')
    parser.add_argument('--sim-rep', choices=['1', '2', '3'], default='1')
    parser.add_argument('--out-dir', default=join(dirname(abspath(__file__)), '..', 'results'))
    parser.add_argument('--tag', default='main', help='区分并行进程的输出文件名')
    args = parser.parse_args()

    args.out_dir = abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(join(args.out_dir, f'config_{args.tag}.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    rows = []
    out_csv = join(args.out_dir, f'run_metrics_{args.tag}.csv')
    for dataset in args.datasets:
        print(f'\n=== {dataset}: 加载 + 预处理 ===', flush=True)
        input_dict = load_inputs(dataset, args)
        save_meta(dataset, input_dict, args.out_dir)
        for seed in args.seeds:
            for variant in args.variants:
                print(f'\n--- {dataset} / seed {seed} / {variant} ---', flush=True)
                args.seed = seed
                ilisi, train_seconds, n_mnn = run_arm(dataset, variant, input_dict, args)
                rows.append(dict(dataset=dataset, variant=variant, seed=seed, iLISI=ilisi,
                                 train_seconds=round(train_seconds, 1), epochs=args.epochs,
                                 T=PARAMS[dataset]['T'], lambda_topola=args.lambda_topola,
                                 **{f'n_mnn_{m}': n for m, n in n_mnn.items()}))
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                print(f'  iLISI={ilisi:.4f} train={train_seconds:.1f}s mnn={n_mnn}', flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del input_dict
        gc.collect()
    print('\n=== done ===', flush=True)


if __name__ == '__main__':
    main()
