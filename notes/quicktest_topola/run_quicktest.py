#!/usr/bin/env python
"""快速验证:TopoLa 分数 + 尺度校准 vs Vanilla SpaMosaic(余弦分数)。

非正式实验,不写入 experiments/。流程与 examples/run_metrics.ipynb 一致:
同一份预处理输入,依次训练 dot / topola_cal / topola_raw 三个 arm,计算
ARI / F1 / PAS / CHAOS / iLISI。

尺度校准(仅 topola_cal):D / mean(diag(D)),使 TopoLa 的"自相似"回到 1,
与余弦同量级,从而沿用 notebook 里的 T,不需要为 TopoLa 重新调温度。
校准标量 detach,只当作常数缩放。

用法:
    conda run --no-capture-output -n spamosaic-env python \
        notes/quicktest_topola/run_quicktest.py --device cuda:5 \
        --datasets Simulations Lymph-S1-S2-S3 Tonsil-S1-S2-S3 \
        --variants dot topola_cal topola_raw
"""
import argparse
import gc
import json
import os
import sys
import time
from os.path import abspath, dirname, join

REPO_ROOT = abspath(join(dirname(__file__), '..', '..'))
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
from sklearn.metrics import adjusted_rand_score

import spamosaic.framework as fw
import spamosaic.train_utils as tu
import spamosaic.utils as utls
from spamosaic.framework import SpaMosaic
from spamosaic.loss import CL_loss
from spamosaic.preprocessing import ADT_preprocess, RNA_preprocess
from spamosaic.topola import full_topola_distance as _raw_topola
from spamosaic.train_utils import set_seeds
import metrics as repro_metrics


class StableCL_loss(CL_loss):
    """数学等价、数值稳定的 CL_loss。

    原实现在 log(sum(exp(simi - row_max))) 中 row_max 是对角自相似度(被 mask 掉),
    当非对角分数整体低于对角超过 ~87*T 时 sum 下溢为 0,log(0) = -inf,梯度变 NaN。
    余弦分数下对角与最优非对角接近所以没触发;校准后的 TopoLa 对角恒为 1,
    训练几轮后即触发(本次 quicktest 第一版在 Lymph 第 4 epoch 崩溃)。
    这里改用 masked logsumexp,数学上与原式相同(row_max 在两项中抵消)。
    """

    def forward(self, simi):
        positives = (simi * self.pos_mask).sum(dim=1) / self.pos_mask.sum(dim=1)
        negatives = torch.logsumexp(simi.masked_fill(self.negatives_mask == 0, -torch.inf), dim=1)
        if self.bias:
            negatives = torch.logaddexp(
                negatives, torch.log(torch.tensor(float(self.bias), device=simi.device)))
        return -(positives - negatives).mean()


fw.CL_loss = StableCL_loss
print('[setup] framework.CL_loss -> StableCL_loss (logsumexp)', flush=True)

PARAMS = {
    'Simulations':     dict(input_key='X_pca_be', T=0.1,  n_cluster=5,  cluster_algo='kmeans', gt_key='gt'),
    'Lymph-S1-S2-S3':  dict(input_key='dimred_bc', T=0.01, n_cluster=10, cluster_algo='mclust', gt_key='lab'),
    'Tonsil-S1-S2-S3': dict(input_key='dimred_bc', T=0.01, n_cluster=4,  cluster_algo='mclust', gt_key='lab'),
}
N_SEC = 3
VARIANTS = ('dot', 'topola_cal', 'topola_raw')


def load_inputs(name, args):
    """与 notebook 第 1-2 节一致:加载 + 预处理,返回 (input_dict, raw, layout)。"""
    if name == 'Simulations':
        data_dir = join(REPO_ROOT, 'data/processed/Simulations', args.sim_setting, args.sim_rep)
        layout = {'rna': ['ad_rna1', 'ad_rna2', None], 'adt': ['ad_adt1', None, 'ad_adt2']}
    else:
        data_dir = join(REPO_ROOT, 'data/processed', name)
        layout = {'rna': ['S1/ad_rna', 'S2/ad_rna', None], 'adt': ['S1/ad_adt', None, 'S3/ad_adt']}

    raw = {}
    for names in layout.values():
        for n in names:
            if n is not None and n not in raw:
                raw[n] = sc.read_h5ad(join(data_dir, n + '.h5ad'))
                print(f'  loaded {n}: {raw[n].shape}', flush=True)

    if name == 'Simulations':
        src_labels = ['mult', 'rna', 'adt']
        for names in layout.values():
            for i, n in enumerate(names):
                if n is not None:
                    raw[n].obs['src'] = src_labels[i]
                    raw[n].X = raw[n].layers['counts'].copy()

    input_dict = {m: [raw[n] if n is not None else None for n in names] for m, names in layout.items()}

    P = PARAMS[name]
    if name == 'Simulations':
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
    return input_dict, raw, layout


def compute_scores(ad_embs, ad_mosaic, raw, layout, P):
    """与 notebook 第 5 节一致,返回 {(metric, scope): value}。"""
    sec_names = [f'slice{i+1}' for i in range(N_SEC)]
    sec_rep = [next(layout[m][i] for m in layout if layout[m][i] is not None) for i in range(N_SEC)]

    gts, preds = [], []
    for i, ad in enumerate(ad_embs):
        gt = raw[sec_rep[i]].obs.loc[ad.obs_names, P['gt_key']].astype(str)
        keep = gt != 'nan'
        if keep.sum() < len(gt):
            print(f'  {sec_names[i]}: 剔除 {len(gt) - keep.sum()} 个无标签细胞', flush=True)
        gts.append(gt[keep])
        preds.append(ad.obs['cluster'].astype(str)[keep])

    out = {}
    out[('ARI', 'all')] = adjusted_rand_score(pd.concat(gts), pd.concat(preds))
    for n, g, p in zip(sec_names, gts, preds):
        out[('ARI', n)] = adjusted_rand_score(g, p)

    sec_mods = [{m for m in layout if layout[m][i] is not None} for i in range(N_SEC)]
    for i in range(N_SEC):
        for j in range(i + 1, N_SEC):
            if sec_mods[i] & sec_mods[j]:
                continue
            pair = []
            for k in (i, j):
                adx = sc.AnnData(np.zeros((len(gts[k]), 2)),
                                 obs=pd.DataFrame({'gt': gts[k].values}, index=gts[k].index))
                adx.obsm['emb'] = ad_embs[k][gts[k].index].obsm['merged_emb']
                pair.append(adx)
            out[('F1', f'{sec_names[i]}-{sec_names[j]}')] = repro_metrics.LabTransfer(*pair, 'emb', 'gt', knn=10)

    for n, ad in zip(sec_names, ad_embs):
        out[('PAS', n)] = repro_metrics.compute_PAS2(ad, 'cluster', spatial_key='spatial')
        out[('CHAOS', n)] = repro_metrics.compute_CHAOS2(ad, 'cluster', spatial_key='spatial')

    ad_lisi = sc.AnnData(np.zeros((ad_mosaic.n_obs, 2)), obs=ad_mosaic.obs[['src']].copy())
    ad_lisi.obsm['emb'] = ad_mosaic.obsm['merged_emb']
    out[('iLISI', 'all')] = repro_metrics.iLISI(ad_lisi, 'src', 'emb')
    return out


def run_arm(dataset, variant, input_dict, raw, layout, args):
    P = PARAMS[dataset]
    set_seeds(args.seed)
    diag = {}

    if variant == 'topola_cal':
        def calibrated(features, regularization):
            D = _raw_topola(features, regularization)
            scale = D.diagonal().mean()
            if not diag:
                with torch.no_grad():
                    n = D.shape[0]
                    mask = ~torch.eye(n, dtype=torch.bool, device=D.device)
                    off = D[mask]
                    diag.update(n_kernel=int(n), diag_mean=float(scale),
                                off_mean=float(off.mean()), off_std=float(off.std()),
                                off_max=float(off.max()))
                print(f'  [calib] kernel {diag["n_kernel"]}: diag_mean={diag["diag_mean"]:.3e} '
                      f'off_std={diag["off_std"]:.3e} -> scale 1/{diag["diag_mean"]:.3e}', flush=True)
            return D / scale.detach()

        tu.full_topola_distance = calibrated

    try:
        model = SpaMosaic(
            modBatch_dict=input_dict, input_key=P['input_key'],
            batch_key='src', intra_knns=10, inter_knn_base=10, w_g=0.8,
            seed=args.seed, device=args.device,
        )
        kwargs = {}
        if variant != 'dot':
            kwargs = dict(score_type='topola', lambda_topola=args.lambda_topola)
        started = time.time()
        model.train(net='wlgcn', lr=0.01, T=P['T'], n_epochs=args.epochs, **kwargs)
        train_seconds = time.time() - started

        ad_embs = model.infer_emb(input_dict, emb_key='emb', final_latent_key='merged_emb')
        ad_mosaic = sc.concat(ad_embs)
        ad_mosaic = utls.clustering(ad_mosaic, n_cluster=P['n_cluster'], used_obsm='merged_emb',
                                    algo=P['cluster_algo'], key='cluster')
        utls.split_adata_ob(ad_embs, ad_mosaic, 'obs', 'cluster')
        scores = compute_scores(ad_embs, ad_mosaic, raw, layout, P)
        return scores, train_seconds, diag
    finally:
        if variant == 'topola_cal':
            tu.full_topola_distance = _raw_topola


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', choices=list(PARAMS), default=list(PARAMS))
    parser.add_argument('--variants', nargs='+', choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lambda-topola', type=float, default=10.0)
    parser.add_argument('--sim-setting', default='rna-0.1-12-adt-2.1')
    parser.add_argument('--sim-rep', choices=['1', '2', '3'], default='1')
    parser.add_argument('--tag', default='r1')
    parser.add_argument('--out-dir', default=join(dirname(abspath(__file__)), 'results'))
    args = parser.parse_args()

    out_dir = join(args.out_dir, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(join(out_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    rows = []
    for dataset in args.datasets:
        print(f'\n=== {dataset}: 加载 + 预处理 ===', flush=True)
        input_dict, raw, layout = load_inputs(dataset, args)
        for variant in args.variants:
            print(f'\n--- {dataset} / {variant} ---', flush=True)
            scores, train_seconds, diag = run_arm(dataset, variant, input_dict, raw, layout, args)
            for (metric, scope), value in scores.items():
                rows.append(dict(dataset=dataset, variant=variant, metric=metric, scope=scope,
                                 value=float(value), epochs=args.epochs, seed=args.seed,
                                 T=PARAMS[dataset]['T'], lambda_topola=args.lambda_topola,
                                 train_seconds=round(train_seconds, 1),
                                 **{k: diag.get(k) for k in ('n_kernel', 'diag_mean', 'off_mean', 'off_std', 'off_max')}))
            pd.DataFrame(rows).to_csv(join(out_dir, 'metrics_long.csv'), index=False)
            print('  scores:', {f'{m}/{s}': round(v, 4) for (m, s), v in scores.items()}, flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        del input_dict, raw, layout
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(join(out_dir, 'metrics_long.csv'), index=False)
    wide = df.pivot_table(index=['dataset', 'metric', 'scope'], columns='variant', values='value')
    wide.to_csv(join(out_dir, 'metrics_wide.csv'))
    print('\n=== 汇总(rows=指标/范围, cols=variant) ===', flush=True)
    print(wide.round(4).to_string(), flush=True)


if __name__ == '__main__':
    main()
