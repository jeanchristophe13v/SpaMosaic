#!/usr/bin/env python
"""exp009: 标签统一后重算 exp005 的基线(训练与 exp005 完全相同,只改评估)。

相对 exp005 run_exp005.py 的改动只有三处:
  1. compute_scores 内按上游 reproduce/code/compute_metrics/run_ARI.ipynb 的 labd 统一 Lymph/Tonsil 标签;
  2. 同时输出未统一标签的 ARI(all)/F1(scope 加后缀 _raw),用于与 exp005 对账;
  3. 保存每个 run 的模态嵌入、合并嵌入、聚类标签、obs 名与 MNN 对到 results/runs/(另存每个数据集的输入特征 meta)。
另加 --tag,使并行进程写各自的 all_metrics_<tag>.csv。

两个 arm:dot(Vanilla)与 topola_cal(D / mean(diag(D)),λ=10,T 沿用 notebook)。
流程与 examples/run_metrics.ipynb 一致;两个 arm 都使用数学等价的稳定版 CL_loss
(原 CL_loss 在尖锐分数下会下溢为 -inf,见 notes/quicktest_topola/insight.md 第 6 节)。

用法(仓库根目录):
    python experiments/exp009_label_fix_rebaseline/scripts/run_exp009.py \
        --device cuda:1 --datasets Simulations Lymph-S1-S2-S3 --tag lymph_sim
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
    """与 CL_loss 数学等价、数值稳定。

    原实现先减去对角(自相似度)再 exp 求和;当某行所有非对角分数都比对角低 87.3 以上时
    求和下溢为 0,loss=inf 且梯度为 NaN。这里用 masked logsumexp,行内常数平移在
    positives 与 log-sum-exp 中抵消,结果相同。bias 在原实现里加在平移后的和上,
    故等价写法是 logaddexp(lse, log(bias) + row_max)。
    """

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
    'Simulations':     dict(input_key='X_pca_be', T=0.1,  n_cluster=5,  cluster_algo='kmeans', gt_key='gt'),
    'Lymph-S1-S2-S3':  dict(input_key='dimred_bc', T=0.01, n_cluster=10, cluster_algo='mclust', gt_key='lab'),
    'Tonsil-S1-S2-S3': dict(input_key='dimred_bc', T=0.01, n_cluster=4,  cluster_algo='mclust', gt_key='lab'),
}
N_SEC = 3
VARIANTS = ('dot', 'topola_cal')

# 逐字抄自 reproduce/code/compute_metrics/run_ARI.ipynb
LABEL_MAPS = {
    'Lymph-S1-S2-S3': {'capsule':'Capsule','cortex':'Cortex','follicle':'Follicle','hilum':'Hilum','medulla Cords':'Medulla cords','medulla cords':'Medulla cords','medulla sinuses':'Medulla sinuses','medulla vessel':'Medulla vessels','medulla vessels':'Medulla vessels','pericapsular adipose tissue':'Pericapsular adipose tissue','subcapsular sinus':'Subcapsular sinus','trabeculae':'Trabeculae'},
    'Tonsil-S1-S2-S3': {'connective & epithelial tissue':'Connective and epithelial tissue','connective and epithelial tissue':'Connective and epithelial tissue','germinal Center':'Germinal Center','germinal center':'Germinal Center','lymphoid follicle':'Lymphoid follicle','nan':'Nan','tonsillar parenchyma':'Tonsillar parenchyma'},
}


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


def map_labels(gt, label_map):
    mapped = gt.map(label_map)
    if mapped.isna().any():
        raise ValueError(f'标签映射未覆盖: {sorted(gt[mapped.isna()].unique())}')
    return mapped


def compute_f1(ad_embs, gts, layout, sec_names, out, suffix=''):
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
            out[('F1', f'{sec_names[i]}-{sec_names[j]}{suffix}')] = repro_metrics.LabTransfer(*pair, 'emb', 'gt', knn=10)


def compute_scores(ad_embs, ad_mosaic, raw, layout, P, label_map=None):
    """与 notebook 第 5 节一致,返回 {(metric, scope): value}。

    label_map 不为 None 时,无后缀的 ARI/F1 用统一后的标签,另输出 _raw 版的 ARI(all)/F1。
    """
    sec_names = [f'slice{i+1}' for i in range(N_SEC)]
    sec_rep = [next(layout[m][i] for m in layout if layout[m][i] is not None) for i in range(N_SEC)]

    gts, preds, gts_raw = [], [], []
    for i, ad in enumerate(ad_embs):
        gt = raw[sec_rep[i]].obs.loc[ad.obs_names, P['gt_key']].astype(str)
        keep = gt != 'nan'
        if keep.sum() < len(gt):
            print(f'  {sec_names[i]}: 剔除 {len(gt) - keep.sum()} 个无标签细胞', flush=True)
        gts_raw.append(gt[keep])
        if label_map is not None:
            mapped = map_labels(gt[keep], label_map)
            if (mapped == 'Nan').any():
                raise ValueError(f'{sec_names[i]}: 剔除 nan 后仍有映射为 Nan 的标签')
            gts.append(mapped)
        else:
            gts.append(gt[keep])
        preds.append(ad.obs['cluster'].astype(str)[keep])

    out = {}
    out[('ARI', 'all')] = adjusted_rand_score(pd.concat(gts), pd.concat(preds))
    if label_map is not None:
        out[('ARI', 'all_raw')] = adjusted_rand_score(pd.concat(gts_raw), pd.concat(preds))
    for n, g, p in zip(sec_names, gts, preds):
        out[('ARI', n)] = adjusted_rand_score(g, p)

    compute_f1(ad_embs, gts, layout, sec_names, out)
    if label_map is not None:
        compute_f1(ad_embs, gts_raw, layout, sec_names, out, suffix='_raw')

    for n, ad in zip(sec_names, ad_embs):
        out[('PAS', n)] = repro_metrics.compute_PAS2(ad, 'cluster', spatial_key='spatial')
        out[('CHAOS', n)] = repro_metrics.compute_CHAOS2(ad, 'cluster', spatial_key='spatial')

    ad_lisi = sc.AnnData(np.zeros((ad_mosaic.n_obs, 2)), obs=ad_mosaic.obs[['src']].copy())
    ad_lisi.obsm['emb'] = ad_mosaic.obsm['merged_emb']
    out[('iLISI', 'all')] = repro_metrics.iLISI(ad_lisi, 'src', 'emb')
    return out


def save_meta(dataset, input_dict, raw, layout, out_dir):
    """每个数据集一次:各模态各切片的输入特征、obs 名、原始标签(诊断脚本只读 runs/)。"""
    P = PARAMS[dataset]
    meta = {}
    for m, ads in input_dict.items():
        for bi, ad in enumerate(ads):
            if ad is None:
                continue
            key = f'{m}_b{bi}'
            meta[f'feat_{key}'] = np.asarray(ad.obsm[P['input_key']], dtype=np.float32)
            meta[f'names_{key}'] = np.asarray(ad.obs_names, dtype=str)
            meta[f'lab_{key}'] = np.asarray(raw[layout[m][bi]].obs.loc[ad.obs_names, P['gt_key']].astype(str).values, dtype=str)
    path = join(out_dir, 'runs', dataset)
    os.makedirs(path, exist_ok=True)
    np.savez(join(path, 'meta.npz'), **meta)


def save_run(dataset, variant, seed, input_dict, ad_embs, model, out_dir):
    dump = {}
    for m, ads in input_dict.items():
        for bi, ad in enumerate(ads):
            if ad is not None:
                dump[f'emb_{m}_b{bi}'] = np.asarray(ad.obsm['emb'], dtype=np.float32)
                dump[f'names_{m}_b{bi}'] = np.asarray(ad.obs_names, dtype=str)
    for i, ad in enumerate(ad_embs):
        dump[f'merged_s{i}'] = np.asarray(ad.obsm['merged_emb'], dtype=np.float32)
        dump[f'names_s{i}'] = np.asarray(ad.obs_names, dtype=str)
        dump[f'cluster_s{i}'] = np.asarray(ad.obs['cluster'].astype(str).values, dtype=str)
    for m, pairs in model.mod_mnn_set.items():
        dump[f'mnn_{m}'] = np.asarray(sorted(pairs), dtype=str).reshape(-1, 2)
    run_dir = join(out_dir, 'runs', dataset, variant, f'seed{seed}')
    os.makedirs(run_dir, exist_ok=True)
    np.savez(join(run_dir, 'run.npz'), **dump)


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
        save_run(dataset, variant, args.seed, input_dict, ad_embs, model, abspath(args.out_dir))
        scores = compute_scores(ad_embs, ad_mosaic, raw, layout, P, label_map=LABEL_MAPS.get(dataset))
        return scores, train_seconds, diag
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
    parser.add_argument('--tag', default='all')
    args = parser.parse_args()

    out_dir = abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(join(out_dir, f'config_{args.tag}.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    rows = []
    for dataset in args.datasets:
        print(f'\n=== {dataset}: 加载 + 预处理 ===', flush=True)
        input_dict, raw, layout = load_inputs(dataset, args)
        save_meta(dataset, input_dict, raw, layout, out_dir)
        for seed in args.seeds:
            for variant in args.variants:
                print(f'\n--- {dataset} / seed {seed} / {variant} ---', flush=True)
                args.seed = seed
                scores, train_seconds, diag = run_arm(dataset, variant, input_dict, raw, layout, args)
                for (metric, scope), value in scores.items():
                    rows.append(dict(dataset=dataset, variant=variant, seed=seed, metric=metric, scope=scope,
                                     value=float(value), epochs=args.epochs,
                                     T=PARAMS[dataset]['T'], lambda_topola=args.lambda_topola,
                                     train_seconds=round(train_seconds, 1),
                                     **{k: diag.get(k) for k in ('n_kernel', 'diag_mean', 'off_mean', 'off_std', 'off_max')}))
                pd.DataFrame(rows).to_csv(join(out_dir, f'all_metrics_{args.tag}.csv'), index=False)
                print('  scores:', {f'{m}/{s}': round(v, 4) for (m, s), v in scores.items()}, flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del input_dict, raw, layout
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(join(out_dir, f'all_metrics_{args.tag}.csv'), index=False)
    summ = (df.groupby(['dataset', 'metric', 'scope', 'variant'])['value']
              .agg(['mean', 'std', 'count']).reset_index())
    summ.to_csv(join(out_dir, f'summary_{args.tag}.csv'), index=False)
    print('\n=== 汇总 ===', flush=True)
    print(summ.round(4).to_string(), flush=True)


if __name__ == '__main__':
    main()
