#!/usr/bin/env python
"""exp008 诊断分析:读取 run_exp008.py 保存的嵌入与输入特征,输出三部分结果。

1. rank_*:跨模态真实配对的排序质量(余弦 vs TopoLa,参考集合 × λ)。
2. mnn_*:输入特征上同模态跨切片互近邻对的标签一致率。
3. geometry_*:dot 与 topola_cal 训练嵌入的几何量。

用法(仓库根目录):
    python experiments/exp008_topola_rank_diag/scripts/analyze_exp008.py
"""
import argparse
import glob
import os
from os.path import abspath, dirname, join, basename

import numpy as np
import pandas as pd
import torch

torch.set_default_dtype(torch.float64)
REPO_ROOT = abspath(join(dirname(__file__), '..', '..', '..'))
ALPHAS = [1e-4, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0]
REFS = ['pair', 'bridge', 'all']
KINDS = ['topo', 'wcos']
PAIRED = ('Simulations', 'Lymph-S1-S2-S2', 'Tonsil-S1-S2-S2')
REAL_S123 = ('Lymph-S1-S2-S3', 'Tonsil-S1-S2-S3')
K_MUTUAL = 10


def t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float64)


def l2n(x):
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def whitener(R, alpha=None, lam=None):
    """返回 X -> X L^{-T},使 (X L^{-T})(Y L^{-T})ᵀ = X (RᵀR+λI)⁻¹ Yᵀ。λ 默认 α·tr(RᵀR)/d。"""
    d = R.shape[1]
    if lam is None:
        lam = alpha * float((R * R).sum()) / d
    S = R.T @ R + lam * torch.eye(d)
    L = torch.linalg.cholesky(S)
    return (lambda X: torch.linalg.solve_triangular(L, X.T, upper=False).T), lam


def score(Q, C, kind, R=None, alpha=None, lam=None):
    if kind == 'cos':
        return l2n(Q) @ l2n(C).T, float('nan')
    W, lam = whitener(R, alpha, lam)
    Qw, Cw = W(Q), W(C)
    if kind == 'wcos':
        Qw, Cw = l2n(Qw), l2n(Cw)
    return Qw @ Cw.T, lam


def mutual_knn(S, k):
    n1, n2 = S.shape
    kr, kc = min(k, n2), min(k, n1)
    A = torch.zeros(n1, n2, dtype=torch.bool)
    A.scatter_(1, S.topk(kr, dim=1).indices, True)
    B = torch.zeros(n1, n2, dtype=torch.bool)
    B.scatter_(0, S.topk(kc, dim=0).indices, True)
    return A & B


def label_agree(M, lab1, lab2):
    i, j = torch.nonzero(M, as_tuple=True)
    i, j = i.numpy(), j.numpy()
    keep = (lab1[i] != 'nan') & (lab2[j] != 'nan')
    if keep.sum() == 0:
        return float('nan'), 0
    return float((lab1[i][keep] == lab2[j][keep]).mean()), int(len(i))


def chance_agree(lab1, lab2):
    a = pd.Series(lab1[lab1 != 'nan']).value_counts(normalize=True)
    b = pd.Series(lab2[lab2 != 'nan']).value_counts(normalize=True)
    return float((a * b.reindex(a.index).fillna(0)).sum())


def rank_metrics(S, lab_q=None, lab_c=None):
    n = S.shape[0]
    d = S.diagonal()
    above_r = (S > d[:, None]).sum(1).double()
    above_c = (S > d[None, :]).sum(0).double()
    out = dict(
        FOSCTTM=float(((above_r / n).mean() + (above_c / n).mean()) / 2),
        MRR=float(((1 / (above_r + 1)).mean() + (1 / (above_c + 1)).mean()) / 2),
        Hit1=float(((above_r == 0).double().mean() + (above_c == 0).double().mean()) / 2),
        Hit10=float(((above_r < 10).double().mean() + (above_c < 10).double().mean()) / 2),
    )
    M = mutual_knn(S, K_MUTUAL)
    n_pairs = int(M.sum())
    out['mutual_pairs'] = n_pairs
    out['mutual_true_frac'] = float(M.diagonal().sum()) / max(1, n_pairs)
    if lab_q is not None:
        out['mutual_label_agree'], _ = label_agree(M, lab_q, lab_c)
    return out


def settings():
    yield ('cos', 'none', 'none')
    for ref in REFS:
        for kind in KINDS:
            for a in ALPHAS:
                yield (kind, ref, a)
    for kind in KINDS:
        yield (kind, 'bridge', 'train_lam10')


def pair_key(dataset, name):
    if dataset == 'Simulations':
        return name.split('-', 1)[1]
    return name.split('-', 2)[2]


def part_rank(res_dir, rows):
    for dataset in PAIRED:
        meta_p = join(res_dir, 'cache', dataset, 'meta.npz')
        if not os.path.exists(meta_p):
            continue
        meta = np.load(meta_p)
        nq, nc = meta['names_rna_b1'], meta['names_adt_b2']
        kq = [pair_key(dataset, x) for x in nq]
        kc = [pair_key(dataset, x) for x in nc]
        if kq != kc:
            raise RuntimeError(f'{dataset}: slice2/slice3 配对顺序不一致')
        use_lab = dataset != 'Simulations'
        lab_q, lab_c = (meta['lab_rna_b1'], meta['lab_adt_b2']) if use_lab else (None, None)
        for run in sorted(glob.glob(join(res_dir, 'runs', dataset, '*', 'seed*', 'emb.npz'))):
            arm = basename(dirname(dirname(run)))
            seed = int(basename(dirname(run))[4:])
            e = np.load(run)
            Q, C = t(e['emb_rna_b1']), t(e['emb_adt_b2'])
            refs = dict(pair=torch.cat([Q, C]),
                        bridge=torch.cat([t(e['emb_rna_b0']), t(e['emb_adt_b0'])]),
                        all=torch.cat([t(e[k]) for k in ('emb_rna_b0', 'emb_adt_b0', 'emb_rna_b1', 'emb_adt_b2')]))
            for kind, ref, a in settings():
                if a == 'train_lam10':
                    S, lam = score(Q, C, kind, refs[ref], lam=10.0)
                elif kind == 'cos':
                    S, lam = score(Q, C, 'cos')
                else:
                    S, lam = score(Q, C, kind, refs[ref], alpha=a)
                m = rank_metrics(S, lab_q, lab_c)
                for k, v in m.items():
                    rows.append(dict(dataset=dataset, arm=arm, seed=seed, score=kind, ref=ref, alpha=str(a),
                                     lam=lam, metric=k, value=v))
            print(f'  rank {dataset} {arm} seed{seed} done', flush=True)


def part_mnn(res_dir, rows):
    for meta_p in sorted(glob.glob(join(res_dir, 'cache', '*', 'meta.npz'))):
        dataset = basename(dirname(meta_p))
        meta = np.load(meta_p)
        names_all = np.concatenate([meta[k] for k in meta.files if k.startswith('names_')])
        labs_all = np.concatenate([meta['lab_' + k[6:]] for k in meta.files if k.startswith('names_')])
        name2lab = dict(zip(names_all, labs_all))
        for m, other in (('rna', 'b1'), ('adt', 'b2')):
            X1, X2 = l2n(t(meta[f'feat_{m}_b0'])), l2n(t(meta[f'feat_{m}_{other}']))
            l1, l2 = meta[f'lab_{m}_b0'], meta[f'lab_{m}_{other}']
            ch = chance_agree(l1, l2)
            base = dict(dataset=dataset, modality=m, chance=ch, n1=len(l1), n2=len(l2))
            R = torch.cat([X1, X2])
            for kind, ref, a in settings():
                if ref in ('pair', 'all') or a == 'train_lam10':
                    continue  # 输入特征只有「两切片合并」一种参考,记为 ref=both
                if kind == 'cos':
                    S, lam = score(X1, X2, 'cos')
                else:
                    S, lam = score(X1, X2, kind, R, alpha=a)
                agree, n = label_agree(mutual_knn(S, K_MUTUAL), l1, l2)
                rows.append(dict(**base, source='exact_mutual_knn', score=kind,
                                 ref='both' if kind != 'cos' else 'none', alpha=str(a) if kind != 'cos' else 'none',
                                 lam=lam, seed=-1, arm='-', n_pairs=n, label_agree=agree))
            for run in sorted(glob.glob(join(res_dir, 'runs', dataset, '*', 'seed*', 'emb.npz'))):
                e = np.load(run)
                pairs = e[f'mnn_{m}']
                if len(pairs) == 0:
                    continue
                la = np.array([name2lab[p] for p in pairs[:, 0]])
                lb = np.array([name2lab[p] for p in pairs[:, 1]])
                keep = (la != 'nan') & (lb != 'nan')
                rows.append(dict(**base, source='pipeline_annoy', score='annoy_l1', ref='none', alpha='none',
                                 lam=float('nan'), seed=int(basename(dirname(run))[4:]),
                                 arm=basename(dirname(dirname(run))), n_pairs=len(pairs),
                                 label_agree=float((la[keep] == lb[keep]).mean())))
        print(f'  mnn {dataset} done', flush=True)


def part_geometry(res_dir, rows):
    for run in sorted(glob.glob(join(res_dir, 'runs', '*', '*', 'seed*', 'emb.npz'))):
        seed = int(basename(dirname(run))[4:])
        arm = basename(dirname(dirname(run)))
        dataset = basename(dirname(dirname(dirname(run))))
        e = np.load(run)
        r0, a0, r1, a2 = (t(e[k]) for k in ('emb_rna_b0', 'emb_adt_b0', 'emb_rna_b1', 'emb_adt_b2'))
        b0 = (r0 + a0) / 2
        M = torch.cat([b0, r1, a2])
        Mc = M - M.mean(0)
        ev, V = torch.linalg.eigh(Mc.T @ Mc / (len(M) - 1))
        ev, V = ev.flip(0).clamp_min(0), V.flip(1)
        p = ev / ev.sum()
        g12 = r1.mean(0) - a2.mean(0)
        res = r0 - a0
        rho = ((res @ V) ** 2).mean(0) / ev.clamp_min(1e-12)
        g_mod = torch.cat([r0, r1]).mean(0) - torch.cat([a0, a2]).mean(0)
        out = dict(
            eff_rank=float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum())),
            top3_var_share=float(p[:3].sum()),
            gap_s2_s3=float(g12.norm()),
            gap_s2_s3_top3_share=float(((V[:, :3].T @ g12) ** 2).sum() / (g12 ** 2).sum()),
            gap_s1_s2=float((b0.mean(0) - r1.mean(0)).norm()),
            gap_s1_s3=float((b0.mean(0) - a2.mean(0)).norm()),
            mod_gap_all=float(g_mod.norm()),
            mod_gap_all_top3_share=float(((V[:, :3].T @ g_mod) ** 2).sum() / (g_mod ** 2).sum()),
            bridge_res_ms=float((res ** 2).sum(1).mean()),
            bridge_res_mean_norm=float(res.mean(0).norm()),
            rho_top3=float(rho[:3].mean()),
            rho_rest=float(rho[3:].mean()),
        )
        for k, v in out.items():
            rows.append(dict(dataset=dataset, arm=arm, seed=seed, metric=k, value=v))
    print('  geometry done', flush=True)


def summarize_rank(df, out_dir):
    key = ['dataset', 'arm', 'score', 'ref', 'alpha']
    summ = df.groupby(key + ['metric'])['value'].agg(['mean', 'std']).reset_index()
    summ.to_csv(join(out_dir, 'rank_summary.csv'), index=False)

    f = df[df.metric == 'FOSCTTM']
    cos = f[f.score == 'cos'][['dataset', 'arm', 'seed', 'value']].rename(columns={'value': 'cos'})
    f = f.merge(cos, on=['dataset', 'arm', 'seed'])
    f['rel'] = (f['value'] - f['cos']) / f['cos']
    paired = (f.groupby(key).agg(foscttm=('value', 'mean'), cos=('cos', 'mean'), rel_mean=('rel', 'mean'),
                                 n_better=('rel', lambda x: int((x < 0).sum())), n=('rel', 'size'))
              .reset_index())
    paired.to_csv(join(out_dir, 'rank_foscttm_vs_cos.csv'), index=False)
    return paired


def summarize_geometry(df, out_dir):
    summ = df.groupby(['dataset', 'metric', 'arm'])['value'].agg(['mean', 'std']).reset_index()
    summ.to_csv(join(out_dir, 'geometry_summary.csv'), index=False)
    w = df.pivot_table(index=['dataset', 'metric', 'seed'], columns='arm', values='value').reset_index()
    if {'dot', 'topola_cal'} <= set(w.columns):
        w['diff'] = w['topola_cal'] - w['dot']
        p = (w.groupby(['dataset', 'metric'])
             .agg(dot=('dot', 'mean'), topola_cal=('topola_cal', 'mean'), diff_mean=('diff', 'mean'),
                  n_pos=('diff', lambda x: int((x > 0).sum())), n_neg=('diff', lambda x: int((x < 0).sum())))
             .reset_index())
        p.to_csv(join(out_dir, 'geometry_paired.csv'), index=False)
        return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-dir', default=join(dirname(abspath(__file__)), '..', 'results'))
    parser.add_argument('--threads', type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    res_dir = abspath(args.res_dir)

    rank_rows, mnn_rows, geo_rows = [], [], []
    part_rank(res_dir, rank_rows)
    part_mnn(res_dir, mnn_rows)
    part_geometry(res_dir, geo_rows)

    rank = pd.DataFrame(rank_rows)
    rank.to_csv(join(res_dir, 'rank_all.csv'), index=False)
    mnn = pd.DataFrame(mnn_rows)
    mnn.to_csv(join(res_dir, 'mnn_input.csv'), index=False)
    geo = pd.DataFrame(geo_rows)
    geo.to_csv(join(res_dir, 'geometry_all.csv'), index=False)

    pd.set_option('display.width', 250)
    pd.set_option('display.max_rows', 500)
    paired = summarize_rank(rank, res_dir) if len(rank) else None
    gp = summarize_geometry(geo, res_dir) if len(geo) else None

    if paired is not None:
        print('\n=== FOSCTTM 相对余弦(dot 嵌入)最好的 15 个设置,按数据集均值排序 ===')
        d = paired[paired.arm == 'dot']
        piv = d.pivot_table(index=['score', 'ref', 'alpha'], columns='dataset', values='rel_mean')
        piv['worst'] = piv.max(axis=1)
        print(piv.sort_values('worst').head(15).round(4).to_string())
    if len(mnn):
        print('\n=== MNN 标签一致率(输入特征) ===')
        ex = mnn[mnn.source == 'exact_mutual_knn']
        print(ex.pivot_table(index=['dataset', 'modality'], columns=['score', 'alpha'], values='label_agree').round(3).to_string())
    if gp is not None:
        print('\n=== 几何:topola_cal − dot ===')
        print(gp.round(4).to_string())


if __name__ == '__main__':
    main()
