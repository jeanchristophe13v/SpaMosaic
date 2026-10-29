#!/usr/bin/env python
"""exp008 判定汇总:读 analyze_exp008.py 的输出,按 README 判定标准逐条检查,写 results/verdict_*.csv。"""
from os.path import abspath, dirname, join
import numpy as np
import pandas as pd

HERE = dirname(abspath(__file__))
RES = join(HERE, '..', 'results')
EXP005 = join(HERE, '..', '..', 'exp005_topola_cal_multiseed', 'results', 'summary.csv')
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 500)

# 1. iLISI 自检
m = pd.concat([pd.read_csv(join(RES, f)) for f in ('run_metrics_lymph_sim.csv', 'run_metrics_tonsil.csv')])
il = m.groupby(['dataset', 'variant'])['iLISI'].agg(['mean', 'std', 'count']).reset_index()
e5 = pd.read_csv(EXP005)
e5 = e5[(e5.metric == 'iLISI') & (e5.scope == 'all')][['dataset', 'variant', 'mean', 'std']]
il = il.merge(e5, on=['dataset', 'variant'], how='left', suffixes=('', '_exp005'))
il['diff_vs_exp005'] = il['mean'] - il['mean_exp005']
w = m.pivot_table(index=['dataset', 'seed'], columns='variant', values='iLISI').reset_index()
w['d'] = w['topola_cal'] - w['dot']
pd_il = w.groupby('dataset')['d'].agg(['mean', lambda x: int((x < 0).sum())]).rename(columns={'<lambda_0>': 'n_topo_lower'})
print('=== iLISI(本实验 vs exp005) ===')
print(il.round(4).to_string())
print(pd_il.round(4).to_string())
il.to_csv(join(RES, 'verdict_ilisi.csv'), index=False)

# 2. H_rank
p = pd.read_csv(join(RES, 'rank_foscttm_vs_cos.csv'), dtype={'alpha': str})
p['alpha'] = p['alpha'].fillna('none')
print('\n=== 余弦 FOSCTTM 绝对值(两个训练 arm) ===')
print(p[p.score == 'cos'][['dataset', 'arm', 'foscttm']].round(4).to_string())
d = p[p.arm == 'dot']
key = ['score', 'ref', 'alpha']
wide = d.pivot_table(index=key, columns='dataset', values='rel_mean')
nb = d.pivot_table(index=key, columns='dataset', values='n_better')
ds = list(wide.columns)
strong = ((wide <= -0.10) & (nb >= 4)).all(axis=1)
n_ge3 = (wide <= -0.03).sum(axis=1)
print('\nH_rank 强标准满足的设置数:', int(strong.sum()))
print('至少 2 个数据集下降 ≥3% 的设置数:', int((n_ge3 >= 2).sum()))
sel = wide.join(nb, rsuffix='_nbetter')
print('\n=== 自检 α=100 与训练核(train_lam10)===')
print(sel[sel.index.get_level_values('alpha').isin(['100.0', 'train_lam10'])].round(4).to_string())
print('\n=== 每个数据集最好的 5 个设置 ===')
for c in ds:
    print(f'-- {c}')
    print(sel.sort_values(c).head(5)[[c, c + '_nbetter']].round(4).to_string())
print('\n=== 至少 2 个数据集下降 ≥3% 的设置 ===')
print(sel[n_ge3 >= 2].round(4).to_string())

# 其他排序指标:dot 嵌入上的 cos 与几个代表设置
s = pd.read_csv(join(RES, 'rank_summary.csv'), dtype={'alpha': str})
s['alpha'] = s['alpha'].fillna('none')
rep = [('cos', 'none', 'none'), ('topo', 'bridge', '100.0'), ('wcos', 'bridge', '10.0'), ('wcos', 'bridge', '3.0'),
       ('topo', 'bridge', '1.0'), ('topo', 'bridge', 'train_lam10'), ('topo', 'pair', '0.0001')]
s['k'] = list(zip(s.score, s.ref, s.alpha))
r = s[s.k.isin(rep)].pivot_table(index=['dataset', 'arm', 'score', 'ref', 'alpha'], columns='metric', values='mean')
print('\n=== 代表设置的排序指标(均值) ===')
print(r[['FOSCTTM', 'MRR', 'Hit1', 'Hit10', 'mutual_true_frac', 'mutual_label_agree', 'mutual_pairs']].round(4).to_string())
r.reset_index().to_csv(join(RES, 'verdict_rank_repr.csv'), index=False)
sel.reset_index().to_csv(join(RES, 'verdict_rank_wide.csv'), index=False)

# 3. H_mnn
mn = pd.read_csv(join(RES, 'mnn_input.csv'), dtype={'alpha': str})
ex = mn[mn.source == 'exact_mutual_knn'].copy()
cos = ex[ex.score == 'cos'][['dataset', 'modality', 'n_pairs', 'label_agree']].rename(
    columns={'n_pairs': 'cos_pairs', 'label_agree': 'cos_agree'})
ex = ex[ex.score != 'cos'].merge(cos, on=['dataset', 'modality'])
ex['d_agree'] = ex['label_agree'] - ex['cos_agree']
ex['pair_ratio'] = ex['n_pairs'] / ex['cos_pairs']
ex['ok'] = (ex['d_agree'] >= 0.02) & (ex['pair_ratio'].between(0.7, 1.3))
best = ex.sort_values('d_agree', ascending=False).groupby(['dataset', 'modality']).head(1)
print('\n=== H_mnn:每个组合 TopoLa 相对余弦的最大一致率增益 ===')
print(best[['dataset', 'modality', 'chance', 'cos_pairs', 'cos_agree', 'score', 'alpha', 'n_pairs', 'label_agree',
            'd_agree', 'pair_ratio']].round(3).to_string())
print('满足 H_mnn 的组合(任意设置):')
print(ex[ex.ok].groupby(['dataset', 'modality']).size().to_string())
pipe = mn[mn.source == 'pipeline_annoy'].groupby(['dataset', 'modality']).agg(
    n_pairs=('n_pairs', 'mean'), agree=('label_agree', 'mean'), agree_std=('label_agree', 'std')).reset_index()
print('\n=== SpaMosaic 实际 MNN(annoy)标签一致率 ===')
print(pipe.merge(cos, on=['dataset', 'modality']).round(3).to_string())
ex.to_csv(join(RES, 'verdict_mnn.csv'), index=False)

# 4. H_geo
g = pd.read_csv(join(RES, 'geometry_paired.csv'))
want = {'eff_rank': 'pos', 'gap_s2_s3': 'pos', 'rho_top3': 'pos'}
print('\n=== H_geo ===')
for met, sign in want.items():
    for dsn in ('Lymph-S1-S2-S3', 'Tonsil-S1-S2-S3'):
        row = g[(g.dataset == dsn) & (g.metric == met)].iloc[0]
        n = row.n_pos if sign == 'pos' else row.n_neg
        print(f"{met:12s} {dsn:16s} dot={row['dot']:.4f} topo={row['topola_cal']:.4f} n_same_dir={int(n)}")

# 5. 参考集合谱的各向异性(解释 α=100 自检):λ_max(RᵀR) / (tr(RᵀR)/d),R=桥接两模态嵌入(未中心化)
import glob
rows = []
for f in sorted(glob.glob(join(RES, 'runs', '*', '*', 'seed*', 'emb.npz'))):
    parts = f.split('/')
    e = np.load(f)
    R = np.concatenate([e['emb_rna_b0'], e['emb_adt_b0']]).astype(np.float64)
    ev = np.linalg.eigvalsh(R.T @ R)[::-1]
    mean_ev = ev.sum() / len(ev)
    rows.append(dict(dataset=parts[-4], arm=parts[-3], seed=int(parts[-2][4:]),
                     top1_over_mean=ev[0] / mean_ev, top3_over_mean=ev[:3].mean() / mean_ev,
                     min_over_mean=ev[-1] / mean_ev))
sp = pd.DataFrame(rows)
sp.to_csv(join(RES, 'verdict_spectrum.csv'), index=False)
print('\n=== 桥接参考 RᵀR 谱 / 平均特征值 ===')
print(sp.groupby(['dataset', 'arm'])[['top1_over_mean', 'top3_over_mean', 'min_over_mean']].mean().round(3).to_string())
