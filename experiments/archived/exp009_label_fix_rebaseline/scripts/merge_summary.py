"""合并并行进程的 all_metrics_<tag>.csv,写 all_metrics/summary/paired,并与 exp005 逐 run 对账(H_fix 自检)。

paired 部分沿用 exp005 的 paired_summary.py(逐 seed 配对差 topola_cal - dot)。
"""
import glob
from os.path import abspath, dirname, join

import numpy as np
import pandas as pd

HERE = dirname(abspath(__file__))
res = abspath(join(HERE, '..', 'results'))
exp005 = abspath(join(HERE, '..', '..', 'exp005_topola_cal_multiseed', 'results', 'all_metrics.csv'))

parts = sorted(p for p in glob.glob(join(res, 'all_metrics_*.csv')))
df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
key = ['dataset', 'variant', 'seed', 'metric', 'scope']
if df.duplicated(key).any():
    raise RuntimeError('合并后存在重复 run')
df = df.sort_values(key).reset_index(drop=True)
df.to_csv(join(res, 'all_metrics.csv'), index=False)
print('合并:', [p.split('/')[-1] for p in parts], '行数', len(df))

summ = (df.groupby(['dataset', 'metric', 'scope', 'variant'])['value']
          .agg(['mean', 'std', 'count']).reset_index())
summ.to_csv(join(res, 'summary.csv'), index=False)

wide = df.pivot_table(index=['dataset', 'metric', 'scope', 'seed'], columns='variant', values='value').reset_index()
wide['diff'] = wide['topola_cal'] - wide['dot']
g = wide.groupby(['dataset', 'metric', 'scope'])
out = g.agg(dot_mean=('dot', 'mean'), dot_std=('dot', 'std'),
            cal_mean=('topola_cal', 'mean'), cal_std=('topola_cal', 'std'),
            diff_mean=('diff', 'mean'), diff_std=('diff', 'std'),
            n_pos=('diff', lambda x: int((x > 0).sum())), n_neg=('diff', lambda x: int((x < 0).sum())),
            n=('diff', 'count')).reset_index()
out.to_csv(join(res, 'paired.csv'), index=False)
print(out[out.metric.isin(['ARI', 'F1', 'iLISI']) & out.scope.str.match(r'^(all|slice2-slice3)')].round(4).to_string(index=False))

# H_fix 自检:exp009 的 (metric, scope) 映射到 exp005 的 (metric, scope)
ref = pd.read_csv(exp005)
real = df.dataset != 'Simulations'
chk = df.copy()
chk['ref_scope'] = chk['scope']
chk.loc[real & chk.scope.str.endswith('_raw'), 'ref_scope'] = chk.loc[real & chk.scope.str.endswith('_raw'), 'scope'].str[:-4]
# 真实数据的无后缀 ARI(all)/F1 是统一标签后的新值,不参与对账
fixed = real & (((chk.metric == 'ARI') & (chk.scope == 'all')) | ((chk.metric == 'F1') & ~chk.scope.str.endswith('_raw')))
chk = chk[~fixed]
m = chk.merge(ref[key + ['value']].rename(columns={'scope': 'ref_scope', 'value': 'exp005'}),
              on=['dataset', 'variant', 'seed', 'metric', 'ref_scope'], how='left')
if m.exp005.isna().any():
    raise RuntimeError(f'exp005 中缺少对应 run: {m[m.exp005.isna()][key].head()}')
m['absdiff'] = (m['value'] - m['exp005']).abs()
m.to_csv(join(res, 'selfcheck_vs_exp005.csv'), index=False)
n_ref = len(ref)
print(f'\n=== H_fix 自检:对账 {len(m)} 项(exp005 共 {n_ref} 项)===')
print(m.groupby(['dataset', 'metric'])['absdiff'].max().unstack().to_string())
bad = m[m.absdiff > 1e-3]
print(f'max |diff| = {m.absdiff.max():.3e};超过 1e-3 的项数 = {len(bad)}')
if len(bad):
    print(bad[key + ['value', 'exp005', 'absdiff']].to_string(index=False))

# 均值层面的对照(逐 run 不可复现时,看 5 seed 均值差相对 exp005 seed 间 std 的大小)
mm = (m.groupby(['dataset', 'variant', 'metric', 'ref_scope'])
        .agg(exp009_mean=('value', 'mean'), exp005_mean=('exp005', 'mean'), exp005_std=('exp005', 'std'),
             max_absdiff=('absdiff', 'max')).reset_index())
mm['mean_diff'] = mm['exp009_mean'] - mm['exp005_mean']
mm['mean_diff_over_std'] = mm['mean_diff'] / mm['exp005_std']
mm.to_csv(join(res, 'selfcheck_means.csv'), index=False)
print('\n=== 5 seed 均值对照(exp009 - exp005)===')
print(mm.round(4).to_string(index=False))
