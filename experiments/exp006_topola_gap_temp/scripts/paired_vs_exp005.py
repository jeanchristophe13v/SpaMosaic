"""exp006 的 topola_cal_gap 与 exp005 的 dot / topola_cal 逐 seed 配对,写 results/paired_vs_exp005.csv。"""
from os.path import abspath, dirname, join

import pandas as pd

here = abspath(join(dirname(__file__), '..'))
a = pd.read_csv(join(here, '..', 'exp005_topola_cal_multiseed', 'results', 'all_metrics.csv'))
b = pd.read_csv(join(here, 'results', 'all_metrics.csv'))
df = pd.concat([a, b])
w = df.pivot_table(index=['dataset', 'metric', 'scope', 'seed'], columns='variant', values='value').reset_index()
rows = []
for ref in ('topola_cal', 'dot'):
    w['d'] = w['topola_cal_gap'] - w[ref]
    g = w.groupby(['dataset', 'metric', 'scope'])['d']
    t = g.agg(diff_mean='mean', diff_std='std', n_pos=lambda x: int((x > 0).sum()),
              n_neg=lambda x: int((x < 0).sum())).reset_index()
    t.insert(3, 'vs', ref)
    rows.append(t)
    m = w.groupby(['dataset', 'metric', 'scope'])[['dot', 'topola_cal', 'topola_cal_gap']].mean().reset_index()
out = pd.concat(rows)
out.to_csv(join(here, 'results', 'paired_vs_exp005.csv'), index=False)
m.to_csv(join(here, 'results', 'means_3arms.csv'), index=False)
key = out[out.scope.isin(['all', 'slice2-slice3']) & out.metric.isin(['ARI', 'F1', 'iLISI'])]
print(key.round(4).to_string(index=False))
print(m[m.scope.isin(['all', 'slice2-slice3']) & m.metric.isin(['ARI', 'F1', 'iLISI'])].round(4).to_string(index=False))
