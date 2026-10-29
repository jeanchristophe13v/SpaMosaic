"""合并两个分片,并与 exp005 的 dot / topola_cal(λ=10)逐 seed 配对。写 results/all_metrics.csv、paired_vs_dot.csv、means_by_lambda.csv。"""
from glob import glob
from os.path import abspath, dirname, join

import pandas as pd

here = abspath(join(dirname(__file__), '..'))
new = pd.concat([pd.read_csv(f) for f in sorted(glob(join(here, 'results', 'all_metrics_*.csv')))
                 if not f.endswith('all_metrics_all.csv')])
new.to_csv(join(here, 'results', 'all_metrics.csv'), index=False)
old = pd.read_csv(join(here, '..', 'exp005_topola_cal_multiseed', 'results', 'all_metrics.csv'))
old['arm'] = old['variant'].map({'dot': 'dot', 'topola_cal': 'lam10'})
new['arm'] = 'lam' + new['lambda_topola'].map(lambda x: f'{x:g}')
df = pd.concat([old, new])
w = df.pivot_table(index=['dataset', 'metric', 'scope', 'seed'], columns='arm', values='value').reset_index()
lams = ['lam1', 'lam10', 'lam100', 'lam1000', 'lam10000', 'lam100000']
means = w.groupby(['dataset', 'metric', 'scope'])[['dot'] + lams].mean().reset_index()
means.to_csv(join(here, 'results', 'means_by_lambda.csv'), index=False)
rows = []
for l in lams:
    w['d'] = w[l] - w['dot']
    t = w.groupby(['dataset', 'metric', 'scope'])['d'].agg(
        diff_mean='mean', diff_std='std', n_pos=lambda x: int((x > 0).sum()), n_neg=lambda x: int((x < 0).sum())).reset_index()
    t.insert(3, 'arm', l)
    rows.append(t)
p = pd.concat(rows)
p.to_csv(join(here, 'results', 'paired_vs_dot.csv'), index=False)
sel = lambda d: d[d.scope.isin(['all', 'slice2-slice3']) & d.metric.isin(['ARI', 'F1', 'iLISI'])]
print(sel(means).round(4).to_string(index=False))
print(sel(p).round(4).to_string(index=False))
