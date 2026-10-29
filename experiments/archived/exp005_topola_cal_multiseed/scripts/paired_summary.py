"""逐 seed 配对差(topola_cal - dot)汇总,读取 results/all_metrics.csv,写 results/paired.csv。"""
from os.path import abspath, dirname, join

import pandas as pd

res = abspath(join(dirname(__file__), '..', 'results'))
df = pd.read_csv(join(res, 'all_metrics.csv'))
wide = df.pivot_table(index=['dataset', 'metric', 'scope', 'seed'], columns='variant', values='value').reset_index()
wide['diff'] = wide['topola_cal'] - wide['dot']
g = wide.groupby(['dataset', 'metric', 'scope'])
out = g.agg(dot_mean=('dot', 'mean'), dot_std=('dot', 'std'),
            cal_mean=('topola_cal', 'mean'), cal_std=('topola_cal', 'std'),
            diff_mean=('diff', 'mean'), diff_std=('diff', 'std'),
            n_pos=('diff', lambda x: int((x > 0).sum())), n_neg=('diff', lambda x: int((x < 0).sum())),
            n=('diff', 'count')).reset_index()
out.to_csv(join(res, 'paired.csv'), index=False)
key = out[(out.scope.isin(['all', 'slice2-slice3'])) & out.metric.isin(['ARI', 'F1', 'iLISI'])]
print(key.round(4).to_string(index=False))
