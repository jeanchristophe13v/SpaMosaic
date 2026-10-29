"""标签统一本身的逐 run 对账(与 exp005 无关,只用本实验 runs/ 中保存的聚类结果):

1. 逐切片 ARI:原始标签 vs 统一标签,应完全相同(若切片内拼写一致,映射是双射);
2. 由保存的聚类重算 ARI(all) 与 ARI(all_raw),应与 all_metrics.csv 一致(确认保存的聚类即评估所用)。
"""
import glob
import sys
from os.path import abspath, basename, dirname, join

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

HERE = dirname(abspath(__file__))
RES = abspath(join(HERE, '..', 'results'))
sys.path.insert(0, HERE)
from diag_exp009 import SEC_MOD, norm_labels  # noqa: E402

df = pd.read_csv(join(RES, 'all_metrics.csv'))
rows = []
for ds in ['Lymph-S1-S2-S3', 'Tonsil-S1-S2-S3']:
    meta = np.load(join(RES, 'runs', ds, 'meta.npz'))
    for run in sorted(glob.glob(join(RES, 'runs', ds, '*', 'seed*', 'run.npz'))):
        arm, seed = basename(dirname(dirname(run))), int(basename(dirname(run))[4:])
        e = np.load(run)
        g_all, r_all, p_all = [], [], []
        for i in range(3):
            raw = np.asarray(meta[f'lab_{SEC_MOD[i]}_b{i}'], dtype=str)
            keep = raw != 'nan'
            raw = raw[keep]
            new = norm_labels(ds, raw)
            pred = e[f'cluster_s{i}'][keep]
            a_raw, a_new = adjusted_rand_score(raw, pred), adjusted_rand_score(new, pred)
            csv = df[(df.dataset == ds) & (df.variant == arm) & (df.seed == seed) & (df.metric == 'ARI') & (df.scope == f'slice{i+1}')].value.iloc[0]
            rows.append(dict(dataset=ds, arm=arm, seed=seed, scope=f'slice{i+1}', ari_raw=a_raw, ari_unified=a_new, csv=csv,
                             n_raw=len(set(raw)), n_unified=len(set(new))))
            g_all.append(new); r_all.append(raw); p_all.append(pred)
        for scope, labs in (('all', g_all), ('all_raw', r_all)):
            v = adjusted_rand_score(np.concatenate(labs), np.concatenate(p_all))
            csv = df[(df.dataset == ds) & (df.variant == arm) & (df.seed == seed) & (df.metric == 'ARI') & (df.scope == scope)].value.iloc[0]
            rows.append(dict(dataset=ds, arm=arm, seed=seed, scope=scope, ari_raw=np.nan, ari_unified=v, csv=csv))
out = pd.DataFrame(rows)
out.to_csv(join(RES, 'check_label_fix.csv'), index=False)
sl = out[out.scope.str.startswith('slice')]
print('逐切片:原始与统一标签的类数', sl.groupby(['dataset', 'scope'])[['n_raw', 'n_unified']].max().to_string())
print(f'逐切片 ARI |raw - unified| 最大 = {np.abs(sl.ari_raw - sl.ari_unified).max():.3e}')
print(f'由保存聚类重算 ARI 与 csv 的最大差 = {np.abs(out.ari_unified - out.csv).max():.3e}')
