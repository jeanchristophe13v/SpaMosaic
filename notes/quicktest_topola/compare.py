#!/usr/bin/env python
"""对比 quicktest 结果与 notebook 基线(examples/out/*/metrics.csv)。"""
import argparse
from os.path import abspath, dirname, join

import pandas as pd

REPO_ROOT = abspath(join(dirname(__file__), '..', '..'))
NOTEBOOK = {
    'Simulations': 'examples/out/Simulations/metrics.csv',
    'Lymph-S1-S2-S3': 'examples/out/Lymph-S1-S2-S3/metrics.csv',
    'Tonsil-S1-S2-S3': 'examples/out/Tonsil-S1-S2-S3/metrics.csv',
}
SCOPES = {'all': 'all', 'slice1': 'slice1', 'slice2': 'slice2', 'slice3': 'slice3',
          'slice2-slice3': 'slice2-slice3'}
CORE = [('ARI', 'all'), ('F1', 'slice2-slice3'), ('iLISI', 'all')]


def notebook_baseline(dataset):
    path = join(REPO_ROOT, NOTEBOOK[dataset])
    df = pd.read_csv(path, index_col=0)
    out = {}
    for metric in ('ARI', 'F1', 'PAS', 'CHAOS', 'iLISI'):
        if metric not in df.columns:
            continue
        for scope, value in df[metric].dropna().items():
            out[(metric, scope)] = float(value)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='r1')
    parser.add_argument('--out-dir', default=join(dirname(abspath(__file__)), 'results'))
    args = parser.parse_args()

    long = pd.read_csv(join(args.out_dir, args.tag, 'metrics_long.csv'))
    wide = long.pivot_table(index=['dataset', 'metric', 'scope'], columns='variant', values='value')

    for dataset in long.dataset.unique():
        base = notebook_baseline(dataset)
        arms = [c for c in ('dot', 'topola_cal', 'topola_raw') if c in wide.columns]
        rows = []
        keys = sorted({k for k in wide.loc[dataset].index} | set(base),
                      key=lambda k: (['ARI', 'F1', 'iLISI', 'PAS', 'CHAOS'].index(k[0]), k[1]))
        for key in keys:
            if key[0] not in ('ARI', 'F1', 'iLISI', 'PAS', 'CHAOS'):
                continue
            row = {'metric': key[0], 'scope': key[1], 'notebook(dot)': base.get(key)}
            sub = wide.loc[dataset].loc[key] if key in wide.loc[dataset].index else None
            for arm in arms:
                column = 'topola+cal' if arm == 'topola_cal' else arm
                row[column] = None if sub is None else float(sub.get(arm, float('nan')))
            if row.get('topola+cal') is not None and row.get('dot') is not None:
                row['d_cal_vs_dot'] = row['topola+cal'] - row['dot']
            rows.append(row)
        print(f'\n=== {dataset} ===')
        print(pd.DataFrame(rows).round(4).to_string(index=False))

    print('\n=== 核心指标 ===')
    summary = []
    for dataset in long.dataset.unique():
        base = notebook_baseline(dataset)
        for metric, scope in CORE:
            row = {'dataset': dataset, 'metric': metric,
                   'notebook(dot)': base.get((metric, scope))}
            for arm, column in (('dot', 'dot'), ('topola_cal', 'topola+cal'), ('topola_raw', 'topola_raw')):
                try:
                    row[column] = float(wide.loc[(dataset, metric, scope), arm])
                except KeyError:
                    row[column] = float('nan')
            row['delta_cal'] = row['topola+cal'] - row['dot']
            summary.append(row)
    print(pd.DataFrame(summary).round(4).to_string(index=False))


if __name__ == '__main__':
    main()
