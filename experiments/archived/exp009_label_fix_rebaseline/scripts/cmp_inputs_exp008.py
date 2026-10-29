import numpy as np, sys
a = np.load('experiments/exp008_topola_rank_diag/results/cache/Simulations/meta.npz')
b = np.load('experiments/exp009_label_fix_rebaseline/results/runs/Simulations/meta.npz')
for k in sorted(a.files):
    if k.startswith('feat_'):
        x, y = a[k], b[k]
        print(k, x.shape, y.shape, 'maxabs', float(np.abs(x - y).max()), 'rel', float(np.abs(x - y).max() / np.abs(x).max()))
    elif k.startswith('names_') or k.startswith('lab_'):
        print(k, 'equal', np.array_equal(a[k], b[k]))
