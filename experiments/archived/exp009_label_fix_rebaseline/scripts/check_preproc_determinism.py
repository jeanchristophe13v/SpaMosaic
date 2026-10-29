"""确定性排查:在新进程中重复 load_inputs(预处理),与 exp008 cache(与 exp005 同一实现:Sim rna MNN 4248)和其他进程对比。

用法:python check_preproc_determinism.py <dataset> <out.npz> [reps]
"""
import sys
from os.path import abspath, dirname, join
import argparse

HERE = dirname(abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import run_exp009 as drv

ds, out = sys.argv[1], sys.argv[2]
reps = int(sys.argv[3]) if len(sys.argv) > 3 else 1
args = argparse.Namespace(sim_setting='rna-0.1-12-adt-2.1', sim_rep='1')
ref = np.load(join(drv.REPO_ROOT, 'experiments/exp008_topola_rank_diag/results/cache', ds, 'meta.npz'))
saved = {}
for r in range(reps):
    input_dict, raw, layout = drv.load_inputs(ds, args)
    key = drv.PARAMS[ds]['input_key']
    for m, ads in input_dict.items():
        for bi, ad in enumerate(ads):
            if ad is None:
                continue
            x = np.asarray(ad.obsm[key], dtype=np.float32)
            k = f'feat_{m}_b{bi}'
            d_ref = float(np.abs(x - ref[k]).max())
            d_first = float(np.abs(x - saved[k]).max()) if k in saved else 0.0
            print(f'[rep {r}] {k}: maxabs vs exp008 cache = {d_ref:.3e}; vs rep0 = {d_first:.3e}', flush=True)
            if r == 0:
                saved[k] = x
np.savez(out, **saved)
