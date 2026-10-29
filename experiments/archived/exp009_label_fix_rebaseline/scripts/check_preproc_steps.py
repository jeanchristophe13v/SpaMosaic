"""确定性排查 3:定位预处理中的非确定步骤(Simulations RNA:scale -> PCA -> Harmony)。"""
import sys
from os.path import abspath, dirname, join
sys.path.insert(0, dirname(abspath(__file__)))
import numpy as np
import scanpy as sc
import run_exp009 as drv
from spamosaic.preprocessing import harmony

d = join(drv.REPO_ROOT, 'data/processed/Simulations/rna-0.1-12-adt-2.1/1')
ads = []
for n, s in (('ad_rna1', 'mult'), ('ad_rna2', 'rna')):
    a = sc.read_h5ad(join(d, n + '.h5ad')); a.obs['src'] = s; a.X = a.layers['counts'].copy(); ads.append(a)
ad = sc.concat(ads)
sc.pp.scale(ad)
pcs = []
for _ in range(3):
    x = ad.copy(); sc.pp.pca(x, n_comps=50); pcs.append(x.obsm['X_pca'].copy())
print('[pca] maxabs rep1-rep0', np.abs(pcs[1] - pcs[0]).max(), 'rep2-rep0', np.abs(pcs[2] - pcs[0]).max(), flush=True)
b = ad.obs['src'].to_list()
for gpu in (True, False):
    hs = [harmony(pcs[0], b, use_gpu=gpu) for _ in range(3)]
    print(f'[harmony use_gpu={gpu}] maxabs rep1-rep0 {np.abs(hs[1]-hs[0]).max():.3e} rep2-rep0 {np.abs(hs[2]-hs[0]).max():.3e}', flush=True)
