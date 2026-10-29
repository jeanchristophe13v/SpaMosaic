import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
R='/data/home/jiboya/SpaMosaic/experiments/exp008_topola_rank_diag/results'
norm=lambda a: np.array([x.lower().replace(' & ',' and ').replace('medulla vessel','medulla vessels').replace('vesselss','vessels') for x in a])
def loo(X, y, k=10):
    _,idx=NearestNeighbors(n_neighbors=k+1).fit(X).kneighbors(X); idx=idx[:,1:]
    cls=np.unique(y); votes=(y[idx][:,:,None]==cls[None,None,:]).sum(1); return f1_score(y, cls[votes.argmax(1)], average='macro')
def spatial_block_loo(X,y,k=10):
    return loo(X,y,k)
for ds in ['Simulations','Lymph-S1-S2-S3','Tonsil-S1-S2-S3']:
    m=np.load(f'{R}/cache/{ds}/meta.npz')
    l0,l1,l2=[norm(m[f'lab_{k}']) for k in ['rna_b0','rna_b1','adt_b2']]
    out=[f'## {ds} input-LOO: S1rna={loo(m["feat_rna_b0"],l0):.3f} S1adt={loo(m["feat_adt_b0"],l0):.3f} S1cat={loo(np.hstack([m["feat_rna_b0"],m["feat_adt_b0"]]),l0):.3f} S2rna={loo(m["feat_rna_b1"],l1):.3f} S3adt={loo(m["feat_adt_b2"],l2):.3f}']
    v=[]
    for s in [1234,2024,2025,2026,2027]:
        e=np.load(f'{R}/runs/{ds}/dot/seed{s}/emb.npz')
        e0=(e['emb_rna_b0']+e['emb_adt_b0'])/2
        v.append([loo(e0,l0),loo(e['emb_rna_b1'],l1),loo(e['emb_adt_b2'],l2)])
    v=np.mean(v,0); out.append(f'   dot-emb LOO (within section; spatial neighbours leak): S1={v[0]:.3f} S2={v[1]:.3f} S3={v[2]:.3f}')
    print('\n'.join(out))
