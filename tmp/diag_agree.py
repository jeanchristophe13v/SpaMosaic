import numpy as np
from sklearn.neighbors import NearestNeighbors
R='/data/home/jiboya/SpaMosaic/experiments/exp008_topola_rank_diag/results'
norm=lambda a: np.array([x.lower().replace(' & ',' and ').replace('medulla vessel','medulla vessels').replace('vesselss','vessels') for x in a])
def l2(x): return x/np.linalg.norm(x,axis=1,keepdims=True)
def agree(Q,lq,Rf,lr,k=10,self_=False):
    _,idx=NearestNeighbors(n_neighbors=k+int(self_)).fit(Rf).kneighbors(Q)
    if self_: idx=idx[:,1:]
    return (lr[idx]==lq[:,None]).mean()
for ds in ['Simulations','Lymph-S1-S2-S3','Tonsil-S1-S2-S3']:
    m=np.load(f'{R}/cache/{ds}/meta.npz')
    l0,l1,l2_=[norm(m[f'lab_{k}']) for k in ['rna_b0','rna_b1','adt_b2']]
    rows=[]
    for s in [1234,2024,2025,2026,2027]:
        e=np.load(f'{R}/runs/{ds}/dot/seed{s}/emb.npz')
        R0,A0,R1,A2=[l2(e[k]) for k in ['emb_rna_b0','emb_adt_b0','emb_rna_b1','emb_adt_b2']]
        M0=l2(R0+A0)
        rows.append([agree(M0,l0,M0,l0,self_=True),agree(R1,l1,R1,l1,self_=True),agree(A2,l2_,A2,l2_,self_=True),
                     agree(R0,l0,A0,l0), agree(R1,l1,M0,l0), agree(A2,l2_,M0,l0), agree(R1,l1,A2,l2_), agree(A2,l2_,R1,l1)])
    r=np.mean(rows,0)
    print(f'## {ds}: within S1={r[0]:.3f} S2={r[1]:.3f} S3={r[2]:.3f} | S1rna->S1adt={r[3]:.3f} | S2->S1={r[4]:.3f} S3->S1={r[5]:.3f} | S2->S3={r[6]:.3f} S3->S2={r[7]:.3f}')
