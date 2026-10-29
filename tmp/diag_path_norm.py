import numpy as np, sys
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score
from collections import Counter
R='/data/home/jiboya/SpaMosaic/experiments/exp008_topola_rank_diag/results'
def knnW(q, ref, k=10):
    nn=NearestNeighbors(n_neighbors=k).fit(ref); _,idx=nn.kneighbors(q)
    W=np.zeros((len(q),len(ref)),np.float32); np.put_along_axis(W,idx,1.0/k,1); return W
def onehot(lab, classes):
    return (lab[:,None]==classes[None,:]).astype(np.float32)
def f1(true, P, classes): return f1_score(true, classes[P.argmax(1)], average='macro')
for ds in ['Simulations','Lymph-S1-S2-S3','Tonsil-S1-S2-S3']:
    m=np.load(f'{R}/cache/{ds}/meta.npz')
    r0,r1,a0,a2=[m[f'feat_{k}'] for k in ['rna_b0','rna_b1','adt_b0','adt_b2']]
    norm=lambda a: np.array([x.lower().replace(' & ',' and ').replace('medulla vessel','medulla vessels').replace('vesselss','vessels') for x in a])
    l0=norm(m['lab_rna_b0']); l1=norm(m['lab_rna_b1']); l2=norm(m['lab_adt_b2'])
    classes=np.array(sorted(set(l0)|set(l1)|set(l2)))
    print(f'\n## {ds}  n=({len(l0)},{len(l1)},{len(l2)})  classes={len(classes)}')
    for nm,l in [('S1',l0),('S2',l1),('S3',l2)]:
        c=Counter(l); print(f'  {nm} labels:', {k:round(v/len(l),3) for k,v in sorted(c.items())})
    Y0,Y1,Y2=[onehot(l,classes) for l in (l0,l1,l2)]
    # same-modality ceilings from S1
    print('  S1->S2 (rna input) F1=%.3f'%f1(l1, knnW(r1,r0)@Y0, classes))
    print('  S1->S3 (adt input) F1=%.3f'%f1(l2, knnW(a2,a0)@Y0, classes))
    # training-free 2-hop path S2<->S3 via S1 bridge
    W21=knnW(r1,r0); W13=knnW(a0,a2); W31=knnW(a2,a0); W12=knnW(r0,r1)
    p_s2=f1(l1, W21@(W13@Y2), classes); p_s3=f1(l2, W31@(W12@Y1), classes)
    print('  path S3->S1->S2 F1=%.3f  S2->S1->S3 F1=%.3f  mean=%.3f'%(p_s2,p_s3,(p_s2+p_s3)/2))
    # vanilla embeddings
    for arm in ['dot','topola_cal']:
        vals=[]
        for s in [1234,2024,2025,2026,2027]:
            e=np.load(f'{R}/runs/{ds}/{arm}/seed{s}/emb.npz')
            e1=e['emb_rna_b1']; e2=e['emb_adt_b2']; e0=(e['emb_rna_b0']+e['emb_adt_b0'])/2
            d=(f1(l2, knnW(e2,e1)@Y1, classes)+f1(l1, knnW(e1,e2)@Y2, classes))/2
            s1s3=f1(l2, knnW(e2,e0)@Y0, classes); s1s2=f1(l1, knnW(e1,e0)@Y0, classes)
            vals.append((d,s1s2,s1s3))
        v=np.array(vals).mean(0)
        print(f'  {arm} emb: S2<->S3 F1={v[0]:.3f}  S1->S2={v[1]:.3f}  S1->S3={v[2]:.3f}')
