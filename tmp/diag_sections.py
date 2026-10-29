import numpy as np
R='/data/home/jiboya/SpaMosaic/experiments/exp008_topola_rank_diag/results'
norm=lambda a: np.array([x.lower().replace(' & ',' and ').replace('medulla vessel','medulla vessels').replace('vesselss','vessels') for x in a])
def l2(x): return x/np.linalg.norm(x,axis=1,keepdims=True)
for ds in ['Simulations','Lymph-S1-S2-S3','Tonsil-S1-S2-S3']:
    m=np.load(f'{R}/cache/{ds}/meta.npz')
    n0,n1,n2=m['names_rna_b0'],m['names_rna_b1'],m['names_adt_b2']
    l0,l1,l2_=[norm(m[f'lab_{k}']) for k in ['rna_b0','rna_b1','adt_b2']]
    res=[]
    for s in [1234,2024,2025,2026,2027]:
        e=np.load(f'{R}/runs/{ds}/dot/seed{s}/emb.npz')
        R0,A0,R1,A2=[l2(e[k]) for k in ['emb_rna_b0','emb_adt_b0','emb_rna_b1','emb_adt_b2']]
        # pool of ADT obs = S1 adt + S3 adt ; RNA obs = S1 rna + S2 rna
        Apool=np.vstack([A0,A2]); Asec=np.r_[np.zeros(len(A0)),np.ones(len(A2))]; Alab=np.r_[l0,l2_]
        Rpool=np.vstack([R0,R1]); Rsec=np.r_[np.zeros(len(R0)),np.ones(len(R1))]; Rlab=np.r_[l0,l1]
        # S2 rna -> ADT pool top10: fraction in S3, label agreement
        S=R1@Apool.T; top=np.argsort(-S,1)[:,:10]
        fr3=(Asec[top]==1).mean(); agr=(Alab[top]==l1[:,None]).mean()
        agr_s1=np.mean([(Alab[t][Asec[t]==0]==l1[i]).mean() if (Asec[t]==0).any() else np.nan for i,t in enumerate(top)])
        S=A2@Rpool.T; top=np.argsort(-S,1)[:,:10]
        fr2=(Rsec[top]==1).mean(); agr2=(Rlab[top]==l2_[:,None]).mean()
        # composite positives: S2 rna a with rna-MNN partner b in S1 -> rank of b_adt in ADT pool
        mn=e['mnn_rna']; i0={n:i for i,n in enumerate(n0)}; i1={n:i for i,n in enumerate(n1)}
        pairs=[(i1[b],i0[a]) for a,b in mn if a in i0 and b in i1]+[(i1[a],i0[b]) for a,b in mn if a in i1 and b in i0]
        pa=np.array(pairs)
        if len(pa):
            sc=R1[pa[:,0]]@Apool.T; pos=(R1[pa[:,0]]*A0[pa[:,1]]).sum(1)
            pct=(sc>pos[:,None]).mean(1)
            comp_rank=np.median(pct); comp_lab=(l1[pa[:,0]]==l0[pa[:,1]]).mean()
        else: comp_rank=comp_lab=np.nan
        res.append([fr3,agr,agr_s1,fr2,agr2,comp_rank,comp_lab,len(pa)])
    r=np.nanmean(res,0)
    print(f'## {ds}\n  S2rna->ADT top10: frac in S3={r[0]:.3f} (S3 share of pool={len(A2)/len(Apool):.3f}) label agree={r[1]:.3f} (agree among S1 hits={r[2]:.3f})')
    print(f'  S3adt->RNA top10: frac in S2={r[3]:.3f} (S2 share={len(R1)/len(Rpool):.3f}) label agree={r[4]:.3f}')
    print(f'  composite pos (S2 rna a, S1 adt of MNN partner): median pct of ADT pool scoring above it={r[5]:.4f}; label agree={r[6]:.3f}; n={int(r[7])}')
