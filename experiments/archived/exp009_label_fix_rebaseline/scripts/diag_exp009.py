"""exp009 诊断 D1-D3:只读 results/runs/,不训练;全部使用统一后的标签(nan 剔除)。

D1:合并嵌入(L2 归一)上 top-10 近邻标签一致率(切片内去掉自身;跨切片为查询→参照)。
D2:切片内留一 kNN(k=10)macro-F1,嵌入(合并嵌入,原尺度)对比输入特征。
D3:SpaMosaic 实际 MNN 对(model.mod_mnn_set)的标签一致率与随机配对基线 sum_c p_S1(c) p_other(c)。
"""
import glob
import os
import sys
from os.path import abspath, basename, dirname, join

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors

HERE = dirname(abspath(__file__))
RES = abspath(join(HERE, '..', 'results'))
sys.path.insert(0, HERE)
from run_exp009 import LABEL_MAPS  # noqa: E402  标签映射表与 driver 共用同一份

DATASETS = ['Simulations', 'Lymph-S1-S2-S3', 'Tonsil-S1-S2-S3']
K = 10
SEC = {0: 'S1', 1: 'S2', 2: 'S3'}
# 每个切片的「代表模态」:S1 取 rna 的标签(与 adt 相同 obs),S2 只有 rna,S3 只有 adt
SEC_MOD = {0: 'rna', 1: 'rna', 2: 'adt'}


def norm_labels(dataset, lab):
    lab = np.asarray(lab, dtype=str)
    lm = LABEL_MAPS.get(dataset)
    if lm is None:
        return lab
    miss = sorted(set(lab) - set(lm))
    if miss:
        raise ValueError(f'{dataset}: 标签映射未覆盖 {miss}')
    out = np.array([lm[x] for x in lab])
    return np.where(out == 'Nan', 'nan', out)


def l2(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def knn_idx(Q, R, k, self_=False):
    _, idx = NearestNeighbors(n_neighbors=k + int(self_)).fit(R).kneighbors(Q)
    return idx[:, 1:] if self_ else idx


def agree(Q, lq, R, lr, self_=False):
    idx = knn_idx(Q, R, K, self_)
    return float((lr[idx] == lq[:, None]).mean())


def chance(la, lb):
    a = pd.Series(la).value_counts(normalize=True)
    b = pd.Series(lb).value_counts(normalize=True)
    return float((a * b.reindex(a.index).fillna(0)).sum())


def loo_f1(X, y):
    idx = knn_idx(X, X, K, self_=True)
    cls = np.unique(y)
    votes = (y[idx][:, :, None] == cls[None, None, :]).sum(1)
    return float(f1_score(y, cls[votes.argmax(1)], average='macro'))


def main():
    agr_rows, loo_rows, mnn_rows, lab_rows = [], [], [], []
    for ds in DATASETS:
        meta = np.load(join(RES, 'runs', ds, 'meta.npz'))
        lab = {k[4:]: norm_labels(ds, meta[k]) for k in meta.files if k.startswith('lab_')}
        lab_raw = {k[4:]: np.asarray(meta[k], dtype=str) for k in meta.files if k.startswith('lab_')}
        names = {k[6:]: np.asarray(meta[k], dtype=str) for k in meta.files if k.startswith('names_')}
        if not np.array_equal(names['rna_b0'], names['adt_b0']) or not np.array_equal(lab['rna_b0'], lab['adt_b0']):
            raise RuntimeError(f'{ds}: S1 的 rna/adt obs 或标签不一致')
        # 切片 -> (标签, 保留掩码)
        sec_lab = {i: lab[f'{SEC_MOD[i]}_b{i}'] for i in SEC}
        sec_keep = {i: sec_lab[i] != 'nan' for i in SEC}
        for i in SEC:
            raw_set = sorted(set(lab_raw[f'{SEC_MOD[i]}_b{i}']) - {'nan'})
            new_set = sorted(set(sec_lab[i][sec_keep[i]]))
            lab_rows.append(dict(dataset=ds, section=SEC[i], n_cells=len(sec_lab[i]), n_nan=int((~sec_keep[i]).sum()),
                                 n_raw_labels=len(raw_set), n_labels=len(new_set),
                                 raw_labels=';'.join(raw_set), labels=';'.join(new_set)))
        all_raw = set().union(*[set(v) for v in lab_raw.values()]) - {'nan'}
        all_new = set().union(*[set(v) for v in lab.values()]) - {'nan'}
        print(f'## {ds}: 全部切片标签类数 原始 {len(all_raw)} -> 统一后 {len(all_new)}', flush=True)

        # D2 输入特征(与 seed 无关)
        feats = {f'S1_rna': meta['feat_rna_b0'], 'S1_adt': meta['feat_adt_b0'],
                 'S1_cat': np.hstack([meta['feat_rna_b0'], meta['feat_adt_b0']]),
                 'S2_rna': meta['feat_rna_b1'], 'S3_adt': meta['feat_adt_b2']}
        inp = {}
        for fk, X in feats.items():
            i = int(fk[1]) - 1
            inp[fk] = loo_f1(X[sec_keep[i]], sec_lab[i][sec_keep[i]])
            loo_rows.append(dict(dataset=ds, arm='input', seed=-1, rep=fk, section=SEC[i], loo_f1=inp[fk]))

        # D3 用的 obs 名 -> 标签
        name2lab, name2raw = {}, {}
        for key in names:
            for n, l, r in zip(names[key], lab[key], lab_raw[key]):
                if name2lab.get(n, l) != l:
                    raise RuntimeError(f'{ds}: obs 名 {n} 在不同切片上标签冲突')
                name2lab[n], name2raw[n] = l, r
        sec_of = {}
        for key in names:
            for n in names[key]:
                sec_of[n] = int(key[-1])

        for run in sorted(glob.glob(join(RES, 'runs', ds, '*', 'seed*', 'run.npz'))):
            arm = basename(dirname(dirname(run)))
            seed = int(basename(dirname(run))[4:])
            e = np.load(run)
            for i in SEC:
                if not np.array_equal(e[f'names_s{i}'], names[f'{SEC_MOD[i]}_b{i}']):
                    raise RuntimeError(f'{run}: 切片 {i} obs 顺序与 meta 不一致')
            M = {i: e[f'merged_s{i}'][sec_keep[i]] for i in SEC}
            Mn = {i: l2(M[i]) for i in SEC}
            L = {i: sec_lab[i][sec_keep[i]] for i in SEC}
            base = dict(dataset=ds, arm=arm, seed=seed)

            # D1
            for i in SEC:
                agr_rows.append(dict(**base, pair=f'within_{SEC[i]}', value=agree(Mn[i], L[i], Mn[i], L[i], self_=True),
                                     chance=chance(L[i], L[i])))
            k0 = sec_keep[0]
            R0, A0 = l2(e['emb_rna_b0'][k0]), l2(e['emb_adt_b0'][k0])
            agr_rows.append(dict(**base, pair='S1rna->S1adt', value=agree(R0, L[0], A0, L[0]), chance=chance(L[0], L[0])))
            for q, r in ((1, 0), (2, 0), (1, 2), (2, 1)):
                agr_rows.append(dict(**base, pair=f'{SEC[q]}->{SEC[r]}', value=agree(Mn[q], L[q], Mn[r], L[r]),
                                     chance=chance(L[q], L[r])))

            # D2
            for i in SEC:
                loo_rows.append(dict(**base, rep=f'{SEC[i]}_emb', section=SEC[i], loo_f1=loo_f1(M[i], L[i])))

            # D3
            for m, other in (('rna', 1), ('adt', 2)):
                pairs = e[f'mnn_{m}']
                if len(pairs) == 0:
                    continue
                secs = np.array([[sec_of[a], sec_of[b]] for a, b in pairs])
                if not np.all(np.sort(secs, 1) == np.array([0, other])):
                    raise RuntimeError(f'{run}: mnn_{m} 含非 S1↔{SEC[other]} 的对')
                for kind, n2l, labs in (('unified', name2lab, lab), ('raw', name2raw, lab_raw)):
                    la = np.array([n2l[a] for a in pairs[:, 0]])
                    lb = np.array([n2l[b] for b in pairs[:, 1]])
                    keep = (la != 'nan') & (lb != 'nan')
                    l1, l2_ = labs[f'{m}_b0'], labs[f'{m}_b{other}']
                    mnn_rows.append(dict(**base, modality=m, pair=f'S1<->{SEC[other]}', labels=kind,
                                         n_pairs=len(pairs), n_used=int(keep.sum()),
                                         label_agree=float((la[keep] == lb[keep]).mean()),
                                         chance=chance(l1[l1 != 'nan'], l2_[l2_ != 'nan'])))
            print(f'  {ds} {arm} seed{seed} done', flush=True)

    agr, loo, mnn, labs = map(pd.DataFrame, (agr_rows, loo_rows, mnn_rows, lab_rows))
    agr.to_csv(join(RES, 'diag_agreement.csv'), index=False)
    loo.to_csv(join(RES, 'diag_loo.csv'), index=False)
    mnn.to_csv(join(RES, 'diag_mnn.csv'), index=False)
    labs.to_csv(join(RES, 'diag_labels.csv'), index=False)
    pd.set_option('display.width', 250)

    print('\n=== 标签集合 ===')
    print(labs[['dataset', 'section', 'n_cells', 'n_nan', 'n_raw_labels', 'n_labels']].to_string(index=False))

    print('\n=== D1 top-10 标签一致率(5 seed 均值±std)===')
    t = agr.groupby(['dataset', 'arm', 'pair']).agg(mean=('value', 'mean'), std=('value', 'std'), chance=('chance', 'mean'))
    print(t.round(3).to_string())

    print('\n=== D2 留一 kNN macro-F1 ===')
    t = loo.groupby(['dataset', 'arm', 'rep'])['loo_f1'].agg(['mean', 'std'])
    print(t.round(3).to_string())

    print('\n=== D3 SpaMosaic MNN 对标签一致率 ===')
    t = mnn.groupby(['dataset', 'modality', 'pair', 'labels', 'arm']).agg(
        agree_mean=('label_agree', 'mean'), agree_std=('label_agree', 'std'), chance=('chance', 'mean'),
        n_pairs=('n_pairs', 'mean'), n_used=('n_used', 'mean'))
    print(t.round(4).to_string())

    # H_S3 判定(dot arm)
    print('\n=== H_S3(dot arm)===')
    rows = []
    for ds in ['Lymph-S1-S2-S3', 'Tonsil-S1-S2-S3']:
        for arm in ('dot', 'topola_cal'):
            w = agr[(agr.dataset == ds) & (agr.arm == arm)].pivot(index='seed', columns='pair', values='value')
            d = w['S3->S1'] - w['S2->S1']
            rows.append(dict(dataset=ds, arm=arm, test='D1 S3->S1 - S2->S1', diff_mean=d.mean(), diff_std=d.std(),
                             n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()),
                             per_seed=';'.join(f'{x:+.4f}' for x in d.values)))
            le = loo[(loo.dataset == ds) & (loo.arm == arm) & (loo.rep == 'S3_emb')].set_index('seed')['loo_f1']
            d = le - inp_lookup(loo, ds, 'S3_adt')
            rows.append(dict(dataset=ds, arm=arm, test='D2 S3 emb - S3 adt input', diff_mean=d.mean(), diff_std=d.std(),
                             n_pos=int((d > 0).sum()), n_neg=int((d < 0).sum()),
                             per_seed=';'.join(f'{x:+.4f}' for x in d.values)))
    hs3 = pd.DataFrame(rows)
    hs3.to_csv(join(RES, 'diag_hs3.csv'), index=False)
    print(hs3.round(4).to_string(index=False))


def inp_lookup(loo, ds, rep):
    return float(loo[(loo.dataset == ds) & (loo.arm == 'input') & (loo.rep == rep)]['loo_f1'].iloc[0])


if __name__ == '__main__':
    main()
