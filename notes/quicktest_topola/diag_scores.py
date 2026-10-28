#!/usr/bin/env python
"""诊断:校准前后 TopoLa 分数的尺度、逐行对角异质性、以及 CL_loss 下溢余量。

只看初始嵌入(seed 固定),不训练。用于解释 quicktest 的现象。
用法: conda run --no-capture-output -n spamosaic-env python notes/quicktest_topola/diag_scores.py --dataset Lymph-S1-S2-S3
"""
import argparse
import sys
from os.path import abspath, dirname, join

import numpy as np
import torch

sys.path.insert(0, abspath(join(dirname(__file__), '..', '..')))
sys.path.insert(0, dirname(abspath(__file__)))

from run_quicktest import PARAMS, load_inputs  # noqa: E402

from spamosaic.framework import SpaMosaic  # noqa: E402
from spamosaic.topola import full_topola_distance  # noqa: E402
from spamosaic.train_utils import set_seeds  # noqa: E402

UNDERFLOW = 87.3  # float32 exp 下溢阈值, log(sum exp) < -87.3 时 sum 归零


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', default='Lymph-S1-S2-S3', choices=list(PARAMS))
    p.add_argument('--device', default='cuda:5')
    p.add_argument('--lambda-topola', type=float, default=10.0)
    args = p.parse_args()

    set_seeds(1234)
    input_dict, _, _ = load_inputs(args.dataset, argparse.Namespace(
        sim_setting='rna-0.1-12-adt-2.1', sim_rep='1'))
    cfg = PARAMS[args.dataset]

    model = SpaMosaic(modBatch_dict=input_dict, input_key=cfg['input_key'], batch_key='src',
                      intra_knns=10, inter_knn_base=10, w_g=0.8, seed=1234, device=args.device)
    nets = model.prepare_net('wlgcn')
    mods = list(model.mod_list)
    z = {}
    with torch.no_grad():
        for k in mods:
            out, _ = nets[k](model.mod_graphs['attribute'][k], model.mod_graphs['edge'][k],
                             model.mod_graphs['edgeW'][k])
            z[k] = torch.split(out, model.mod_batch_split[k])

    bi = next(iter(model.bridge_batch_num_ids2mod))
    feat = torch.cat([z[k][bi] for k in mods])
    D = full_topola_distance(feat, args.lambda_topola)
    scale = D.diagonal().mean()
    Dn = D / scale
    n = D.shape[0]
    off_mask = ~torch.eye(n, dtype=torch.bool, device=D.device)
    cos = feat @ feat.T

    diag = Dn.diagonal()
    neg_inf = torch.full_like(Dn, -torch.inf)
    off_max = torch.where(off_mask, Dn, neg_inf).max(dim=1).values
    margin = diag - off_max

    print(f'\n=== {args.dataset} initial embeddings, kernel n={n}, lambda={args.lambda_topola} ===')
    print(f'diag(D): mean={float(D.diagonal().mean()):.4e}')
    print(f'diag(D)/mean:  min={float(diag.min()):.3f} p05={float(diag.quantile(0.05)):.3f} '
          f'median={float(diag.median()):.3f} p95={float(diag.quantile(0.95)):.3f} max={float(diag.max()):.3f}')
    print(f'calibrated off-diag: std={float(Dn[off_mask].std()):.4f} max={float(Dn[off_mask].max()):.4f}')
    print(f'cosine    off-diag: std={float(cos[off_mask].std()):.4f} max={float(cos[off_mask].max()):.4f}')
    print(f'row margin (self - best negative), calibrated scores: '
          f'min={float(margin.min()):.3f} median={float(margin.median()):.3f} max={float(margin.max()):.3f}')
    for T in (0.01, 0.1):
        over = int((margin / T > UNDERFLOW).sum())
        print(f'  T={T}: rows with logits margin > {UNDERFLOW} (原 CL_loss 下溢) = {over}/{n}')
    cos_margin = cos.diagonal() - torch.where(off_mask, cos, torch.full_like(cos, -torch.inf)).max(dim=1).values
    for T in (0.01, 0.1):
        over = int((cos_margin / T > UNDERFLOW).sum())
        print(f'  cosine T={T}: rows over threshold = {over}/{n}')

    # exp004 式温度校准:D 的 pos-neg gap 标准差 / 余弦的 gap 标准差
    n_b = len(z[mods[0]][bi])
    pos = torch.cat([torch.arange(n_b, device=D.device) + n_b,
                     torch.arange(n_b, device=D.device)])
    pos_col = torch.zeros_like(off_mask)
    pos_col[torch.arange(2 * n_b, device=D.device), pos] = True
    neg = off_mask & ~pos_col
    gap_top = (Dn.gather(1, pos[:, None]) - Dn)[neg]
    gap_cos = (cos.gather(1, pos[:, None]) - cos)[neg]
    ratio = float(gap_top.std() / gap_cos.std())
    print(f'gap std: topola_cal={float(gap_top.std()):.4f} cosine={float(gap_cos.std()):.4f} '
          f'ratio={ratio:.3f}')
    print(f'  -> 温度校准(exp004 式):T_cal = T_dot * {ratio:.3f};'
          f'等价于把校准分数再乘 {1/ratio:.3f}')


if __name__ == '__main__':
    main()
