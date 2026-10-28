#!/usr/bin/env python
"""Smoke test: 全模块 import + 核心 loss / topola 小前向。改 spamosaic/ 后必跑。

用法: python tests/test_smoke.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

import spamosaic  # 触发全部子模块 import(不含重量级的 spamosaic.metrics)
from spamosaic.loss import CL_loss, BPR_loss
from spamosaic.topola import compute_topola, full_topola_distance


def main():
    # --- loss 前向 + 反向 ---
    n, rep = 6, 2
    simi = torch.randn(n * rep, n * rep, requires_grad=True)
    for loss_cls in (CL_loss, BPR_loss):
        crit = loss_cls(n, rep=rep)
        out = crit(simi)
        assert out.ndim == 0, f'{loss_cls.__name__} 输出应为标量'
        out.backward()

    # --- topola 前向 + 反向 ---
    C = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    Z = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    D, S = compute_topola(C, Z, lambda_topola=10.0)
    assert D.shape == (8, 8) and S.shape == (4, 4)
    S.sum().backward()
    assert C.grad is not None and Z.grad is not None
    d = full_topola_distance(torch.cat([C, Z], dim=0), 10.0)
    assert torch.is_tensor(d)

    print('smoke test passed')


if __name__ == '__main__':
    main()
