"""TopoLa 完整关系矩阵与 C-Z 打分块：独立代码文件。


    1. E = [C; Z]                    沿样本维上下拼接
    2. H = E @ E.T
       D_lambda = H @ (H + lambda * I)^(-1)
    3. S_lambda = D_lambda[0:N, N:2*N]

输入：C、Z 均为 [N,d]，由外部提供原流程已逐行L2归一化的特征。
本文件不额外归一化特征或关系核。
输出：完整 D_lambda 为 [2N,2N]；S_lambda 为 [N,N]。

与实际实验相同，D_lambda 通过 CPU float64 QR 等价求解：
    A = [E; sqrt(lambda)*I_d] = Q R
    R.T @ R = E.T @ E + lambda*I_d
    F = E @ inverse(R)
    D_lambda = F @ F.T
因此 D_lambda = E(E.T E + lambda I_d)^(-1)E.T
             = H(H + lambda I_2N)^(-1)。
代码使用三角方程求解计算 F，不显式求逆。

完整 D_lambda 先计算完毕，再抽取 S_lambda；设备转换保留自动求导。
不使用 detach、额外jitter、对角归一化或额外lambda缩放。

依赖：PyTorch；使用摩尔线程MUSA设备时还需要 torch_musa。
函数逐字提取自当前实验实现。

调用示例：
    from spamosaic.topola import compute_topola
    D_lambda, S_lambda = compute_topola(C, Z, lambda_topola=10.0)
"""
from contextlib import nullcontext
import math
import torch


def autocast(device, enabled=False):
    if device == "musa":
        import torch_musa
        return torch_musa.amp.autocast(enabled=enabled)
    if device == "cuda":
        return torch.cuda.amp.autocast(enabled=enabled)
    return nullcontext()


def full_topola_distance(features, regularization):
    """Return every entry of D, in the original row order of features E."""
    if not math.isfinite(regularization) or regularization <= 0:
        raise ValueError('Positive finite paper lambda required')
    if features.ndim != 2 or 0 in features.shape:
        raise ValueError('Nonempty feature matrix E required')
    device = features.device
    dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
    with autocast(device.type, enabled=False):
        e = features.cpu().to(torch.float64)
        identity = torch.eye(e.shape[1], dtype=e.dtype)
        augmented = torch.cat((e, math.sqrt(regularization) * identity), dim=0)
        _, r = torch.linalg.qr(augmented, mode='reduced')
        f = torch.linalg.solve_triangular(r.T, e.T, upper=False).T
        # Full 2N x 2N distance exists before any C-Z block is selected.
        distance = f @ f.T
        if not bool(torch.isfinite(distance).all()):
            raise FloatingPointError('Nonfinite full TopoLa distance')
        return distance.to(dtype).to(device)


def compute_topola(C, Z, lambda_topola=10.0):
    """上下拼接 C、Z，先求完整 D_lambda，再返回其 C-Z 块 S_lambda。"""
    if C.ndim != 2 or Z.ndim != 2 or C.shape != Z.shape:
        raise ValueError("C、Z 必须具有相同的二维形状 [N,d]")
    if C.device != Z.device:
        raise ValueError("C、Z 必须位于同一设备")

    # 第一步：沿样本维上下拼接，样本数翻倍。
    E = torch.cat((C, Z), dim=0)           # [2N,d]
    N = C.shape[0]

    # 第二步：整体计算完整矩阵；内部为上文公式的 QR 等价求解。
    D_lambda = full_topola_distance(E, lambda_topola)  # [2N,2N]

    # 第三步：完整 D_lambda 计算完成后，再抽取右上角 C-Z 块。
    S_lambda = D_lambda[0:N, N:2*N]        # [N,N]
    return D_lambda, S_lambda


def reference_topola_features(query, reference, regularization):
    """Whiten query using a shared reference; preserve both gradient paths."""
    r = reference.to(torch.float64)
    q = query.to(torch.float64)
    chol = torch.linalg.cholesky(
        r.T @ r + regularization * torch.eye(r.shape[1], device=r.device, dtype=r.dtype)
    )
    return torch.linalg.solve_triangular(chol, q.T, upper=False).T.to(query.dtype)
