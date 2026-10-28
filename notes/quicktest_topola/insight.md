# quicktest_topola:TopoLa 分数 + 尺度校准的快速验证记录

> **状态:快速验证,不是正式实验。** 没有开 exp005,没有写 `experiments/index.csv`。
> 单 seed,未扫 λ,未调温度。结论是「暂不达标」,下面所有内容用于让别人明天复现并定位问题。
>
> 运行日期 2026-10-28。代码在 `notes/quicktest_topola/`,`spamosaic/` 包**未做任何修改**。

---

## 1. 一句话结论

把对齐分数从余弦换成 TopoLa 并做尺度校准后:**Simulations 明确变好(ARI +0.037),Tonsil 的 ARI 小幅变好(+0.014),Lymph 明确变差(ARI −0.051),两个真实数据集的 iLISI 都下降 0.10 左右。** 按「必须比 Vanilla 全面更好」的标准,不达标,不建议现在开正式实验。

附带发现一个真 bug:`spamosaic/loss.py` 的 `CL_loss` 在分数尖锐时会下溢成 `-inf`,换 TopoLa 后必然触发,换余弦时侥幸没触发。详见第 6 节。

---

## 2. 要回答的问题

**主问题**:`examples/run_metrics.ipynb` 的三个数据集上,把 `train_utils.py` 里 `feat @ feat.T`(余弦,因 WLGCN 末层已 L2 归一)换成 TopoLa 分数 `D_λ = E(EᵀE+λI)⁻¹Eᵀ`,并做尺度校准,能否超过 Vanilla SpaMosaic。

**判定标准(用户给定)**:比 Vanilla 好 → 才做正式实验。这里「好」按主指标 ARI、F1、iLISI 综合判断。

**三个 arm**:

| arm | 分数 | 说明 |
|---|---|---|
| `dot` | `feat @ feat.T` | Vanilla 基线,应复现 notebook |
| `topola_cal` | `D / mean(diag(D))` | 本次主张的方案 |
| `topola_raw` | `D` | 诊断项,证明「不校准会坏」 |

---

## 3. 复现指南

### 3.1 环境与前置

- conda 环境 `spamosaic-env`(`/data/software/miniforge3/bin/conda run -n spamosaic-env`)。
- `R_HOME`:脚本里 `os.environ.setdefault('R_HOME', join(sys.prefix, 'lib', 'R'))`,自动推导,无需手工设置。若换机器跑请确认 `$CONDA_PREFIX/lib/R/bin/R` 存在。
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` 在 `import torch` 之前设置,否则确定性算法会报错。
- GPU:先 `nvidia-smi` 看空闲卡,最多占 2 张。本次用 `cuda:5`。
- 数据:`data/processed/`,已 gitignore,需自行准备(见 `AGENTS.md`)。
- 依赖只读:`reproduce/code/compute_metrics/metrics.py` 按目录加入 `sys.path` 后 import,不修改。

### 3.2 命令

冒烟(1 epoch,约 1 分钟,验证管线):

```bash
cd /data/home/jiboya/SpaMosaic
conda run --no-capture-output -n spamosaic-env python notes/quicktest_topola/run_quicktest.py \
    --device cuda:5 --datasets Simulations --variants dot topola_cal --epochs 1 --tag smoke
```

正式(3 数据集 × 3 arm,约 3 分钟训练 + 预处理):

```bash
conda run --no-capture-output -n spamosaic-env python notes/quicktest_topola/run_quicktest.py \
    --device cuda:5 --datasets Simulations Lymph-S1-S2-S3 Tonsil-S1-S2-S3 \
    --variants dot topola_cal topola_raw --epochs 100 --tag r2 \
    2>&1 | tee notes/quicktest_topola/log/run_r2.log
```

对比基线与汇总:

```bash
conda run --no-capture-output -n spamosaic-env python notes/quicktest_topola/compare.py --tag r2
```

分数尺度诊断(不训练,只看初始嵌入):

```bash
conda run --no-capture-output -n spamosaic-env python notes/quicktest_topola/diag_scores.py \
    --dataset Lymph-S1-S2-S3 --device cuda:5
```

单元测试(改 `spamosaic/` 后必跑):

```bash
conda run --no-capture-output -n spamosaic-env python tests/test_smoke.py
conda run --no-capture-output -n spamosaic-env python scripts/run_loss_experiments.py --smoke-test
```

### 3.3 输出与预期

`results/<tag>/` 下三个文件:

- `config.json`:完整参数快照。
- `metrics_long.csv`:长表,列含 `dataset, variant, metric, scope, value, epochs, seed, T, lambda_topola, train_seconds, n_kernel, diag_mean, off_mean, off_std, off_max`。
- `metrics_wide.csv`:透视表,行 `(dataset, metric, scope)`,列三个 arm。

复现应在下列噪声内一致(见 4.3):
- ARI ±0.005
- iLISI ±0.011
- F1 ±0.005

超出这个范围先查:GPU 是否同一张/同一型号、MNN 对数是否与日志一致、mclust 是否回退 kmeans。

### 3.4 复现时的坑(本次实际踩过)

1. **`conda run` + heredoc 会吞掉 stdin**。`conda run ... python - <<'PY'` 退出码 0 但什么都不执行(没有任何输出,极易误判为「代码无声通过」);去掉 `conda run` 直接跑同一条,又会因为系统 Python 缺包报 `ModuleNotFoundError`,两种表现都在误导排查。要么用 `python -c`,要么写成脚本文件。
2. **`R_HOME` 必须在 `import rpy2` 之前设置**。`utils.mclust_R` 内部已经做了兜底推导,但只对它自己那次调用有效;若有其他代码先 import rpy2,仍会报 `openrlib.R_HOME cannot be None`(exp001 就是这么崩的)。
3. **`torch.use_deterministic_algorithms(True)` 由 `set_seeds` 打开**,`CUBLAS_WORKSPACE_CONFIG` 必须在 `import torch` 前设。`diag_scores.py` 通过 `import run_quicktest` 间接拿到这个设置,所以会先打印 `[setup] framework.CL_loss -> StableCL_loss`,这是 import 副作用,不是错误。
4. **`diag_scores.py` 会修改全局 `CL_loss`**,因为它 import 了 `run_quicktest`。诊断本身不用损失函数,所以无影响,但不要在同一进程里继续做依赖原始 `CL_loss` 的事。
5. **mclust 失败会静默回退 kmeans**(`utils.clustering` 里 `except: print('mclust failed')`)。本次 `grep -c "mclust failed"` 为 0,三次都成功。复现时必须检查这一行,否则 ARI 会不可比。

---

## 4. 结果

### 4.1 核心指标(100 epochs,seed 1234)

`notebook` 列是 `examples/out/*/metrics.csv` 的既有基线。

| 数据集 | 指标 | notebook | dot | topola_cal | topola_raw | Δ(cal−dot) |
|---|---|---|---|---|---|---|
| Simulations | ARI | 0.8840 | 0.8832 | **0.9199** | 0.5923 | **+0.0367** |
| | F1 | 0.9353 | 0.9339 | 0.9437 | 0.6020 | +0.0098 |
| | iLISI | 0.7744 | 0.7771 | 0.7789 | 0.3211 | +0.0018 |
| Lymph | ARI | 0.2257 | 0.2209 | 0.1704 | 0.1813 | **−0.0505** |
| | F1 | 0.3059 | 0.3061 | 0.3183 | 0.3018 | +0.0122 |
| | iLISI | 0.7346 | 0.7358 | 0.6159 | 0.5963 | **−0.1198** |
| Tonsil | ARI | 0.1329 | 0.1330 | 0.1474 | 0.0981 | **+0.0144** |
| | F1 | 0.2324 | 0.2324 | 0.2341 | 0.2392 | +0.0017 |
| | iLISI | 0.8049 | 0.8049 | 0.7022 | 0.6889 | **−0.1027** |

### 4.2 全指标

**ARI(逐切片)**

| 数据集 | slice | dot | topola_cal | topola_raw |
|---|---|---|---|---|
| Simulations | 1 | 0.9421 | 0.9759 | 0.9861 |
| | 2 | 0.8670 | 0.9088 | 0.4848 |
| | 3 | 0.8450 | 0.8774 | 0.4584 |
| Lymph | 1 | 0.2171 | 0.1738 | 0.1984 |
| | 2 | 0.2150 | 0.1734 | 0.1857 |
| | 3 | 0.2075 | 0.1740 | 0.1748 |
| Tonsil | 1 | 0.1584 | 0.1562 | 0.1150 |
| | 2 | 0.1426 | 0.1569 | 0.1110 |
| | 3 | 0.1285 | 0.1327 | 0.0990 |

**PAS(越小越好)**

| 数据集 | slice | dot | topola_cal | topola_raw |
|---|---|---|---|---|
| Simulations | 1 | 0.0185 | 0.0147 | 0.0177 |
| | 2 | 0.0463 | 0.0370 | 0.1451 |
| | 3 | 0.0424 | 0.0432 | 0.1860 |
| Lymph | 1 | 0.2417 | 0.1493 | 0.2991 |
| | 2 | 0.3171 | 0.2096 | 0.4802 |
| | 3 | 0.2230 | 0.1511 | 0.2347 |
| Tonsil | 1 | 0.1012 | 0.0733 | 0.1257 |
| | 2 | 0.0974 | 0.0668 | 0.1461 |
| | 3 | 0.0849 | 0.0659 | 0.1099 |

**CHAOS(越小越好)**

| 数据集 | slice | dot | topola_cal | topola_raw |
|---|---|---|---|---|
| Simulations | 1 | 0.0973 | 0.0963 | 0.0969 |
| | 2 | 0.0998 | 0.1009 | 0.1016 |
| | 3 | 0.1017 | 0.0984 | 0.1034 |
| Lymph | 1 | 0.0651 | 0.0648 | 0.0682 |
| | 2 | 0.0689 | 0.0701 | 0.0776 |
| | 3 | 0.0671 | 0.0663 | 0.0684 |
| Tonsil | 1 | 0.0539 | 0.0537 | 0.0540 |
| | 2 | 0.0528 | 0.0528 | 0.0535 |
| | 3 | 0.0504 | 0.0501 | 0.0505 |

**F1(LabTransfer,仅 slice2–slice3)**

| 数据集 | dot | topola_cal | topola_raw |
|---|---|---|---|
| Simulations | 0.9339 | 0.9437 | 0.6020 |
| Lymph | 0.3061 | 0.3183 | 0.3018 |
| Tonsil | 0.2324 | 0.2341 | 0.2392 |

### 4.3 管线自检与噪声

**dot 臂复现基线的程度**(说明 harness 可信):

| 数据集 | 指标 | notebook | 本次 dot | 差 |
|---|---|---|---|---|
| Simulations | ARI | 0.8840 | 0.8832 | 0.0008 |
| Lymph | ARI | 0.2257 | 0.2209 | 0.0048 |
| Tonsil | ARI | 0.1329 | 0.1330 | 0.0001 |
| Tonsil | iLISI | 0.8049 | 0.8049 | <0.0001 |

Tonsil 几乎逐位一致;Lymph 的 PAS 差异最大(0.016),来源是 mclust 聚类标签的细微差异,不是分数实现。

**run-to-run 噪声**(r1 与 r2 同参数,仅损失实现不同;dot 臂数学等价,可视为重复):

| 臂 | 数据集 | 指标 | r1 | r2 | 差 |
|---|---|---|---|---|---|
| dot | Simulations | ARI | 0.8817 | 0.8832 | 0.0015 |
| dot | Simulations | iLISI | 0.7790 | 0.7771 | 0.0019 |
| cal | Simulations | ARI | 0.9149 | 0.9199 | 0.0050 |
| cal | Simulations | iLISI | 0.7841 | 0.7789 | 0.0052 |
| dot | Lymph | ARI | 0.2212 | 0.2209 | 0.0003 |
| dot | Lymph | iLISI | 0.7253 | 0.7358 | 0.0105 |

取上限:**ARI 噪声 ≈0.005,iLISI 噪声 ≈0.011**。因此 Lymph 的 ARI −0.051、两个数据集的 iLISI −0.10 都远超噪声,不是抖动。

噪声来源之一是 MNN 构图不确定:Lymph 的 adt MNN 对数在两次运行中分别是 **11250 和 11151**。`MNN.mnn(approx=True, way='annoy')` 走的是近似最近邻,即使设了 seed 也不保证完全一致。

### 4.4 与历史实验的一致性

- Simulations 的正向结果与 **exp004** 一致:exp004 多种子结论是「校准 TopoLa-CL ARI 0.905±0.010 vs CL 0.884±0.013」;本次单 seed 100 轮得到 0.9199 vs 0.8832,方向和量级都吻合。
- **Lymph / Tonsil 没有任何 TopoLa 先例**。exp001–004 只做 Simulations 和 MB-H3K4me3。所以这两个数据集上是首次观测,不能拿历史结论背书。
- `topola_raw` 全面崩溃与 exp001/exp002 的观察一致(Sim ARI 0.554,MB iLISI 0.395)。

---

## 5. 实现细节:到底改了什么

### 5.1 校准公式(唯一新增逻辑)

```python
def calibrated(features, regularization):
    D = _raw_topola(features, regularization)   # spamosaic.topola.full_topola_distance
    scale = D.diagonal().mean()
    return D / scale.detach()
tu.full_topola_distance = calibrated            # 仅在 quicktest 进程内 monkeypatch
```

几点约定:

- **为什么不除以 λ**:λ 是核的正则项,和分数尺度不是一回事。除以 `mean(diag(D))` 的道理是:余弦的自相似度为 1,TopoLa 的自亲和度 `D_ii` 就是这个尺度,除回去两者可比,notebook 里的 `T` 无需改动。
- **`detach()`**:缩放常数当作常数,不参与梯度。梯度的相对结构不变,只是整体缩放。
- **校准后 `mean(diag) = 1`**,但**逐行**对角并不等于 1(见 5.4),这是 H2 的由来。

### 5.2 稳定版 CL(为了跑通,不是方案的一部分)

见第 6 节。两个 arm 都用稳定版,保证公平;dot 臂结果与 notebook 一致,说明这个替换在不下溢时数值等价。

### 5.3 未改动的部分

- 预处理、构图、MNN、WLGCN、优化器、`w_g=0.8`、`lr=0.01`、`intra_knns=10`、`inter_knn_base=10`、`n_epochs=100`、`seed=1234` 全部沿用 notebook。
- `T` 沿用 notebook 原值:Sim `0.1`,Lymph/Tonsil `0.01`。**没有为 TopoLa 调温度**,这是有意的最小改动。
- `λ_topola = 10`(代码默认值),未扫。
- `spamosaic/` 包零改动;所有 monkeypatch 只在测试进程内。

### 5.4 初始嵌入的分数尺度诊断(`diag_scores.py`,λ=10)

| 数据集 | 核 n | `mean(diag D)` | `D_ii/mean` 范围 | cal 非对角 std | 余弦非对角 std | gap std 比值 |
|---|---|---|---|---|---|---|
| Simulations | 2592 | 1.0261e-2 | 0.535–1.766 | 0.1766 | 0.2394 | 0.792 |
| Lymph | 6968 | 4.2818e-3 | 0.465–1.755 | 0.1764 | 0.2320 | 0.805 |
| Tonsil | 8674 | 3.4590e-3 | 0.415–2.058 | 0.1765 | 0.2528 | 0.764 |

四条要紧的观察:

1. **校准后的非对角 std 恒等于 0.1766 ≈ 1/√32**,而嵌入维度 d=32。这说明分数校准确实把 TopoLa 拉回了「单位球面上的随机余弦」量级,与数据集无关。
2. **余弦的非对角 std 反而更大(0.232–0.253)**。真实嵌入是各向异性、成簇的,所以余弦分布比各向同性基准更宽。也就是说 diag 校准**没有精确对齐余弦**:按单格分数算,校准后的 TopoLa 比余弦窄 24–30%(0.1766 / 0.232–0.253);按 CL 真正用到的量(正样本减负样本的 gap)算,窄 19–24%(比值 0.76–0.81)。
   后者直接给出 exp004 式温度校准的系数:`T_cal = T_dot × ratio`,等价于把校准分数再乘 1.24–1.31。**这是最值得先试的一项(见 H1)。**
3. **逐行对角异质性显著**:`D_ii/mean` 从 0.42 到 2.06。全局缩放后「自相似度 = 1」只在平均意义上成立;余弦是逐行严格等于 1。这是 H2。
4. **校准后存在非对角分数高于自相似的单元格**:最大非对角 1.05 / 1.17 / 1.34,而余弦最大非对角 0.94–0.96。TopoLa 的 CL 几何因此和余弦不同:某些跨模态对可以「比自己更像自己」,这些行不再提供正的排序压力。

---

## 6. 关键发现:`CL_loss` 会下溢成 `-inf`(独立 bug)

### 6.1 现象

第一版(r1)在 **Lymph 的 `topola_cal` 第 4 个 epoch** 崩溃:

```
File "spamosaic/topola.py", line 63, in full_topola_distance
    raise FloatingPointError('Nonfinite full TopoLa distance')
```

报错位置在 TopoLa,但 **TopoLa 不是原因**,它只是第一个检查有限性的地方。

### 6.2 机制

`spamosaic/loss.py` 的 `CL_loss.forward`:

```python
simi_max, _ = torch.max(simi, dim=1, keepdim=True)   # 对角自相似度最大
simi = simi - simi_max.detach()
negatives = (torch.exp(simi) * self.negatives_mask).sum(dim=1)
loss = -(positives - torch.log(negatives + self.bias)).mean()
```

`simi_max` 是对角(自相似度),而它在 `negatives_mask` 里被排除了。因此当某一行「对角 − 最优非对角」超过约 `87.3 × T` 时,`exp` 在 float32 下全部下溢为 0,`negatives = 0`,`log(0) = -inf`,损失 `+inf`,梯度 `NaN`,下一步编码器输出 `NaN`,TopoLa 前向检查有限性时抛错。

**一行 `-inf` 就足以毁掉整个 batch 的均值。**

### 6.3 为什么余弦不发作、TopoLa 必发作

同一批初始嵌入上的统计(λ=10,T 取 notebook 值):

| 数据集 | 校准 TopoLa:`margin/T > 87.3` 的行数(T=0.01) | 余弦 |
|---|---|---|
| Simulations | 1 / 2592 | 0 / 2592 |
| Lymph | 2 / 6968 | 0 / 6968 |
| Tonsil | 3 / 8674 | 0 / 8674 |

余弦的自相似度和最优非对角非常接近(最大非对角 0.94–0.96),margin 极小;校准 TopoLa 的 margin 中位数 0.24–0.31、最大值接近 1,在 `T=0.01` 下放大 100 倍,轻松越过阈值。初始时只有个位数行越界,训练几轮分数变尖后行数增加,于是第 4 个 epoch 崩。

### 6.4 最小复现

```python
import torch
from spamosaic.loss import CL_loss
s = torch.full((6968, 6968), -0.5)
s.fill_diagonal_(1.0)
s.requires_grad_(True)
out = CL_loss(3484, rep=2)(s / 0.01)   # -inf
out.backward()
torch.isfinite(s.grad).all()           # False
```

### 6.5 数学等价的修复

```python
class StableCL_loss(CL_loss):
    def forward(self, simi):
        positives = (simi * self.pos_mask).sum(dim=1) / self.pos_mask.sum(dim=1)
        negatives = torch.logsumexp(simi.masked_fill(self.negatives_mask == 0, -torch.inf), dim=1)
        if self.bias:
            negatives = torch.logaddexp(
                negatives, torch.log(torch.tensor(float(self.bias), device=simi.device)))
        return -(positives - negatives).mean()
```

等价性:原式把逐行常数 `simi_max` 从 `positives` 和 `log(sum exp)` 里各减一次,两项抵消;`logsumexp` 自带最大值平移,所以两者数学上相同。`log(sum + bias)` 用 `logaddexp` 保持等价。本次 `bias=0` 走不到该分支。

### 6.6 为什么没有直接改包

虽然数学等价,但它会**改变所有 CL 实验在原本下溢情形下的数值**:原本 `-inf`/NaN 直接失败(或被静默跳过),修复后变成有限损失继续训练。这是一个独立的行为变更,应该单独立项并配 `tests/test_smoke.py` 的新增用例,不宜混在「换分数」这件小事里。

建议的独立处理:
- 在 `tests/test_smoke.py` 加一条:`CL_loss` 在尖锐分数下必须返回有限值和有限梯度。
- 修改 `loss.py`,跑 `python tests/test_smoke.py` 和 `scripts/run_loss_experiments.py --smoke-test`。
- 若复跑 exp001–004 的既有结论,注意数值可能变化(预期只在原本就下溢的配置上有差异)。

---

## 7. 结果解读与待查假设(给复现者的清单)

按优先级排序,每条都写明「怎么试、预期什么、怎么看」。

### H1 温度/梯度尺度没有真正对齐(最可能,建议先做)

**依据**:5.4 的 gap std 比值 0.76–0.81,即进入损失的 logit 比余弦小 19–24%。diag 校准只让平均尺度可比,没有对齐 CL 实际使用的 gap 分布。CL 的锐度直接决定对齐强度,软掉的对齐项会**降低跨切片混合**,正好对应 iLISI 下降。

**怎么做**:两种等价写法任选
- 校准常数改成 `mean(diag) * ratio`(即分数再乘 1.24–1.31),或
- 保持校准不变,把 `T` 改成 `T_dot × ratio`(exp004 的做法)。

更彻底的是 exp004 的联合校准:温度用 gap std 比值,再按 `‖∇L_dot‖/‖∇L_topo‖` 缩放对齐项权重(`alpha`),因为分数尺度对齐了,梯度范数未必。

**预期**:iLISI 回升,ARI 可能略降。若 iLISI 回到 baseline 附近且 ARI 仍有优势,这个方向就成立。

**注意**:不要同时改 λ,否则无法归因。

### H2 逐行对角异质(结构性差异,值得一试)

**依据**:`D_ii/mean` 分布 0.42–2.06。余弦逐行自相似度恒为 1,全局缩放后的 TopoLa 不是。

**怎么做**:改成逐行归一 `D / D.diagonal()[:, None]`(detach)。这样每行自相似度严格为 1,与余弦的几何一致。

**预期**:改变不同 spot 在损失中的相对权重。若 H2 是主因,这一项会同时改善 ARI 和 iLISI;若无效,说明异质性不是关键。

**风险**:逐行归一后分数的整体尺度会变,需与 H1 一起或分开各测一次,避免混淆。

### H3 λ 的选择(形状 vs 尺度)

**依据**:exp003 在 Sim 嵌入上测得 `ρ(H)=345–411`、`eig_min≈19–20`(d=32,行归一)。所以 λ ≤ 19 属于「精确投影区」,λ=1000 才进入平滑区。λ=10 在投影区内。校准已经把 λ 的**整体尺度效应**除掉了,所以校准后再扫 λ,扫的是**谱加权形状**,这正是有意义的那个维度。

**怎么做**:`λ ∈ {1e-3, 0.1, 1, 10, 100, 1000}`,配合 H1 的校准,固定 T。

**预期**:小 λ(白化强)更强调全局主方向(簇级结构),可能抬高 ARI/PAS、压低 iLISI;大 λ 退化为余弦缩放,应趋近 dot 臂。这个单调趋势本身就能验证「iLISI 下降来自白化」。

**验证方式**:看 λ→大 时 iLISI 是否单调回到 0.73/0.80。

### H4 核的参考集合与范围

本次核范围 = 整个桥接 batch 拼接后的全部行(Lymph 6968、Tonsil 8674、Sim 2592)。这符合笔记中「正负分数共享同一参考集合」的要求。但要注意:

- `use_mini_thr=8000` 是按 **bridge spot 数**判断,不是按拼接后的行数。Tonsil 桥接 4337 < 8000,所以没有分块,一次核 8674 行。若某个数据集桥接 spot 超过 8000,会切成 1024×2 行的小核,**语义会变**,必须显式记录。
- exp004 做过「固定/随机/空间参考」和「全桥接参考」的消融,结论是「无证据支持全局参考必然更差」,但那是 Sim 和 MB 的结论,不能外推到 Lymph/Tonsil。

**怎么做**:固定 512/1024/2048 的参考集合,与全 batch 比。若 iLISI 对参考规模敏感,说明问题在核范围而不是分数本身。

### H5 ARI/PAS 与 iLISI 的张力:是缺陷还是取舍

**观察**:TopoLa 一致改善 PAS(Lymph 0.242→0.149,Tonsil 0.101→0.073),ARI 在 Sim/Tonsil 改善,F1 不降;但 iLISI 在两个真实数据集下降约 0.10。

**两种解释**:
1. **整合真的变差**:TopoLa 把 spot 推向簇级结构,牺牲了跨切片一一对应。
2. **iLISI 对簇结构不敏感**:iLISI 衡量邻域里 src 的混合程度,当嵌入形成紧密簇时,同簇内仍可混合良好,但跨簇邻域变少,数值下降不代表对齐失败。

**怎么区分**:在 Lymph 上直接查「MNN 保留率」或「slice2↔slice3 的 kNN 命中率」(不需要新指标,用现有 `build_graph` 的 MNN 对数即可),以及 F1(已有,TopoLa 反而略高 0.318 vs 0.306)。F1 升高而 iLISI 下降,更支持解释 2。**如果确认是解释 2,那么当前结果其实是可以接受的**,只是不能用 iLISI 单独下结论。

这是本次最需要人判断的一点,建议复现时优先做这项诊断。

### H6 单 seed

所有结论都是 seed=1234 单次。噪声估计(4.3)显示 ARI 抖动约 0.005,所以 Lymph 的 −0.05 大概率是真的;但 Tonsil 的 +0.014 只有噪声的 3 倍,必须多 seed 才能定论。正式实验至少 4 个独立 seed,报均值±标准差。

---

## 8. 已排除的可能

复现时若怀疑以下方向,可以先跳过:

1. **不是 TopoLa 数值实现的问题**。`tests/test_smoke.py` 覆盖前向/反向;`scripts/run_loss_experiments.py --smoke-test` 覆盖 QR 与闭式的等价性、`gradcheck`、BPR 参考值。r2 全程无 `Nonfinite`。
2. **不是 NaN 残留导致的**。r1 的崩溃已定位到 `CL_loss` 下溢(第 6 节),r2 修复后三个数据集全部跑完,`grep -cE "Nonfinite|Traceback"` 为 0。
3. **不是 harness 与 notebook 不一致**。dot 臂几乎逐位复现基线(4.3)。
4. **不是 mclust 回退**。两个真实数据集的日志里没有 `mclust failed`,R 6.0.0 正常加载。
5. **不是 R_HOME / 环境问题**。r2 无相关报错。
6. **不是 OOM**。峰值显存与内存见第 9 节,远低于单卡 48GB。

---

## 9. 性能与资源

**单次核计算**(CPU,16 线程,d=32):

| 行数 2N | 前向 | 反向 | `D` float32 |
|---|---|---|---|
| 2592(Sim) | 0.01 s | 0.01 s | 26 MB |
| 6968(Lymph) | 0.05 s | 0.04 s | 194 MB |
| 8674(Tonsil) | 0.06 s | 0.06 s | 301 MB |

`full_topola_distance` 内部会把 `E` 转到 CPU float64,`D` 的 float64 中间量是 float32 的 2 倍(Tonsil 约 602 MB),峰值约 0.9 GB,对 3 TB 内存的机器不是问题。**真正的限制是 `D` 是 `[2N,2N]` 稠密矩阵**:VisiumHD 那种 9.9 万点(拼接后约 19.8 万行)的 float64 中间量约 310 GB、float32 结果约 156 GB,单次核就吃光内存,必须靠 mini-batch 限制核规模。

**端到端 100 epochs**(含每轮 CPU/GPU 往返):

| 数据集 | dot | topola_cal | 倍数 |
|---|---|---|---|
| Simulations | 1.0 s | 4.7 s | 4.7× |
| Lymph | 4.6 s | 23.4 s | 5.1× |
| Tonsil | 4.1 s | 41.4 s | 10.1× |

绝对时间都很短,TopoLa 的 5–10 倍开销不是决策因素。三个数据集全部重跑约 2 分钟训练 + 3 分钟预处理。

**显存**:输入特征 50 维,`out_dim=32`,单卡占用 <2 GB,与 `AGENTS.md` 的描述一致。

---

## 10. 下一步与门槛建议

**建议顺序**:

1. **先独立修 `CL_loss`**(第 6 节),加测试,单独提交。不需要等这个方向的决定。
2. **复现 r2**,确认数字落在噪声内(ARI ±0.005、iLISI ±0.011)。若不一致,先查 4.3 的三个排查项。
3. **做 H5 的诊断**(MNN 保留率 / kNN 命中率),判断 iLISI 下降是整合变差还是指标不敏感。这一项成本最低、信息量最大。
4. **做 H1**(温度/梯度校准),因为依据最硬(有 0.76–0.81 的实测比值)。单数据集(Lymph,负例)先验证能否救回 iLISI。
5. 若 H1 有效,再上 4 seeds × 3 datasets 的正式评估,视情况加 H2/H3 消融。
6. 只有全部门槛达标,才开 exp005 并写 `README.md` / `findings.md` / `index.csv`。

**建议的正式门槛**(供讨论):

- 两个真实数据集的 iLISI 相对 dot 的下降 ≤ 0.02(当前 −0.10,差距很大)。
- Simulations 的 ARI 相对 dot 提升 ≥ 0.02,且 4 seeds 的均值±标准差区间不重叠。
- F1、PAS 不出现显著退化。
- 同一配置下 λ 与 T 的选择必须在测试集之外(或至少报告敏感性)。

**不建议**在没有解决 iLISI 之前直接开正式实验,否则只是把「不达标」写进 `index.csv`。

---

## 11. 文件清单

```
notes/quicktest_topola/
├── insight.md              本文档
├── run_quicktest.py        主脚本(3 arm × 3 数据集,含校准与稳定 CL)
├── compare.py              与 examples/out/*/metrics.csv 基线对比
├── diag_scores.py          分数尺度/对角异质/gap 比值/下溢余量诊断(不训练)
├── log/
│   ├── run_r1.log          第一版:Sim 跑完,Lymph cal 第 4 epoch 崩溃(下溢证据)
│   └── run_r2.log          第二版:修复后完整运行
└── results/
    ├── r1/                 第一版部分结果(Sim 三臂 + Lymph dot)
    └── r2/                 完整结果:config.json / metrics_long.csv / metrics_wide.csv
```

**版本对照**:

- `r1` = 原始 `CL_loss`,Sim 三臂 + Lymph dot 有效,`topola_cal` 在 Lymph 崩溃。
- `r2` = 稳定 `CL_loss`,三个数据集九次运行全部完成。**本文所有结论取自 r2。**

**未纳入版本控制的部分**:`results/*/` 尚未被 `.gitignore` 覆盖(`notes/` 不在忽略列表内),提交前自行决定是否入库;`examples/out/` 已被忽略。
