# exp005 topola_cal_multiseed

## 问题

把 `examples/run_metrics.ipynb` 三个数据集(Simulations / Lymph-S1-S2-S3 / Tonsil-S1-S2-S3)里
CL 的对齐分数从余弦(`feat @ feat.T`)换成**尺度校准的 TopoLa 分数**
`D_λ / mean(diag(D_λ))`,`D_λ = E(EᵀE+λI)⁻¹Eᵀ`,能否稳定地比 Vanilla SpaMosaic 更好?

来源:`notes/quicktest_topola/insight.md` 的单 seed 快速验证(seed 1234,r2)。
该验证未开正式实验,结论是 Simulations 变好、Lymph 变差、两个真实数据集 iLISI 下降约 0.10。
本实验用多 seed 复核这些方向,并给出可入库的正式数字。

## 假设

- H_A:Simulations 上 topola_cal 的 ARI 高于 dot(与 exp004 的 100 轮结果同向)。
- H_B:Lymph / Tonsil 上 topola_cal 的 iLISI 低于 dot,下降幅度远大于 seed 间波动。
- H_C:Tonsil 上 topola_cal 的 ARI 提升(单 seed +0.014)是否在多 seed 下仍成立,事先不假设方向。

## 唯一改动的因素

对齐分数:`dot`(余弦)→ `topola_cal`(校准 TopoLa)。其余完全一致:
预处理、构图、WLGCN、`w_g=0.8`、`lr=0.01`、`n_epochs=100`、`intra_knns=10`、`inter_knn_base=10`、
`T`(Simulations 0.1;Lymph / Tonsil 0.01,沿用 notebook)、`λ=10`(代码默认值,不扫)。

两个 arm 都使用 `StableCL_loss`(masked logsumexp,与原 `CL_loss` 数学等价,只在原实现下溢为
`-inf` 时才有数值差异)。该替换只在 driver 内 monkeypatch,`spamosaic/` 包未修改。

## 基线

`dot` arm 即 Vanilla;配置沿用 `examples/run_metrics.ipynb`。单 seed 1234 下 quicktest 已复现
notebook 的 ARI(差 ≤0.005)。校准方案沿用 `notes/quicktest_topola/run_quicktest.py` 的 `topola_cal`。

## 设计

- 3 数据集 × 2 arm × 5 seed(1234、2024、2025、2026、2027)= 30 次训练。
- seed 同时控制 `set_seeds` 与 `SpaMosaic(seed=...)`(MNN 构图使用该 seed)。mclust 内部固定 `random_seed=2020`。
- 指标:ARI(all 与逐切片)、F1(LabTransfer slice2-slice3)、PAS、CHAOS、iLISI,实现取自
  `reproduce/code/compute_metrics/metrics.py`(只读)。
- 单卡运行(cuda:2)。

## 判定标准(事先给定)

按数据集,用 5 seed 的均值±标准差,以及逐 seed 配对差(topola_cal − dot)判断:

- 「iLISI 下降」成立:两个真实数据集的配对差均值 < −0.02,且 5 个 seed 中至少 4 个为负。
- 「Simulations ARI 提升」成立:配对差均值 ≥ 0.02,且 5 个 seed 全部为正。
- 「全面优于 Vanilla」要求:三个数据集的 ARI、F1、iLISI 配对差均值都不为负(容差 0.02 内视作持平)。
  不满足则 topola_cal 不构成对 Vanilla 的无条件改进。
- 逐 run 检查日志中无 `mclust failed`、无 `Traceback`;出现则该 run 不计入并重跑(只修 bug)。
