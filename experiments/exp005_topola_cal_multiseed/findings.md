# exp005 findings

配置:`dot`(Vanilla,余弦)vs `topola_cal`(`D/mean(diag D)`,λ=10,T 沿用 notebook);
3 数据集 × 2 arm × 5 seed(1234/2024/2025/2026/2027),100 轮;两个 arm 都用 `StableCL_loss`。
共 30 次训练全部完成,日志中 `mclust failed` 为 0,`Traceback`/`Nonfinite` 为 0。
数据:`results/all_metrics.csv`(逐 run)、`summary.csv`(均值±std)、`paired.csv`(逐 seed 配对差)。

## 核心结果(5 seed 均值±std;配对差 = topola_cal − dot;n_pos/n_neg 为 5 个 seed 中差为正/负的个数)

| 数据集 | 指标 | dot | topola_cal | 配对差 | 正/负 |
|---|---|---|---|---|---|
| Simulations | ARI | 0.8807±0.0079 | 0.9160±0.0038 | +0.0353 | 5/0 |
| | F1 | 0.9347±0.0033 | 0.9443±0.0052 | +0.0095 | 4/1 |
| | iLISI | 0.7812±0.0069 | 0.7788±0.0148 | −0.0024 | 1/4 |
| Lymph | ARI | 0.2276±0.0107 | 0.1859±0.0145 | −0.0416 | 0/5 |
| | F1 | 0.3120±0.0044 | 0.3245±0.0101 | +0.0124 | 4/1 |
| | iLISI | 0.7265±0.0143 | 0.6246±0.0108 | −0.1018 | 0/5 |
| Tonsil | ARI | 0.1206±0.0125 | 0.1694±0.0230 | +0.0488 | 5/0 |
| | F1 | 0.2331±0.0006 | 0.2356±0.0019 | +0.0026 | 5/0 |
| | iLISI | 0.8007±0.0042 | 0.7162±0.0118 | −0.0845 | 0/5 |

PAS(越小越好,逐切片均值):topola_cal 在 Lymph(0.138/0.181/0.130 vs 0.205/0.277/0.199)、
Tonsil(0.075/0.077/0.067 vs 0.096/0.096/0.088)、Simulations(0.014/0.036/0.047 vs 0.018/0.049/0.053)
三个数据集的三个切片上均低于 dot。CHAOS 两 arm 差异 ≤0.002。

## 按 README 判定标准

- 「iLISI 下降」:成立。Lymph −0.102、Tonsil −0.085,均 < −0.02,且各 5/5 个 seed 为负。
- 「Simulations ARI 提升」:成立。配对差 +0.035 ≥ 0.02,5/5 个 seed 为正。
- 「全面优于 Vanilla」:不成立。Lymph 的 ARI(−0.042)与两个真实数据集的 iLISI 配对差均值为负且超出 0.02 容差。
- Tonsil 的 ARI(H_C):+0.049,5/5 个 seed 为正;高于 quicktest 单 seed 的 +0.014。

## 与 quicktest(seed 1234,r2)的对照

- Simulations ARI:本实验 seed 1234 的 dot/topola_cal 为 0.8847/0.9179,quicktest 为 0.8832/0.9199,差 ≤0.002。
- Lymph/Tonsil 的 iLISI 方向与幅度与 quicktest 一致(Lymph −0.120→5 seed 均值 −0.102;Tonsil −0.103→−0.085)。
- Tonsil dot 的 5 seed ARI 均值 0.1206 低于 notebook 值 0.1329(quicktest seed 1234 为 0.1330),说明该数据集上 dot 的 ARI 对 seed 敏感(std 0.0125);seed 1234 的 Tonsil topola_cal 配对差(+0.014)落在 5 个 seed 配对差(均值 +0.049,std 0.0245)的低端。

## 分析

- 稳定可靠的两个现象:topola_cal 在 Simulations 与 Tonsil 上提高 ARI;在两个真实数据集上降低 iLISI(约 0.08–0.10,是 seed 间 std 的 6–8 倍)。
- 同时 PAS 在全部数据集上改善,F1 非负,Lymph 的 ARI 下降而 F1 上升(+0.012)。ARI 与 iLISI 的方向在 Lymph 上同时为负。
- Simulations 的 iLISI 无变化(−0.002,不显著),因此 iLISI 下降只出现在两个真实数据集上。
- 本实验未测量 iLISI 下降是整合变差还是指标对簇结构不敏感,没有对机制作结论。
- 训练耗时:topola_cal 约为 dot 的 5–10 倍(见 `all_metrics.csv` 的 `train_seconds`)。
