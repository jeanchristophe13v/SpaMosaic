# exp006 findings

配置:`topola_cal_gap` = `D/mean(diag D) * k`,`k` 为每次 run 第一个训练步测得的初始 gap 标准差比值(余弦 / 校准 TopoLa),之后固定;
其余与 exp005 的 `topola_cal` 完全一致(λ=10,T 沿用 notebook,100 轮,`StableCL_loss`)。
3 数据集 × 5 seed(1234/2024/2025/2026/2027)= 15 次训练,全部完成;`mclust failed` 为 0,`Traceback`/`Nonfinite` 为 0。
基线 `dot` 与 `topola_cal` 取自 exp005 的 `results/all_metrics.csv`(同 seed 配对)。
数据:`results/all_metrics.csv`(新 arm 逐 run,含 `gap_gain`)、`paired_vs_exp005.csv`(配对差)、`means_3arms.csv`。

## 测得的 k

| 数据集 | k 最小 | k 均值 | k 最大 |
|---|---|---|---|
| Simulations | 1.262 | 1.301 | 1.379 |
| Lymph | 1.217 | 1.238 | 1.266 |
| Tonsil | 1.308 | 1.345 | 1.361 |

## 核心结果(5 seed 均值)

| 数据集 | 指标 | dot | topola_cal | topola_cal_gap |
|---|---|---|---|---|
| Simulations | ARI | 0.8807 | 0.9160 | 0.9032 |
| | F1 | 0.9347 | 0.9443 | 0.9456 |
| | iLISI | 0.7812 | 0.7788 | 0.7713 |
| Lymph | ARI | 0.2276 | 0.1859 | 0.1870 |
| | F1 | 0.3120 | 0.3245 | 0.3284 |
| | iLISI | 0.7265 | 0.6246 | 0.6284 |
| Tonsil | ARI | 0.1206 | 0.1694 | 0.1471 |
| | F1 | 0.2331 | 0.2356 | 0.2358 |
| | iLISI | 0.8007 | 0.7162 | 0.7131 |

## 配对差(topola_cal_gap − 参照;n_pos/n_neg 为 5 个 seed 中为正/负的个数)

相对 `topola_cal`:

| 数据集 | 指标 | 配对差 | 正/负 |
|---|---|---|---|
| Lymph | iLISI | +0.0038 | 3/2 |
| | ARI | +0.0011 | 2/3 |
| | F1 | +0.0039 | 3/2 |
| Tonsil | iLISI | −0.0031 | 1/4 |
| | ARI | −0.0223(std 0.0604) | 2/3 |
| | F1 | +0.0002 | 3/2 |
| Simulations | iLISI | −0.0075 | 2/3 |
| | ARI | −0.0128 | 0/5 |
| | F1 | +0.0013 | 4/1 |

相对 `dot`:

| 数据集 | 指标 | 配对差 | 正/负 |
|---|---|---|---|
| Lymph | iLISI | −0.0981 | 0/5 |
| | ARI | −0.0405 | 0/5 |
| Tonsil | iLISI | −0.0876 | 0/5 |
| | ARI | +0.0265(std 0.0606) | 4/1 |
| Simulations | ARI | +0.0225 | 5/0 |
| | iLISI | −0.0099 | 1/4 |

## 按 README 判定标准

- 「H_1 成立」:不成立。Lymph 相对 `topola_cal` 的 iLISI 配对差 +0.004,Tonsil −0.003,均 < +0.02。
- 「部分恢复」:不成立,同样因为相对 `topola_cal` 的差 < +0.02。
- 「无效」:成立。两个真实数据集的 iLISI 与 `topola_cal` 的差都在 seed 间波动内(std 0.004–0.009),
  相对 `dot` 仍分别下降 0.098 和 0.088。

## 分析

- 把分数放大约 1.24–1.35 倍(即温度缩小到约 0.74–0.81)没有改变 iLISI。
  因此在本配置下,初始 gap 宽度失配不是 iLISI 下降的主要原因。
- Simulations 的 ARI 相对 `topola_cal` 下降 0.013(5/5 个 seed 为负),仍高于 `dot` 0.023(5/5 为正);
  即放大分数削弱了 exp005 中的 ARI 提升,但没有消除。
- Tonsil 的 ARI 在新 arm 下 seed 间 std 为 0.066(`topola_cal` 为 0.023,`dot` 为 0.013),配对差的符号不稳定(2/3)。
- Lymph 的 ARI、F1、PAS 与 `topola_cal` 接近(PAS 三个切片 0.134/0.174/0.124,`topola_cal` 为 0.138/0.181/0.130)。
- 本实验只改变了固定的标量 `k`;逐行对角异质(H2)、λ(H3)、核范围(H4)未涉及,没有对这些因素作结论。

## 勘误(2026-10-29,见 exp009)

- 评估时没有统一 Lymph/Tonsil 各切片的标签拼写。表中 Lymph/Tonsil 的 ARI 与 F1 是未统一标签的值(所有 arm 一样受影响);iLISI 与 Simulations 的指标不受影响。
  本实验未保存嵌入，无法用统一标签重算。判定依据是 iLISI,所以「H_1 不成立」「无效」两条结论不受标签问题影响。
- 预处理中的 Harmony 不确定，不同进程的预处理结果与 MNN 图不同。本实验的新 arm 与 exp005 的基线来自不同进程，配对差中含有预处理随机性。
  同一配置跨进程重跑(exp009 对 exp005)的 5 seed 均值差:iLISI 为 −0.005 到 +0.004,Simulations ARI 为 −0.000 到 +0.005;逐 run 的 Simulations ARI 差最大约 0.011。
  相对 topola_cal 的 iLISI 配对差(+0.004/−0.003)落在这个范围内，与「无效」一致。Simulations ARI 的 −0.013(0/5)只略大于逐 run 跨进程差，结论偏弱。
