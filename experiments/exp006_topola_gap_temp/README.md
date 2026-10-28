# exp006 topola_gap_temp

## 问题

exp005 显示校准 TopoLa(`topola_cal`)在 Lymph / Tonsil 上 iLISI 稳定下降(−0.102 / −0.085,各 0/5 seed 为正)。
`notes/quicktest_topola/insight.md` 的 H1 认为原因之一是温度没有对齐:按「正对减负对」的 gap 标准差计,
校准 TopoLa 的 logit 比余弦窄约 19–24%(初始嵌入上比值 0.76–0.81),对齐变软,跨切片混合变弱。
把校准分数乘以该比值的倒数(等价于 T 乘该比值),能否恢复 iLISI 且保留 ARI 优势?

## 假设

H_1:`topola_cal_gap` 相对 `topola_cal`,Lymph / Tonsil 的 iLISI 上升,并回到 dot 附近(与 dot 差 ≤0.02)。
事先不假设 ARI 的方向。若 iLISI 无变化,则温度失配不是 iLISI 下降的主因。

## 唯一改动的因素

在 `topola_cal` 上再乘常数 `k`(分数放大):

`score = D / mean(diag D) * k`,`k = std(余弦 gap) / std(校准 TopoLa gap)`。

gap = 同一行里「正对分数 − 负对分数」(负对指除自身和正对以外的所有桥接核内格子)。
`k` 在每次 run 的第一个训练步用初始嵌入测一次,之后固定,不随训练更新。
其余全部不变:λ=10、T(Simulations 0.1;Lymph / Tonsil 0.01)、100 轮、`StableCL_loss`、预处理与构图。

## 基线

沿用 exp005 的 `dot` 与 `topola_cal` 结果(同数据、同 5 个 seed、同配置),本实验只跑新 arm。
exp005 的 `all_metrics.csv` 为对照来源。

## 设计

- 3 数据集 × 1 个新 arm × 5 seed(1234、2024、2025、2026、2027)= 15 次训练。单卡 cuda:2。
- 指标与 exp005 相同:ARI、F1、PAS、CHAOS、iLISI。
- 每次 run 的 `k` 记录在 `all_metrics.csv` 的 `gap_gain` 列。

## 判定标准(事先给定)

按数据集,逐 seed 与 exp005 配对:

- 「H_1 成立」:Lymph 与 Tonsil 的 iLISI 相对 `topola_cal` 的配对差均值 ≥ +0.02 且至少 4/5 个 seed 为正,
  并且相对 `dot` 的差均值 ≥ −0.02。
- 「部分恢复」:相对 `topola_cal` 的 iLISI 配对差均值 ≥ +0.02,但相对 `dot` 仍 < −0.02。
- 「无效」:相对 `topola_cal` 的 iLISI 配对差均值 < +0.02。
- 同时报告 ARI、F1、PAS 相对 `topola_cal` 与 `dot` 的配对差。
- 日志中出现 `mclust failed` 或 `Traceback` 的 run 不计入并重跑(只修 bug)。
