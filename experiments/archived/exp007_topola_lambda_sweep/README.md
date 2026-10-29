# exp007 topola_lambda_sweep

## 问题

exp005 显示校准 TopoLa(λ=10)在 Lymph / Tonsil 上 iLISI 稳定下降(−0.102 / −0.085);exp006 显示把分数整体放大(温度失配)不能恢复。
本实验检验另一个候选原因:TopoLa 的「白化」形状(λ 控制的谱加权)。

数学依据:`D_λ = E(EᵀE+λI)⁻¹Eᵀ`。λ 远小于 `EᵀE` 的特征值时 `D_λ` 近似投影,损失对联合线性变换 `E→EA` 近似不变(λ=1e-6 实测相对误差 4e-5;λ=10 时不再不变),
即损失不约束嵌入的各向异性,而评估(kNN、聚类、iLISI)用的是原始嵌入几何。
λ 远大于 `EᵀE` 的最大特征值时 `D_λ ≈ E Eᵀ/λ`;E 逐行 L2 归一,故 `mean(diag D_λ) ≈ 1/λ`,校准后恰为余弦,即回到 dot。
因此校准后的 λ 扫描是从「白化」平滑过渡到 dot 的路径,λ 足够大时应复现 dot(自检)。

## 假设

H_3:校准 TopoLa 的 iLISI 随 λ 增大单调回升,在 λ 足够大时回到 dot 附近;
同时 exp005 中 Simulations / Tonsil 的 ARI 提升随 λ 增大而消失。
若 iLISI 在 λ 全范围内不随 λ 变化,则白化形状不是 iLISI 下降的原因。

## 唯一改动的因素

λ:`{1, 100, 1000, 10000, 100000}`(新跑)加上 exp005 已有的 λ=10。
其余全部沿用 exp005 的 `topola_cal`:分数 `D_λ/mean(diag D_λ)`(无 exp006 的 k)、T(Simulations 0.1;Lymph / Tonsil 0.01)、100 轮、`StableCL_loss`、预处理与构图。

## 基线

exp005 的 `dot` 与 `topola_cal`(λ=10),同数据、同 5 个 seed(1234、2024、2025、2026、2027)逐 seed 配对。本实验只跑新的 λ 值。

## 设计

- 3 数据集 × 5 个新 λ × 5 seed = 75 次训练。
- 两张卡并行(cuda:2 跑 Tonsil;cuda:5 跑 Lymph 与 Simulations)。
- 每个 run 在首个训练步记录 `EᵀE` 的 `eig_min`、`eig_max` 与 `tr(D_λ)`,用于判断 λ 相对谱的位置。
- 指标:ARI、F1、PAS、CHAOS、iLISI,实现取自 `reproduce/code/compute_metrics/metrics.py`(只读)。

## 判定标准(事先给定)

按数据集,把每个 λ 的 5 seed 均值与 dot 逐 seed 配对:

- 自检:最大 λ(100000)与 dot 的 iLISI 配对差绝对值 ≤ 0.02 且 ARI 配对差绝对值 ≤ 0.02(Simulations)/ 0.03(Lymph、Tonsil,因 dot 本身 seed 间 std 更大)。不满足则先排查实现,不对其他 λ 下结论。
- 「H_3 成立」:Lymph 与 Tonsil 上,iLISI 均值随 λ 从 10 到 100000 单调不降(相邻 λ 允许 −0.01 的波动),且最大 λ 的 iLISI 与 dot 差 ≥ −0.02。
- 「存在有利 λ」:某 λ 下 Lymph 与 Tonsil 的 iLISI 相对 dot 差 ≥ −0.02,同时该 λ 的 Simulations ARI 相对 dot 配对差 ≥ +0.02 且 5/5 个 seed 为正。
- 「无关」:iLISI 在 λ 全范围的变化 < 0.03。
- 日志中出现 `mclust failed` 或 `Traceback` 的 run 不计入并重跑(只修 bug)。
