# exp009 label_fix_rebaseline

## 问题

exp005–008 在 Lymph/Tonsil 上计算 ARI(all)、F1 与跨切片标签一致率时，没有统一各切片的标签拼写。
例如 `medulla Cords`/`medulla cords`、`medulla vessel`/`medulla vessels`、
`germinal Center`/`germinal center`、`connective & epithelial tissue`/`connective and epithelial tissue`。
上游 `reproduce/code/compute_metrics/run_ARI.ipynb` 用映射表 `labd` 统一了这些拼写，我们的 driver 漏掉了。
结果是 Lymph 从 10 类变成 12 类,Tonsil 从 5 类变成 7 类(含 nan)。

本实验只修评估，回答三个问题:
1. 统一标签后,Vanilla(dot)与 topola_cal 在 Sim/Lymph/Tonsil 上的正确基线是多少;
2. exp005 按其事先给定的判定标准，结论是否改变;
3. 统一标签后，真实数据的短板是否集中在只测 ADT、与 S1 不相邻的 S3 切片(见下方 H_S3)。

## 假设

- H_fix:统一标签只改变 Lymph/Tonsil 的 ARI(all) 与 F1;逐切片 ARI、iLISI、PAS、CHAOS 与 exp005 逐 seed 一致。
  Simulations 的全部指标与 exp005 一致。
- H_S3(来自 2026-10-29 的临时诊断，在 exp008 的 dot 嵌入上算得，未入库):
  - S3→S1 的 top-10 近邻标签一致率低于 S2→S1;
  - Tonsil 上,S3 嵌入的切片内留一 kNN macro-F1 低于 S3 ADT 输入特征的留一 F1(诊断值 0.573 对 0.640)。

## 唯一改动的因素

评估时的标签统一:按上游 `labd` 映射表统一 Lymph/Tonsil 各切片的 `lab`;Simulations 不变。
训练、预处理、构图、超参、seed、聚类全部沿用 exp005,不做任何改动。

## 基线

exp005(`experiments/exp005_topola_cal_multiseed/scripts/run_exp005.py`):两个 arm 是 `dot`(Vanilla)与
`topola_cal`(`D_λ/mean(diag D_λ)`,λ=10),都用 `StableCL_loss`;3 个数据集 × 5 个 seed(1234/2024–2027);100 轮。

## 设计

- driver:从 exp005 复制 `run_exp005.py` 为 `scripts/run_exp009.py`,只做以下改动:
  1. 在 `compute_scores` 内对 gt 应用标签映射(Lymph/Tonsil 各一张表，逐字抄自上游 notebook);
     `nan` 照旧剔除。映射后若出现未覆盖的标签，直接报错。
  2. 同时输出未统一标签的 ARI(all)/F1(scope 加后缀 `_raw`),用来与 exp005 对账。
  3. 保存每个 run 的模态嵌入、合并嵌入、聚类标签与 MNN 对到 `results/runs/<dataset>/<arm>/seed*/`(已 gitignore)。
- 诊断脚本 `scripts/diag_exp009.py`(只读 runs/,不训练),全部用统一后的标签:
  - D1:合并嵌入(L2 归一后)上的 top-10 近邻标签一致率：切片内(S1/S2/S3)、S1-RNA→S1-ADT(模态嵌入)、
    S2→S1、S3→S1、S2→S3、S3→S2;
  - D2:切片内留一 kNN(k=10)macro-F1:嵌入对比输入特征(`dimred_bc`/`X_pca_be`);
  - D3:SpaMosaic 实际 MNN 对(S1↔S2 RNA、S1↔S3 ADT)的标签一致率与随机基线，用于订正 exp008 第 2 部分的绝对值。
- 资源：先 `nvidia-smi` 选空闲卡，最多 2 张卡;按数据集或 arm 拆成两个进程并行。

## 判定标准(事先给定)

- 自检(H_fix):每个 run 的逐切片 ARI、iLISI、PAS、CHAOS 与 exp005 `all_metrics.csv` 的同一 run 相差 ≤1e-3;
  `_raw` 版的 ARI(all)/F1 与 exp005 相差 ≤1e-3。不满足则先排查(确定性、环境),不得进入分析。
- exp005 结论重判：沿用 exp005 README 的三条标准，用统一标签后的数字重新判定，逐条给出成立/不成立。
- H_S3 成立的条件:
  - Lymph 与 Tonsil 上,S3→S1 一致率比 S2→S1 低 ≥0.02,且 5/5 个 seed 同向;
  - Tonsil 上 D2 的 S3 嵌入 F1 低于 S3 输入 F1,且 5/5 个 seed 同向。
  以 dot arm 为准;topola_cal 只报告，不参与判定。
- 日志中无 `mclust failed`、`Traceback`、`Nonfinite`。
