# AGENTS.md

SpaMosaic:空间多组学 mosaic 整合工具(对比学习 + GNN)。conda env `spamosaic-env`;rpy2/mclust 依赖 `R_HOME` 环境变量。

## 仓库地图

| 路径 | 是什么 | 规则 |
|---|---|---|
| `spamosaic/` | 算法实现:framework(主类)、loss、architectures/(网络)、topola、metrics(评估指标) | **主要开发区** |
| `scripts/` | 入口脚本:run_demo.py、run_loss_experiments.py | 实验 driver 模板 |
| `experiments/` | 实验记录:`index.csv`(编号索引)、各 `expNNN/`(含 README、scripts、results、log、findings、summary);历史记录在 `archived/` | 按工作流写入 |
| `notes/` | 个人笔记与方案文档 | — |
| `examples/` | 自己的 demo 运行记录(notebook + 输出) | — |
| `reproduce/` | 论文复现分支快照(baseline 方法、指标代码原版) | **只读** |
| `demo/`、`docs/` | 上游原仓库的教程与用户文档 | **只读** |
| `data/` | 本地大数据集 | 已 gitignore |

## 开发约定
- `reproduce/`、`demo/`、`docs/` 是上游/冻结内容,不修改;要用其中的代码就复制到开发区再改。

## 机器与资源(node1,共享服务器)

- 配置:512 CPU 核、~3TB 内存、8× RTX 4090(各 48GB)。多人共用,**卡不是每天都有**。
- 跑实验前先 `nvidia-smi` 查空闲卡;**最多同时占 2 张卡**,给他人留余量。
- 单个任务通常远小于一张卡的能力(显存 <2GB),资源余量充裕,设计并发时放心利用。

## 实验工作流
历史实验已归档到 `experiments/archived/`，如无提起，不去查看。
3. 每个实验应遵循以下结构：
   - `README.md`：运行前写明计划。
   - `scripts/`：本实验的代码。
   - `results/`：实验结果与评估产物。
   - `log/`：运行日志。
   - `findings.md`：运行后记录完整结果、分析、失败或错误及其影响；数字必须与 results 一致。
   - `summary.md`：运行后用直觉性、易读的语言简要总结本实验；与 findings 和 results 保持一致，该文件主要是给用户总结。
4. 运行使用已确认的配置。需要改变实验设计时，先与用户讨论并确认。修复重跑沿用原编号；新的改动探索使用新编号。
5. 创建实验时在当前 `index.csv` 登记一行，运行后更新同一行。列为 `id,date,change,key_result`；change 写改动与配置，key_result 运行后写核心结论。单元格内用 `;` 分隔、不用 `,`。
6. 跨实验分析直接读取相关实验的 findings、summary 和原始结果；不自动创建额外汇总文档。

一些占地空间大的无用中间文件需要及时清理，有用但大的文件写进.gitignore。
