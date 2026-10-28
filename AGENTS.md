# AGENTS.md

SpaMosaic:空间多组学 mosaic 整合工具(对比学习 + GNN)。conda env `spamosaic-env`;rpy2/mclust 依赖 `R_HOME` 环境变量。

## 仓库地图

| 路径 | 是什么 | 规则 |
|---|---|---|
| `spamosaic/` | 算法实现:framework(主类)、loss、architectures/(网络)、topola、metrics(评估指标) | **主要开发区** |
| `scripts/` | 入口脚本:run_demo.py、run_loss_experiments.py | 实验 driver 模板 |
| `experiments/` | 实验记录:`index.csv`(编号索引)、各 `expNNN/`(含 `README.md` 设计、`findings.md` 结果分析) | 按工作流写入 |
| `tests/` | test_smoke.py | 改 `spamosaic/` 后必跑 |
| `notes/` | 个人笔记与方案文档 | — |
| `examples/` | 自己的 demo 运行记录(notebook + 输出) | — |
| `reproduce/` | 论文复现分支快照(baseline 方法、指标代码原版) | **只读** |
| `demo/`、`docs/` | 上游原仓库的教程与用户文档 | **只读** |
| `data/` | 本地大数据集 | 已 gitignore |

## 开发约定

- 加网络:新文件放 `spamosaic/architectures/`,并在其 `__init__.py` 注册;改 loss 在 `spamosaic/loss.py`、`train_utils.py`。
- **铁律**:`spamosaic/` 包内禁止 import 包外散落文件或绝对路径。指标一律 `from spamosaic import metrics`;TopoLa 一律 `from spamosaic.topola import ...`。
- 改完跑 `python tests/test_smoke.py`(metrics 因 scib 依赖较重,不在其中)。
- `reproduce/`、`demo/`、`docs/` 是上游/冻结内容,不修改;要用其中的代码就复制到开发区再改。

## 机器与资源(node1,共享服务器)

- 配置:512 CPU 核、~3TB 内存、8× RTX 4090(各 48GB)。多人共用,**卡不是每天都有**。
- 跑实验前先 `nvidia-smi` 查空闲卡;**最多同时占 2 张卡**,给他人留余量。
- 单个任务通常远小于一张卡的能力(显存 <2GB),资源余量充裕,设计并发时放心利用。

## 实验工作流(必须遵守)

设计原则:
- **主问题由用户定**。agent 可以提议新方向,但必须经用户同意后才能开新实验。
- 一个实验只改一个因素;必须带基线(写明沿用哪个实验的配置);比较变体至少 3 个种子。
- 修复重跑只修 bug,不改其他参数。

步骤:
0. 设计前读相关实验的 `expNNN/findings.md`;跨实验分析靠读多个实验的 findings,不另设汇总文件。
1. 读 `experiments/index.csv` 取下一个编号 NNN,`mkdir -p experiments/expNNN_slug/{scripts,results,log}`(slug 小写蛇形 ≤4 词;编号永不复用)。
2. 运行前写 `expNNN/README.md`(exp005 起必需):问题、假设、唯一改动的因素、基线、判定标准。
3. 从 `scripts/` 复制 driver 到 `expNNN/scripts/` 并在其中修改,保证快照自包含。
4. 运行:输出写 `results/`,日志 `2>&1 | tee .../log/run.log`。
5. 结束后写 `expNNN/findings.md`(exp005 起必需):完整总结本实验结果并简要分析;数字与 `results/` 一致,注明种子数与配置条件;只分析,不写下一步建议。
6. 在 `index.csv` 末尾追加一行(列 `id,date,change,key_result`):change 写改动与配置;key_result 只写一行内最核心结论,详细数据看 `results/`。单元格内用 `;` 分隔、不用 `,`。
7. 同一实验的修复重跑沿用原编号;新的改动探索开新编号。

Git:`index.csv`、`scripts/`、`log/`、各 exp 的 `README.md`/`findings.md`、`results/` 顶层汇总 csv/json(如 all_metrics、summary、config、manifest)入库;逐 run 产物(`seed*/`、`runs/`、`cache/`)与 `*.npz`、`*.pt`、`*.h5ad`、`docs/build/` 已 ignore(见 `.gitignore`)。
