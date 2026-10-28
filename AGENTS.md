# AGENTS.md

SpaMosaic:空间多组学 mosaic 整合工具(对比学习 + GNN)。conda env `spamosaic-env`;rpy2/mclust 依赖 `R_HOME` 环境变量。

## 仓库地图

| 路径 | 是什么 | 规则 |
|---|---|---|
| `spamosaic/` | 算法实现:framework(主类)、loss、architectures/(网络)、topola、metrics(评估指标) | **主要开发区** |
| `scripts/` | 入口脚本:run_demo.py、run_loss_experiments.py | 实验 driver 模板 |
| `experiments/` | 实验记录,编号索引(见下) | 按工作流写入 |
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

## 实验工作流(必须遵守)

1. 读 `experiments/index.csv` 取下一个编号 NNN,`mkdir -p experiments/expNNN_slug/{scripts,results,log}`(slug 小写蛇形 ≤4 词;编号永不复用)。
2. 从 `scripts/` 复制 driver 到 `expNNN/scripts/` 并在其中修改,保证快照自包含。
3. 运行:输出写 `results/`,日志 `2>&1 | tee .../log/run.log`。
4. 结束后在 `index.csv` 末尾追加一行(列 `id,date,change,key_result`):change 写改动与配置;key_result 只写一行内最核心结论,详细数据看 `results/`。单元格内用 `;` 分隔、不用 `,`。
5. 同一实验的修复重跑沿用原编号;新的改动探索开新编号。

Git:`index.csv`、`scripts/`、`log/`、`results/` 下 csv/json 入库;`*.h5ad`、`docs/build/` 已 ignore。
