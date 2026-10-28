# AGENTS.md — SpaMosaic 项目协作规范

SpaMosaic:基于对比学习 + GNN 的空间多组学 mosaic 整合工具(模态无关、批次校正的 latent space,下游做空间域识别与缺失组学插补)。

- 运行环境: conda env `spamosaic-env`。rpy2/mclust 依赖 `R_HOME`,跑实验前需确保该环境变量已设置(exp001 曾因此崩溃,见 exp002)。
- 数据集放在 `data/`,常用: `Simulations`(有 gt)、`MB-H3K4me3`(无 gt,只算 PAS/CHAOS/iLISI)。
- 常用指标: ARI、F1、PAS、CHAOS、iLISI。

## 实验目录规范(重要)

所有实验统一放在 `experiments/`,**每个实验一个自包含文件夹**,内部固定三个子目录:

```
experiments/
├── index.csv                      # 全局实验索引,只放最核心的结论
├── exp001_topola_first_run/       # 命名: exp + 3位递增编号 + 小写蛇形短标签
│   ├── scripts/                   # 本次实验实际运行的脚本快照(self-contained)
│   ├── results/                   # 详细结果: config.json, metrics.csv, summary.csv, 各 variant/seed 产物
│   └── log/                       # 本次实验的完整运行日志(如 run.log)
└── exp002_topola_mb_rfixed/
    └── ...
```

### 编号规则

- 文件夹名: `expNNN_short_slug`,NNN 为 3 位递增编号(新实验取 `index.csv` 中最大编号 + 1),slug 用小写蛇形命名、不超过 4 个词。
- 同一实验的重跑/修复**不新开编号**,沿用原编号(如 exp002 是 exp001 的 MB 部分修复重跑,因改动语义上是同一次消融;若是一次新的改动探索,则新开编号)。编号一旦分配永不复用。

### index.csv 规范

列: `id,date,change,key_result`

- `change`: 本次改动细节(改了什么、跑了哪些配置)。
- `key_result`: **一行内最核心最简洁的结论**(如最优变体 + 关键指标值),只写看全局时需要的信息;单元格内用 `;` 分隔、不要用逗号(避免 CSV 转义)。
- 详细数据一律看对应 `expNNN_*/results/`,不往 index.csv 里堆。

### 各子目录职责

- `scripts/`: 当次实验**实际运行版本**的脚本快照(入口脚本 + 被修改/被依赖的自有模块,如 `run_loss_experiments.py`、`topola_D_and_S.py`)。改了代码就 copy 进来,保证不看 git 历史也能复现本次实验。
- `results/`: 实验输出(`config.json`、`metrics.csv`、`summary.csv`、embedding 等产物)。跑实验时把脚本的 `output_dir` 指到这里。
- `log/`: 必要的运行日志 `... 2>&1 | tee experiments/expNNN_*/log/run.log`。

### 开展新实验的工作流(Agent 必须遵守)

1. 读 `experiments/index.csv`,确定下一个编号 NNN,起好 slug。
2. `mkdir -p experiments/expNNN_slug/{scripts,results,log}`。
3. 把(修改后的)脚本快照 copy 到 `scripts/`。
4. 运行实验:`output_dir` 指向 `results/`,日志 tee 到 `log/run.log`。
5. 实验结束后,在 `index.csv` **末尾追加一行**,填好 date、change、key_result。

### Git 策略

- 入库: `experiments/index.csv`、各实验的 `scripts/`、`log/`、`results/` 下的 csv/json。
- 忽略: `experiments/**/*.h5ad` 等大体积可再生产物(见 `.gitignore`)。

### 历史说明

- exp001 / exp002 由旧的 `loss_experiments/topola_first_run`、`loss_experiments/topola_mb_rfixed` 迁移而来(2026-10-27),原始内容未改动,仅按上述结构重新归位。
