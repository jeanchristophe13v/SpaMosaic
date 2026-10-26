# Loss 替换方案

目标：修改 SpaMosaic 的对比 loss，重新跑 `run_demo.py`，看 ARI 是否提升。

最小方案：只改 `run_demo.py`。在脚本里用自定义 loss 类替换 `spamosaic.framework.CL_loss`，再在聚类后面补上 ARI 计算。`spamosaic/` 下的库代码不需要改。

## 现有代码要点

- **对比 loss 的位置**：`spamosaic/loss.py` 中的 `CL_loss.forward(simi)`。默认的 `loss_type='adapted'` 走的就是它。
- **它在哪里被创建**：`spamosaic/framework.py:421-424`，调用方式是 `CL_loss(batch_size, rep=模态数, bias=bias)`。`framework.py` 用 `from spamosaic.loss import CL_loss` 导入，所以替换 `spamosaic.framework.CL_loss` 这个名字就能生效。
- **`simi` 是什么**：见 `spamosaic/train_utils.py:193-194`。把各模态的嵌入按行拼起来得到 `feat`，然后 `simi = feat @ feat.T / T`。嵌入在 `architectures/wlgcn.py:183` 已经做过 L2 归一化，所以 `simi` 是余弦相似度除以温度 T。
- **这个 demo 里的具体形状**：只有 slice1 同时有 RNA 和 ADT，所以只有它参与对比学习。`rep=2`，`batch_size=3484`（低于 8000 的阈值，整片一次算完，不分 mini-batch），`simi` 是 6968×6968。第 i 行的正样本是另一个模态里的同一个 spot，已经由 `pos_mask` 编码好。
- **ARI 目前没有计算**：`run_demo.py` 只做了 kmeans 聚类。三个切片的 `obs` 里都有真值列 `final_annot`（10 类），`infer_emb` 会把 `obs` 原样复制到返回的 AnnData 里，可以直接用。
- **不要用 `loss_type='ce'` 当对照**：这条分支里 `crit1` 是单个 `nn.CrossEntropyLoss`，但 `train_utils.py:198` 用 `crit1[bi]` 去取它，看代码应该会直接报错。

## 改动内容（都在 `run_demo.py`）

### 1. 在文件顶部加上 loss 变体

```python
import torch
import spamosaic.framework as fw
from spamosaic.loss import CL_loss
from sklearn.metrics import adjusted_rand_score

class MyCL(CL_loss):
    """示例变体：分母只算跨模态的负样本（类似 CLIP），同模态内的 spot 不再作为负样本。"""
    def __init__(self, batch_size, rep=3, bias=0):
        super().__init__(batch_size, rep, bias)
        mod_ids = torch.arange(rep).repeat_interleave(batch_size)
        self.register_buffer('cross_mask', (mod_ids[:, None] != mod_ids[None, :]).float())

    def forward(self, simi):
        simi = simi - simi.max(dim=1, keepdim=True)[0].detach()
        pos = (simi * self.pos_mask).sum(1) / self.pos_mask.sum(1)
        neg = (torch.exp(simi) * self.cross_mask).sum(1)
        return -(pos - torch.log(neg + self.bias)).mean()
```

`__init__` 和 `forward` 的签名要和原来的 `CL_loss` 保持一致。`forward` 的内容换成你自己想试的 loss 即可。

### 2. 加 `--loss` 开关，并在训练前替换

```python
parser.add_argument('--loss', default='orig', choices=['orig', 'new'])
...
if args.loss == 'new':
    fw.CL_loss = MyCL
model.train(net='wlgcn', lr=0.01, T=0.01, n_epochs=100)
```

### 3. 在 `utls.split_adata_ob(...)` 之后打印 ARI

```python
for ad in ad_embs:
    print(ad.obs['src'].iloc[0], 'ARI =', adjusted_rand_score(ad.obs['final_annot'], ad.obs['kmeans']))
print('overall ARI =', adjusted_rand_score(ad_mosaic.obs['final_annot'], ad_mosaic.obs['kmeans']))
```

如果只比较 ARI，可以把第 5 步缺失模态填补注释掉，能省不少时间。

## 运行

```bash
conda activate spamosaic-env
python run_demo.py --device cuda:0 --loss orig   # 基线
python run_demo.py --device cuda:0 --loss new    # 新 loss
```

## 比较时的注意事项

- 两次运行要用同一个 seed（脚本里已固定为 1234）和同一个设备，结果才可比。
- kmeans 和训练本身都有随机性，只差 0.01 到 0.02 的 ARI 不足以说明问题。最好把 seed 也做成参数，每种 loss 跑 3 到 5 个 seed，比较均值。
- 重点看 slice2（只有 RNA）和 slice3（只有 ADT）的 ARI。它们不参与对比训练，最能反映跨模态对齐有没有变好。

## 后续

如果这个变体确实有提升，再考虑把它正式写进 `spamosaic/loss.py`，并在 `framework.train` 里加一个 `loss_type` 选项。

注：以上代码尚未实际运行验证。
