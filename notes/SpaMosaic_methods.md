**SpaMosaic 的核心是：每种模态各建一张跨切片图，用空间邻居与同模态跨切片邻居传播信息，再通过同一 spot 的多模态配对，把各模态编码器训练到共同空间。** 单模态切片通过图连接和特征重构参与学习；最终得到每个 spot 的统一表示，并可据此补全缺失模态。

下面以论文 Methods（PDF 第 13–14 页）为主，补充已核对的官方代码细节。

**1．输入是什么，为什么能处理 mosaic**

例如三张切片分别测到：

| 切片 | RNA | ATAC | Protein |
|---|---|---|---|
| A | ✓ | ✓ | — |
| B | ✓ | — | ✓ |
| C | — | — | ✓ |

A 提供 RNA–ATAC 配对，B 提供 RNA–Protein 配对，RNA 因而连接三种模态；C 再通过 Protein 接入。**多模态切片内部需要知道哪些观测属于同一个 spot；不同切片之间不要求 spot 一一对应。** 模态共测关系需要连通，官方代码会检查这一点。

每种模态 \(m\) 的输入是：

- \(X_m\in\mathbb R^{N_m\times d_m}\)：所有测到该模态的 spot 的低维特征。
- 每个 spot 的空间坐标、切片编号，以及跨模态配对索引。

原始计数先做模态专属预处理：RNA 用 log-normalization，表观组用 TF-IDF，蛋白用 CLR；RNA、表观组分别选取 5,000、100,000 个高变特征，再降维。同模态不同切片的低维表示先经过 Harmony 或 Seurat 批次校正，得到 \(X_m\)。**批次校正是模型前处理的一部分。**

**2．图如何构建：跨切片与跨模态是两种不同联系**

每种模态独立构图，只包含实际测到该模态的 spot：

\[
A_m=A_{\mathrm{spatial},m}+w_gA_{\mathrm{MNN},m}.
\]

- **切片内空间边**：按坐标找 \(k\) 近邻，构建无向图，边权为 1。
- **切片间 MNN 边**：在批次校正后的 \(X_m\) 中寻找互为近邻的 spot，边权为 \(w_g\)。

默认 \(k=10\)，MNN 基础邻居数为 10，\(w_g=0.8\)。图固定后进入训练。

因此，**MNN 边连接的是不同切片的同模态 spot；跨模态联系由后面的对比损失建立。** MNN 配对本身不是跨模态正样本。

论文另提供两项可选处理：切片大小失衡时采用非对称 MNN 邻居数；根据配对 spot 的坐标及空间偏移，用 Isolation Forest 删除异常程度最高的 50% 配对。

**3．编码器做什么：固定图扩散，再学习投影**

每种模态一个编码器，同模态的所有切片共享参数。先给图加自环并对称归一化：

\[
\hat A_m=D_m^{-1/2}(A_m+I)D_m^{-1/2}.
\]

默认传播两步，把原始、一跳、两跳特征沿特征维拼接：

\[
H_m=[X_m\;\|\;\hat A_mX_m\;\|\;\hat A_m^2X_m]
\in\mathbb R^{N_m\times3d_m}.
\]

这里的 WLGCN **传播过程没有可学习权重或逐层激活**；学习发生在后续 MLP：

\[
H_m
\xrightarrow{\mathrm{Linear}(3d_m,128)}
\mathrm{LeakyReLU}(0.2)
\rightarrow\mathrm{BatchNorm}
\rightarrow\mathrm{Dropout}(0.2)
\xrightarrow{\mathrm{Linear}(128,32)}
U_m.
\]

实现中有两个输出分支：

\[
Z_m=\operatorname{L2Normalize}(U_m),\qquad
\hat X_m=\operatorname{Linear}_m(U_m).
\]

\(Z_m\) 用于对比学习和下游分析；线性解码器从**归一化之前**的 \(U_m\) 重构模型输入 \(X_m\)。它重构的是预处理后的低维特征，不是原始基因计数。[编码器实现](https://github.com/JinmiaoChenLab/SpaMosaic/blob/main/spamosaic/architectures/wlgcn.py)

**4．如何训练：配对 spot 对齐，单模态 spot 保留信息**

对于同一切片中同时测到模态 \(a,b\) 的 \(n\) 个 spot：

- 正样本：同一 spot 的 \(z_i^a,z_i^b\)。
- 负样本：其他 spot 的表示，**包括另一模态和同一模态**。
- 相似度：余弦相似度除以温度 \(\tau\)。

以模态 \(a\) 为 anchor：

\[
\ell_i^{a\to b}
=-\log
\frac{\exp(s(z_i^a,z_i^b))}
{\sum_{j=1}^{n}\exp(s(z_i^a,z_j^b))
+\sum_{j\ne i}\exp(s(z_i^a,z_j^a))},
\quad
s(u,v)=\frac{\cos(u,v)}{\tau}.
\]

再对两个方向、所有 anchor 求平均。这样既拉近同一 spot 的不同模态，又避免所有 spot 坍缩成相同表示。三种以上模态共测时，官方默认实现把同一 spot 的其他模态都作为正样本，对各正样本的 log-probability 求平均。[对比损失实现](https://github.com/JinmiaoChenLab/SpaMosaic/blob/main/spamosaic/loss.py)

单模态 spot 没有跨模态正样本，使用特征重构：

\[
\mathcal L_{\mathrm{rec}}=\operatorname{MSE}(\hat X_m,X_m).
\]

**复现时要区分论文简写与代码：**论文写成“对比损失之和＋重构损失之和”；当前 `main` 实际在多模态切片上计算对比损失，在单模态切片上计算重构损失，分别平均后相加。默认 `w_rec_g=0` 表示使用特征重构，而非关闭重构；该参数控制的是图重构与特征重构的比例。[训练循环](https://github.com/JinmiaoChenLab/SpaMosaic/blob/main/spamosaic/train_utils.py)

训练默认值：Adam，学习率 0.01，weight decay \(5\times10^{-4}\)，100 epochs，\(\tau=0.01\)。每轮先计算各模态的全图表示，再按切片计算损失；大切片的 mini-batch 用于对比损失计算，并非图邻居采样。

**5．训练后如何得到结果**

**统一表示与聚类。** 对每个 spot，只融合它实际测到的模态：

\[
z_i=\sum_{m\in T_b}w_mz_i^m,\qquad \sum_{m\in T_b}w_m=1.
\]

论文允许加权平均，当前代码默认等权平均；单模态 spot 直接使用自己的表示。聚类在此后单独进行，论文采用 MClust，`modelNames="EEE"`，聚类数另外指定。

**缺失模态补全。** 假设 spot \(i\) 缺少 Protein：

1. 用它的 RNA 表示，在所有实测 Protein 的表示中寻找 \(k_{\mathrm{imp}}\) 个近邻。
2. 平均这些邻居的实测 Protein 特征，作为预测。
3. 若该 spot 还测到 ATAC，再独立预测一次，最后平均两份预测。

\[
\hat c_i^{\,p}
=\frac{1}{|T_b|}
\sum_{m\in T_b}
\frac{1}{k_{\mathrm{imp}}}
\sum_{j\in\operatorname{kNN}(z_i^m,Z_p)}
c_j^p.
\]

**补全依靠对齐空间中的近邻特征转移，训练中的线性解码器不承担跨模态生成。** 当前代码默认 `imp_knn=10`。[融合与补全实现](https://github.com/JinmiaoChenLab/SpaMosaic/blob/main/spamosaic/framework.py)

**6．修改源码时对应看哪里**

| 修改目标 | 代码入口 |
|---|---|
| 特征预处理、批次校正 | `preprocessing.py` |
| 空间边、MNN 边和边权 | `build_graph.py`、`MNN.py` |
| 图传播层数、MLP、潜在维度、解码器 | `architectures/wlgcn.py`、`configs/` |
| 正负样本定义、多模态对比目标 | `loss.py` |
| 哪些 spot 计算什么损失、损失权重、更新方式 | `train_utils.py` |
| 输入组织、训练调度、表示融合与补全 | `framework.py` |