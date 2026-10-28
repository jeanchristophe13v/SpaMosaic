#!/usr/bin/env python
"""
SpaMosaic 源码运行示例:从仓库根目录直接运行,不需要 pip install。
用法: python scripts/run_demo.py [--device cpu|cuda:0]
"""
import os
import sys
from os.path import join, dirname, abspath

# ---- 源码运行的关键:把仓库根目录加入 sys.path,使 import spamosaic 直接用源代码 ----
REPO_ROOT = dirname(dirname(abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# CUDA >= 10.2 时需要(必须在 import torch 之前设置)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
import scanpy as sc
import spamosaic.utils as utls
from spamosaic.framework import SpaMosaic
from spamosaic.preprocessing import RNA_preprocess, ADT_preprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu', help="cpu 或 cuda:0")
    args = parser.parse_args()

    seed = 1234
    utls.set_seeds(seed)

    # ========== 1. 加载输入数据 ==========
    # 每个模态 x 每个切片一个 AnnData。AnnData 需要包含:
    #   .X             原始 counts (细胞 x 基因/蛋白)
    #   .obsm['spatial']  空间坐标
    #   .obs['src']    批次(切片)标签
    data_dir = join(REPO_ROOT, 'demo/data/Human_lymph_node')
    ad1_rna = sc.read_h5ad(join(data_dir, 'slice1/s1_adata_rna.h5ad'))
    ad1_adt = sc.read_h5ad(join(data_dir, 'slice1/s1_adata_adt.h5ad'))
    ad2_rna = sc.read_h5ad(join(data_dir, 'slice2/s2_adata_rna.h5ad'))
    ad3_adt = sc.read_h5ad(join(data_dir, 'slice3/s3_adata_adt.h5ad'))

    # 马赛克结构:缺失的模态用 None 占位
    # 切片1 = RNA+ADT, 切片2 = 仅RNA, 切片3 = 仅ADT
    input_dict = {
        'rna': [ad1_rna, ad2_rna, None],
        'adt': [ad1_adt, None,    ad3_adt],
    }
    input_key = 'dimred_bc'

    # ========== 2. 预处理:各模态归一化+降维+Harmony批次校正 ==========
    # 结果写入每个 adata.obsm['dimred_bc']
    RNA_preprocess(input_dict['rna'], batch_corr=True, favor='scanpy',
                   n_hvg=5000, batch_key='src', key=input_key)
    ADT_preprocess(input_dict['adt'], batch_corr=True,
                   batch_key='src', key=input_key)

    # ========== 3. 初始化模型(内部完成空间图+MNN图构建)+ 训练 ==========
    model = SpaMosaic(
        modBatch_dict=input_dict, input_key=input_key,
        batch_key='src',
        intra_knns=10, inter_knn_base=10,
        w_g=0.8,
        seed=seed,
        device=args.device,
    )
    model.train(net='wlgcn', lr=0.01, T=0.01, n_epochs=100)

    # ========== 4. 推理嵌入 ==========
    # 返回每个切片的 AnnData,嵌入存于 obsm['merged_emb']
    ad_embs = model.infer_emb(input_dict, emb_key='emb', final_latent_key='merged_emb')

    # 下游分析:合并、UMAP、聚类、空间可视化
    ad_mosaic = sc.concat(ad_embs)
    utls.get_umap(ad_mosaic, use_reps=['merged_emb'])
    utls.plot_basis(ad_mosaic, basis='merged_emb_umap', color=['src'],
                    show=True, save=join(REPO_ROOT, 'examples', 'human_lymph_umap.png'))

    utls.clustering(ad_mosaic, n_cluster=10, used_obsm='merged_emb',
                    algo='kmeans', key='kmeans')  # mclust 需要 R,失败会自动退回 kmeans
    utls.split_adata_ob(ad_embs, ad_mosaic, 'obs', 'kmeans')
    for ad in ad_embs:
        utls.plot_basis(ad, 'spatial', 'kmeans', s=70,
                        show=True, save=join(REPO_ROOT, 'examples', 'human_lymph_spatial.png'))

    # ========== 5. (可选)缺失模态填补 ==========
    for ads in input_dict.values():
        for ad in ads:
            if ad is not None:
                ad.layers['counts'] = ad.X.copy()  # 指定待填补的 layer
    imputed_featureDict = model.impute(input_dict, emb_key='emb',
                                       layer_key='counts', imp_knn=10)
    print('填补结果 keys:', list(imputed_featureDict.keys()))
    for k, v in imputed_featureDict.items():
        print(k, v.shape)


if __name__ == '__main__':
    main()
