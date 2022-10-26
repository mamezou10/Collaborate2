
import pandas as pd
import glob
import scanpy as sc
import scvelo as scv
import os
import subprocess


def transform_exp(adata):
    # adata.layers["count"] =  pd.DataFrame.sparse.from_spmatrix(adata.X)
    adata.layers["count"] =  adata.X
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, )
    # adata.raw = adata
    sc.pp.scale(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.tsne(adata)
    sc.tl.louvain(adata)
    sc.tl.leiden(adata)
    return adata
'''
exprs = glob.glob("/mnt/Daisy/sc_BRAIN_GSE153424/GSE153424_RAW/*.tar.gz")

## 解凍
for i in range(len(exprs)):
    dir_name = exprs[i].split(".tar.gz")[0]
    subprocess.run(["mkdir", dir_name])
    subprocess.run(["tar", "xvzf", exprs[i], "-C", dir_name])
'''

## データ読み込み
mtxs = glob.glob("/mnt/Daisy/sc_BRAIN_GSE153424/GSE153424_RAW/*_filtered_gene_bc_matrix")
adatas = {}
for i in range(len(mtxs)):  
    dir_name = mtxs[i]
    batch = os.path.basename(dir_name)
    print(batch)
    adata = sc.read_10x_mtx(dir_name)
    adatas[batch] = adata

# adatas_list = list(adatas.values())

import anndata
adata = anndata.concat(adatas, label="batch")

df_adata = pd.DataFrame(adata.obs)
df_adata["cell_index"] = list(df_adata.batch.str.split("_", expand=True)[1] + "_" +  df_adata.index)

meta = sc.read_h5ad('/mnt/Daisy/sc_BRAIN_GSE153424/brain.h5ad')
df_meta = pd.DataFrame(meta.obs)
df_meta["cell_index"] = list(df_meta.index + "-1")

adata.obs = df_adata.merge(df_meta, "left")

# adata = transform_exp(adata)
# sc.pl.umap(adata, color="adata = transform_exp(adata)")

adata.write_h5ad("/mnt/Daisy/sc_BRAIN_GSE153424/count_adata.h5ad")


