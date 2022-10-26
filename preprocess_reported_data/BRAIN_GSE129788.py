
import pandas as pd
import glob
import scanpy as sc
import scvelo as scv

def transform_exp(adata):
    # adata.layers["count"] =  pd.DataFrame.sparse.from_spmatrix(adata.X)
    adata.layers["count"] =  adata.X
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, )
    adata.raw = adata
    sc.pp.scale(adata)
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.tsne(adata)
    sc.tl.louvain(adata)
    sc.tl.leiden(adata)
    return adata


exprs = glob.glob("/mnt/Daisy/sc_BRAIN_GSE129788/*_10X.txt.gz")

df = pd.DataFrame()
i=0
for i in range(len(exprs)):
    expr = pd.read_table(exprs[i], header=0)
    df = df.join(expr, how="outer")


df = df.T
df = df.sort_index()

meta = pd.read_table("/mnt/Daisy/sc_BRAIN_GSE129788/GSE129788_Supplementary_meta_data_Cell_Types_Etc.txt.gz")
meta = meta.iloc[1:,].reset_index(drop=True)
meta.index = meta.NAME.str.split("portal_data_", expand=True).iloc[:,1]
meta = meta.sort_index().reset_index(drop=True)

adata = sc.AnnData(df)
adata.obs = meta
adata.obs_names = df.index


adata = transform_exp(adata)

sc.pl.umap(adata, color="cluster", save= "total.pdf")

adata.write_h5ad("/mnt/Daisy/sc_BRAIN_GSE129788/adata.h5ad")

