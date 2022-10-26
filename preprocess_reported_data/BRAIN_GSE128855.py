
import pandas as pd
import glob
import scanpy as sc
import scvelo as scv


def transform_exp(adata):
    # adata.layers["count"] =  pd.DataFrame.sparse.from_spmatrix(adata.X)
    adata.layers["count"] =  adata.X.copy()
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

adata = sc.read_10x_mtx("/mnt/Daisy/sc_BRAIN_GSE128855/filtered_gene_bc_matrices_mex_WT_fullAggr/filtered_gene_bc_matrices_mex/mm10")
meta = pd.read_csv("/mnt/Daisy/sc_BRAIN_GSE128855/annot_fullAggr.csv")
meta.index = meta.cell 
adata.obs = pd.DataFrame(adata.obs).join(meta, how="left")

# adata.layers["count"] =  adata.X.copy()
adata = transform_exp(adata)

sc.pl.umap(adata, color="cluster")

adata.write_h5ad("/mnt/Daisy/sc_BRAIN_GSE128855/adata.h5ad")




adata = sc.read_10x_mtx("/mnt/Daisy/sc_BRAIN_GSE128855/filtered_gene_bc_matrices_WT_wholeBrain/filtered_gene_bc_matrices/mm10")
meta = pd.read_csv("/mnt/Daisy/sc_BRAIN_GSE128855/annot_K10.csv")
meta.index = meta.cell + "-1"
adata.obs = pd.DataFrame(adata.obs).join(meta, how="left")

adata = transform_exp(adata)

sc.pl.umap(adata, color="cluster")

adata.write_h5ad("/mnt/Daisy/sc_BRAIN_GSE128855/adata_wholebrain.h5ad")