## https://figshare.com/articles/dataset/MCA_DGE_Data/5435866
## https://github.com/czi-hca-comp-tools/easy-data/blob/master/datasets/tabula_muris.md#count-files-for-r



import pandas as pd
from anndata import read_h5ad
import scanpy as sc

metadata = pd.read_csv('22008948', index_col=1)
metadata2 = pd.read_csv('11083451', index_col=1)
adata = read_h5ad('22008951')
adata.obs = pd.DataFrame(adata.obs).merge(metadata2, how="left", left_index=True, right_index=True)

# kidney
kidney = adata[adata.obs.Tissue=="Kidney",:].copy()
kidney.layers["count"] = adata[adata.obs.Tissue=="Kidney",:].X.copy()

def transform_exp(adata):
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=100)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.scale(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=4000 )
    sc.pp.pca(adata)
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.tsne(adata)
    return adata

kidney = transform_exp(kidney)
sc.pl.umap(kidney, color="Annotation", save="kidney")

kidney.write_h5ad("kidney.h5ad")