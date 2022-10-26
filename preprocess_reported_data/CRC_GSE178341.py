import os
wd = '/media/hirose/Bambi/Projects/morimoto2/analysis/220429'
os.makedirs(wd, exist_ok=True)
os.chdir(wd)
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

## scCRC の整理
## prepare sc_adata
sc_adata = sc.read_10x_h5("/media/hirose/Bambi/scCRC_GSE178341/GSE178341_crc10x_full_c295v4_submit.h5")
sc_adata.var_names_make_unique()
sc_adata.layers["count"] =  pd.DataFrame.sparse.from_spmatrix(sc_adata.X)
sc.pp.filter_cells(sc_adata, min_genes=200)
sc.pp.filter_genes(sc_adata, min_cells=3)
sc_adata.var['mt'] = sc_adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(sc_adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

sc_adata = sc_adata[sc_adata.obs.n_genes_by_counts < 2500, :]
sc_adata = sc_adata[sc_adata.obs.pct_counts_mt < 6, :]
sc.pp.normalize_total(sc_adata, target_sum=1e4)
sc.pp.log1p(sc_adata)
sc.pp.highly_variable_genes(sc_adata, n_top_genes=2000)

sc.pp.regress_out(sc_adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(sc_adata, max_value=10)

meta_data = pd.read_csv("/media/hirose/Bambi/scCRC_GSE178341/GSE178341_crc10x_full_c295v4_submit_metatables.csv")
meta_data_=meta_data.set_index("cellID")
sc_adata.obs = pd.merge(pd.DataFrame(sc_adata.obs), meta_data_, how="left", left_index=True, right_index=True)

cluster_data = pd.read_csv("/media/hirose/Bambi/scCRC_GSE178341/GSE178341_crc10x_full_c295v4_submit_cluster.csv")
cluster_data_=cluster_data.set_index("sampleID")
sc_adata.obs = pd.merge(pd.DataFrame(sc_adata.obs), cluster_data_, how="left", left_index=True, right_index=True)

sc.tl.pca(sc_adata, svd_solver='arpack')
sc.tl.tsne(sc_adata, n_pcs = 30)
sc.pp.neighbors(sc_adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(sc_adata)
sc.tl.leiden(sc_adata)
sc.tl.louvain(sc_adata) 

sc_adata.write("/media/hirose/Bambi/scCRC_GSE178341/scCRC.h5ad")