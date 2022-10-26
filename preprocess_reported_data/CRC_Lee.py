
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt


sc_adata = sc.read_h5ad("/mnt/Bambi/Projects/yao/Lee_2021/sc_crc.h5ad")


sc_adata.var_names_make_unique()
sc_adata.layers["count"] = sc_adata.X
sc.pp.filter_cells(sc_adata, min_genes=200)
sc.pp.filter_genes(sc_adata, min_cells=3)
sc_adata.var['mt'] = sc_adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(sc_adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(sc_adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)


sc_adata = sc_adata[sc_adata.obs.n_genes_by_counts < 2500, :]
sc_adata = sc_adata[sc_adata.obs.pct_counts_mt < 5, :]
sc.pp.normalize_total(sc_adata, target_sum=1e4)
sc.pp.log1p(sc_adata)
#sc.pp.highly_variable_genes(sc_adata, n_top_genes=2000)
sc.pp.regress_out(sc_adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(sc_adata, max_value=10)

sc.tl.pca(sc_adata, svd_solver='arpack')
sc.tl.tsne(sc_adata, n_pcs = 30)
sc.pp.neighbors(sc_adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(sc_adata)
sc.tl.leiden(sc_adata)
sc.tl.louvain(sc_adata) 

sc_adata.write("/mnt/Bambi/scCRC_Lee/scCRC.h5ad")
