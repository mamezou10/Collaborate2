import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata
import os

data_dir = "/mnt/Donald/tsuji/mouse_sc/221116/"
out_dir = "/mnt/Donald/tsuji/mouse_sc/221122/"
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)

adata_mouse=sc.read_h5ad(f'{data_dir}/adata_total_cellClass1.h5')

c_d4 = sc.read_10x_h5("/mnt/Daisy/tsuji_sc_mouse/tsuji_day4/outs/filtered_feature_bc_matrix.h5")
c_d7 = sc.read_10x_h5("/mnt/Daisy/tsuji_sc_mouse/tsuji_day7/outs/filtered_feature_bc_matrix.h5")
c_d10 = sc.read_10x_h5("/mnt/Daisy/tsuji_sc_mouse/tsuji_day10/outs/filtered_feature_bc_matrix.h5")
#ko_d7 = sc.read_10x_h5("/mnt/Daisy/tsuji_sc_mouse/tsuji_KO_day7/outs/filtered_feature_bc_matrix.h5")


## preprocess
samples = [c_d4, c_d7, c_d10]#, ko_d7 ]
sampleNames = ["Day4", "Day7", "Day10"]#, "KO_Day7"]
adatas = anndata.AnnData(np.ones((1,len(c_d4.var_names)))); adatas.var_names=c_d4.var_names; adatas.var_names_make_unique()
# adatas = sc.AnnData()
for i in range(len(samples)):
    adata = samples[i].copy()
    sampleName = sampleNames[i]
    adata.var_names_make_unique(); adata
    adata.layers["count"] = adata.X.copy()
    sc.pl.highest_expr_genes(adata, n_top=20, show=False); plt.savefig(f'{out_dir}/highest_expr_genes_{sampleName}.png')
    sc.pp.filter_cells(adata, min_genes=200); adata
    sc.pp.filter_genes(adata, min_cells=3); adata
    #
    adata.var['mt'] = adata.var_names.str.startswith('mt-') 
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, show=False); plt.savefig(f'{out_dir}/violin_{sampleName}.png')
    #
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', show=False);     plt.savefig(f'{out_dir}/scatter1_{sampleName}.png');     # plt.show()
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', show=False); plt.savefig(f'{out_dir}/scatter2_{sampleName}.png'); # plt.show()
    # adata = adata[adata.obs.n_genes_by_counts > 2500, :]; adata
    adata = adata[adata.obs.n_genes_by_counts < 5000, :]; adata
    adata = adata[adata.obs.pct_counts_mt < 5, :];       adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(adata, show=False); plt.savefig(f'{out_dir}/highly_variable_genes_{sampleName}.png')
    #
    adata.raw = adata
    # adata = adata[:, adata.var.highly_variable]
    sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata)
    sc.pl.pca(adata, show=False); plt.savefig(f'{out_dir}/pca_{sampleName}.png')
    sc.pl.pca_variance_ratio(adata, log=True, show=False); plt.savefig(f'{out_dir}/pca_variance_ratio_{sampleName}.png')
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.pl.umap(adata, show=False); plt.savefig(f'{out_dir}/umap_{sampleName}.png')
    sc.tl.leiden(adata)
    sc.pl.umap(adata, color=['leiden'], show=False); plt.savefig(f'{out_dir}/umap_{sampleName}.png')
    sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
    sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, show=False); plt.savefig(f'{out_dir}/rank_genes_groups_{sampleName}.png')
    result = adata.uns['rank_genes_groups']
    groups = result['names'].dtype.names
    pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']}).to_csv(f'{out_dir}/degs_{sampleName}.csv')
    adata.obs["sample"] = sampleName
    adata.X = adata.layers["count"]
    # adatas = anndata.concat([adatas, adata])#, merge="only")
    adatas = anndata.AnnData.concatenate(adatas, adata, join="outer", fill_value=0)


adatas.layers["count"] = adatas.X.copy()


adata_mouse.obs["cellID"] = [str[:16] for str in adata_mouse.obs_names.tolist()]
adata_mouse.obs["cellID"] = adata_mouse.obs["cellID"].str.cat(adata_mouse.obs["sample"], sep="-")

adatas.obs["cellID"] = [str[:16] for str in adatas.obs_names.tolist()]
adatas.obs["cellID"] = adatas.obs["cellID"].str.cat(adatas.obs["sample"], sep="-")

common_cells = np.intersect1d(adata_mouse.obs["cellID"].tolist(), adatas.obs["cellID"].tolist())

adatas = adatas[adatas.obs["cellID"].isin(common_cells),:]
adata_mouse = adata_mouse[adata_mouse.obs["cellID"].isin(common_cells),:]


adatas.obs = adata_mouse.obs
adatas.obsm = adata_mouse.obsm

plt.close("all")

sc.pp.normalize_total(adatas, target_sum=1e4)
sc.pp.log1p(adatas)
sc.pp.scale(adatas, max_value=10)

sc.pl.umap(adatas, color=["GRCh38_SALL4", "GRCh38_EPCAM"], vmax=0.0001)

# mouseとhumanの割合
>>> adata[:,[str[1:2]==str[1:2].upper() for str in adata.var_names]].X.sum()
4674.0
>>> adata.X.sum()
27008912.0

sc.pl.highest_expr_genes(adatas[:,[str[1:2]==str[1:2].upper() for str in adatas.var_names]], n_top=50)

sc.pl.umap(adatas, color="CT010467.1", vmax=1000)
## 一部の遺伝子が大多数、妥当ではなさそう