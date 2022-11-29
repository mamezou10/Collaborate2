import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata
import os

out_dir = "/mnt/Donald/tsuji/mouse_sc/221124_human/"
os.makedirs(out_dir,exist_ok=True)
os.chdir(out_dir)
# plt.savefig(f'{out_dir}/q_adata_markers.png');plt.close()

c_d4 = sc.read_10x_h5("/mnt/Daisy/tsuji_sc_mouse/tsuji_human_day4/outs/filtered_feature_bc_matrix.h5")
c_d7 = sc.read_10x_h5("/mnt/Daisy/tsuji_sc_mouse/tsuji_human_day7/outs/filtered_feature_bc_matrix.h5")
c_d10 = sc.read_10x_h5("/mnt/Daisy/tsuji_sc_mouse/tsuji_human_day10/outs/filtered_feature_bc_matrix.h5")
ko_d7 = sc.read_10x_h5("/mnt/Daisy/tsuji_sc_mouse/tsuji_human_KO_day7/outs/filtered_feature_bc_matrix.h5")

## preprocess
samples = [c_d4, c_d7, c_d10, ko_d7 ]
sampleNames = ["Day4", "Day7", "Day10", "KO_Day7"]
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
    adata.var['mt'] = adata.var_names.str.startswith('MT-') 
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, show=False); plt.savefig(f'{out_dir}/violin_{sampleName}.png')
    #
    sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', show=False);     plt.savefig(f'{out_dir}/scatter1_{sampleName}.png');     # plt.show()
    sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', show=False); plt.savefig(f'{out_dir}/scatter2_{sampleName}.png'); # plt.show()
    # adata = adata[adata.obs.n_genes_by_counts > 2500, :]; adata
    ##  adata = adata[adata.obs.n_genes_by_counts < 5000, :]; adata
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
    adatas = anndata.AnnData.concatenate(adatas, adata)


adatas.layers["count"] = adatas.X.copy()

sc.pp.normalize_total(adatas, target_sum=1e4)
sc.pp.log1p(adatas)
sc.pp.scale(adatas, max_value=10)
sc.tl.pca(adatas)
sc.pp.neighbors(adatas, n_neighbors=10, n_pcs=40)
sc.tl.umap(adatas)
sc.tl.leiden(adatas, resolution=0.3)
plt.close("all")
sc.pl.umap(adatas, color=["total_counts", "sample", "leiden"])

# leiden2が何か？

adata = adatas[:,adatas.X.sum(axis=0)>10].copy()

sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
res = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']})
res.to_csv(f'{out_dir}/degs_total.csv')

# pd.DataFrame.sparse.from_spmatrix(adatas[:,adatas.var_names.isin(res["2_n"][:10])].layers["count"]).mean(axis=0)

# マウスのデータからumap取ってくる
adata_mouse = sc.read_h5ad("/mnt/Donald/tsuji/mouse_sc/221124_for_duplicates/adata_duplicate.h5")

common_cells = np.intersect1d(adata_mouse.obs_names, adata.obs_names)
adata = adata[common_cells,:]
adata.obsm = adata_mouse[common_cells,:].obsm

# 改めてUMAPかく
sc.pl.umap(adata, color=["total_counts", "sample", "leiden"])
sc.pl.umap(adata, color=res["2_n"][:12], layer="count")
sc.pl.umap(adatas, color=["ABCA3", "KRAS",  "BRAF", "CD47", "CD24"], layer="count")

sc.pl.violin(adatas, ["CD24", "CD47"], groupby="sample", log=False, use_raw=False)#, stripplot=True,)
 jitter=True, size=1, layer=None, scale='width', order=None, multi_panel=None, xlabel='', ylabel=None, rotation=None, show=None, save=None, ax=None)


sc.pl.heatmap(adatas,["CD24", "CD47"], groupby="sample")
