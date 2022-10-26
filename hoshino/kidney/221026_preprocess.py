import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata

out_dir = "/mnt/Donald/hoshino/2_kidney/221024/"
# plt.savefig(f'{out_dir}/q_adata_markers.png');plt.close()

CTL = sc.read_10x_h5("/mnt/Donald/hoshino/2_kidney/Mouse_kidney/scRNA/CellRanger_Output/Kidney_CTL/filtered_feature_bc_matrix.h5")
PBS = sc.read_10x_h5("/mnt/Donald/hoshino/2_kidney/Mouse_kidney/scRNA/CellRanger_Output/Kidney_PBS/filtered_feature_bc_matrix.h5")
PE = sc.read_10x_h5("/mnt/Donald/hoshino/2_kidney/Mouse_kidney/scRNA/CellRanger_Output/Kidney_PE/filtered_feature_bc_matrix.h5")


samples = [CTL, PBS, PE]
sampleNames = ["CTL", "PBS", "PE"]
adatas = anndata.AnnData(np.ones((1,len(CTL.var_names)))); adatas.var_names=CTL.var_names; adatas.var_names_make_unique()
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
    adata = adata[adata.obs.n_genes_by_counts > 2500, :]; adata
    adata = adata[adata.obs.n_genes_by_counts < 10000, :]; adata
    adata = adata[adata.obs.pct_counts_mt < 20, :];       adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    sc.pl.highly_variable_genes(adata, show=False); plt.savefig(f'{out_dir}/highly_variable_genes_{sampleName}.png')
    #
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
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
    adatas = anndata.AnnData.concatenate(adatas, adata)


    
plt.close("all")
adata = adatas[1:,:].copy()
sampleName = "total"
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, show=False); plt.savefig(f'{out_dir}/violin_{sampleName}.png')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)

sc.pl.pca(adata, show=False); plt.savefig(f'{out_dir}/pca_{sampleName}.png')
sc.pl.pca_variance_ratio(adata, log=True, show=False); plt.savefig(f'{out_dir}/pca_variance_ratio_{sampleName}.png')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.pl.umap(adata, show=False); plt.savefig(f'{out_dir}/umap_{sampleName}.png')
sc.tl.leiden(adata)
sc.pl.umap(adata, color=['leiden', "sample"], alpha=0.3, show=False); plt.savefig(f'{out_dir}/umap_{sampleName}.png')
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, show=False); plt.savefig(f'{out_dir}/rank_genes_groups_{sampleName}.png')
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']}).to_csv(f'{out_dir}/degs_{sampleName}.csv')

adata.write_h5ad(f'{out_dir}/adata_total.h5')




