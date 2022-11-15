import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata

out_dir = "/mnt/Donald/hoshino/2_kidney/221108/"
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
    sc.pp.filter_cells(adata, min_genes=200); adata
    sc.pp.filter_genes(adata, min_cells=3); adata
    #
    adata.var['mt'] = adata.var_names.str.startswith('mt-') 
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    #
    adata = adata[adata.obs.n_genes_by_counts > 2500, :]; adata
    adata = adata[adata.obs.n_genes_by_counts < 10000, :]; adata
    adata = adata[adata.obs.pct_counts_mt < 40, :];       adata
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
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    adata.obs["sample"] = sampleName
    adata.X = adata.layers["count"]
    adatas = anndata.AnnData.concatenate(adatas, adata)



adatas.layers["count"] = adatas.X.copy()

adata = adatas[1:,:].copy()
sampleName = "total"
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, show=False); plt.savefig(f'{out_dir}/violin_{sampleName}.png')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)

sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata)




sc_adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/221103/sc_adata_train.h5')
sp_adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/221103/sp_adata_train.h5')
sc_adata = sc_adata[:, sc_adata.layers['count'].toarray().sum(axis=0) > 10]
sp_adata = sp_adata[:, sp_adata.layers['count'].toarray().sum(axis=0) > 10]
common_genes = np.intersect1d(sc_adata.var_names, sp_adata.var_names)
sc_adata = sc_adata[:, common_genes]

sc_adata.X = sc_adata.layers["count".copy()]
sc.pp.normalize_total(sc_adata, target_sum=1e4)
sc.pp.log1p(sc_adata)
sc.pp.scale(sc_adata, max_value=10)
sc.tl.pca(sc_adata)
sc.pp.neighbors(sc_adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(sc_adata)

adata.obs = sc_adata.obs.copy()
adata.obsm = sc_adata.obsm.copy()

sc.pl.umap(adata, save="kari.png", color=adata.var_names[adata.var_names.str.startswith('Ifn')])

adata.var_names[adata.var_names.str.startswith('Ifn')]

sc.pl.umap(adata, save="kari.png", color=adata.var_names[adata.var_names.str.startswith('Ifn')])
sc.pl.umap(adata, save="kari2.png", color=adata.var_names[adata.var_names.str.startswith('Isg')])

df = pd.DataFrame(sc_adata.obs)
df = df[df.clusters=="9"].sort_values("library_id")
df = df[["library_id", "6","11","17"]].melt(id_vars=["library_id"])
df.library_id = df.library_id.str.replace("PE1", "PE").replace("PE2", "PE")

sns.violinplot(x='library_id', y="value", data=df,  dodge=True, jitter=True, palette='Set3')
plt.savefig("figures/violin_cluster9_macrophage.png")
plt.show()

sc.pl.violin(adata[adata.obs.leiden=="0"],["Isg15", "Isg20"], groupby="sample", save="kari3.png")#, log=False, use_raw=None, stripplot=True, jitter=True, size=1, layer=None, scale='width', order=None, multi_panel=None, xlabel='', ylabel=None, rotation=None, show=None, save=None, ax=None, **kwds)

sc.pl.violin(adata[adata.obs.leiden=="6"],["Isg15", "Isg20"], groupby="sample", save="kari4.png")

, color=['leiden', "sample"], alpha=0.3, show=False); plt.savefig(f'{out_dir}/umap_{sampleName}.png')
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, show=False); plt.savefig(f'{out_dir}/rank_genes_groups_{sampleName}.png')
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']}).to_csv(f'{out_dir}/degs_{sampleName}.csv')

adata.write_h5ad(f'{out_dir}/adata_total.h5')




