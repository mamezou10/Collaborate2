import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata
import os

out_dir = "/mnt/Donald/tsuji/mouse_sc/221124_for_duplicates/"
os.makedirs(out_dir)
os.chdir(out_dir)
# plt.savefig(f'{out_dir}/q_adata_markers.png');plt.close()

c_d4 = sc.read_10x_h5("/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_cellranger/CMT167_1G2_D4_5DE/outs/filtered_feature_bc_matrix.h5")
c_d7 = sc.read_10x_h5("/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_cellranger/CMT167_1G2_D7_5DE/outs/filtered_feature_bc_matrix.h5")
c_d10 = sc.read_10x_h5("/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_cellranger/CMT167_1G2_D10_5DE/outs/filtered_feature_bc_matrix.h5")
ko_d7 = sc.read_10x_h5("/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_cellranger/CMT167_DK1_D7_5DE/outs/filtered_feature_bc_matrix.h5")

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
    adata.var['mt'] = adata.var_names.str.startswith('mt-') 
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
sc.tl.leiden(adatas)
sc.pl.umap(adatas, color=["total_counts", "sample", "Tmem119", "Ptprc", "Cd68", "Csf1r"])


# mCherry載せる
mcherry_d4 = sc.read_10x_h5("/mnt/Donald/tsuji/mouse_sc/221122/tsuji_day4_mCherry/outs/filtered_feature_bc_matrix.h5")
mcherry_d7 = sc.read_10x_h5("/mnt/Donald/tsuji/mouse_sc/221122/tsuji_day7_mCherry/outs/filtered_feature_bc_matrix.h5")
mcherry_d10 = sc.read_10x_h5("/mnt/Donald/tsuji/mouse_sc/221122/tsuji_day10_mCherry/outs/filtered_feature_bc_matrix.h5")
mcherry_KO_d7 = sc.read_10x_h5("/mnt/Donald/tsuji/mouse_sc/221122/tsuji_KO_day7_mCherry/outs/filtered_feature_bc_matrix.h5")

library_names = sampleNames = ["Day4", "Day7", "Day10", "KO_Day7"]

adata = mcherry_d4.concatenate(
    [mcherry_d7, mcherry_d10, mcherry_KO_d7],
    batch_key="library_id",
    uns_merge="unique",
    batch_categories=library_names
)
# mCherryを載せる
mcherry = pd.DataFrame.sparse.from_spmatrix(adata.X)
mcherry.columns = ['test1', 'mCherry']
mcherry.index = adata.obs_names
#mcherry.index = [str[:16] for str in adata.obs_names.tolist()]


adatas.obs["sampleID"] = [str[:18] for str in adatas.obs_names.tolist()] 
adatas.obs["sampleID"] = adatas.obs["sampleID"].str.cat(adatas.obs["sample"], sep="-")

kari = pd.merge(adatas.obs, mcherry, how="left", left_on="sampleID", right_index=True )
kari = kari.fillna({"mCherry":0})
adatas.obs = kari
sc.pl.umap(adatas, color=["total_counts", "sample", "Tmem119", "Ptprc", "Cd68", "Csf1r", "mCherry"], save="various.png")


##  前のmicrogliaクラスタがどこにいるか
data_dir = "/mnt/Donald/tsuji/mouse_sc/221116/"
adata=sc.read_h5ad(f'{data_dir}/adata_total_cellClass1.h5')
adata = adata[adata.obs["cellClass1"]=="cellClass1-1"].copy()
adata.X = adata.layers["count"].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40, use_rep="Scanorama")
sc.tl.umap(adata)
sc.tl.leiden(adata )
sc.pl.umap(adata, color="leiden")
adata.obs["sampleID"] = [str[:18] for str in adata.obs_names.tolist()] 
adata.obs["sampleID"] = adata.obs["sampleID"].str.cat(adata.obs["sample"], sep="-")

adatas.obs = pd.merge(kari, pd.DataFrame(adata.obs), how="left", left_on="sampleID", right_on="sampleID")
sc.pl.umap(adatas, color=["n_genes_x"])

## 前の全体クラスタがどこにいるか
adata=sc.read_h5ad(f'{data_dir}/adata_total_scanorama.h5')
adata.obs["sampleID"] = [str[:18] for str in adata.obs_names.tolist()] 
adata.obs["sampleID"] = adata.obs["sampleID"].str.cat(adata.obs["sample"], sep="-")
sc.tl.leiden(adata, resolution=0.5, key_added="coarse_leiden")
adatas.obs = pd.merge(pd.DataFrame(adatas.obs), pd.DataFrame(adata.obs), how="left", left_on="sampleID", right_on="sampleID")


adata=sc.read_h5ad(f'{data_dir}/adata_total_cellClass1.h5')
adata.obs["sampleID"] = [str[:18] for str in adata.obs_names.tolist()] 
adata.obs["sampleID"] = adata.obs["sampleID"].str.cat(adata.obs["sample"], sep="-")
adatas.obs = pd.merge(pd.DataFrame(adatas.obs), pd.DataFrame(adata.obs[["cellClass1", "sampleID"]]), how="left", left_on="sampleID", right_on="sampleID")


sc.pl.umap(adatas, color=["leiden_y", "coarse_leiden_y", "cellClass1_y"], alpha=0.7, legend_loc="on data", save="duplicate_cluster.png")


adatas.obs_names = kari.index

obs = adatas.obs[["sampleID", "sample", "cellClass1_y", "n_genes", "n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt", "mCherry",  "leiden_x",     "leiden_y",          "clusters_x",         "coarse_leiden_x",          "leiden"]]
obs.columns =    ["sampleID", "sample", "cellClass1_y", "n_genes", "n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt", "mCherry",  "leiden_total", "leiden_cellClass1", "clusters_scanorama", "coarse_leiden_single_cell", "leiden_single_cell"]

adatas.obs = obs

adatas = adatas[1:,:].copy()
adatas.obs["mCherry"] = list(adatas.obs["mCherry"])

adatas.write_h5ad("adata_duplicate.h5")





# sc.pl.umap(adatas, color=["leiden_y", "coarse_leiden_y", "leiden", "leiden_x", "clusters_x"], alpha=0.7, legend_loc="on data")