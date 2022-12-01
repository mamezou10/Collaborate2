import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata
import os

out_dir = "/mnt/Donald/tsuji/mouse_sc/221130/"
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)


## splice countデータ読み込み
c_d4 = sc.read_h5ad("/mnt/Donald/tsuji/mouse_sc/221130_kb_out/kb_tsuji/day4/counts_filtered/adata.h5ad")
c_d7 = sc.read_h5ad("/mnt/Donald/tsuji/mouse_sc/221130_kb_out/kb_tsuji/day7/counts_filtered/adata.h5ad")
c_d10 = sc.read_h5ad("/mnt/Donald/tsuji/mouse_sc/221130_kb_out/kb_tsuji/day10/counts_filtered/adata.h5ad")
ko_d7 = sc.read_h5ad("/mnt/Donald/tsuji/mouse_sc/221130_kb_out/kb_tsuji/ko_day7/counts_filtered/adata.h5ad")

sampleNames = ["Day4", "Day7", "Day10", "KO_Day7"]

adatas = c_d4.concatenate(
    [c_d7, c_d10, ko_d7],
    batch_key="sample",
    uns_merge="unique",
    batch_categories=sampleNames
)

## gene nameに変換
adatas.var_names = adata.var["gene_name"].tolist()

## mCherry countデータ読み込み
mcherry_d4 = sc.read_10x_h5("/mnt/Donald/tsuji/mouse_sc/221122/tsuji_day4_mCherry/outs/filtered_feature_bc_matrix.h5")
mcherry_d7 = sc.read_10x_h5("/mnt/Donald/tsuji/mouse_sc/221122/tsuji_day7_mCherry/outs/filtered_feature_bc_matrix.h5")
mcherry_d10 = sc.read_10x_h5("/mnt/Donald/tsuji/mouse_sc/221122/tsuji_day10_mCherry/outs/filtered_feature_bc_matrix.h5")
mcherry_KO_d7 = sc.read_10x_h5("/mnt/Donald/tsuji/mouse_sc/221122/tsuji_KO_day7_mCherry/outs/filtered_feature_bc_matrix.h5")

mCherry_adata = mcherry_d4.concatenate(
    [mcherry_d7, mcherry_d10, mcherry_KO_d7],
    batch_key="library_id",
    uns_merge="unique",
    batch_categories=sampleNames
)
mcherry = pd.DataFrame.sparse.from_spmatrix(mCherry_adata.X)
mcherry.columns = ['test1', 'mCherry']
mcherry.index = [str.replace("-1", "") for str in mCherry_adata.obs_names]

adatas.obs = pd.merge(adatas.obs, mcherry[["mCherry"]], how="left", left_index=True, right_index=True).fillna({"mCherry":0})



adata = adatas.copy()

sampleName = "total"
## preprocess
adata.layers["count"] = adata.X.copy()
sc.pl.highest_expr_genes(adata, n_top=20, show=False); plt.savefig(f'{out_dir}/highest_expr_genes_{sampleName}.png')
sc.pp.filter_cells(adata, min_genes=200); adata
sc.pp.filter_genes(adata, min_cells=3); adata
#
adata.var['mt'] = adata.var_names.str.startswith('mt-') 
adata.var_names_make_unique()
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, show=False); plt.savefig(f'{out_dir}/violin_{sampleName}.png')
#
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', show=False);     plt.savefig(f'{out_dir}/scatter1_{sampleName}.png');     # plt.show()
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', show=False); plt.savefig(f'{out_dir}/scatter2_{sampleName}.png'); # plt.show()
# adata = adata[adata.obs.n_genes_by_counts > 2500, :]; adata
# adata = adata[adata.obs.n_genes_by_counts < 5000, :]; adata
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
sc.pl.umap(adata, color=['leiden'], legend_loc="on data", show=False); plt.savefig(f'{out_dir}/umap_{sampleName}.png')
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, show=False); plt.savefig(f'{out_dir}/rank_genes_groups_{sampleName}.png')
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']}).to_csv(f'{out_dir}/degs_{sampleName}.csv')
adata.obs["sample"] = sampleName

plt.close("all")
sc.pl.umap(adata, color=['mCherry', "total_counts", "Tmem119"], legend_loc="on data", show=False); plt.savefig(f'{out_dir}/umap2_{sampleName}.png')


adata.obs["mCherry"] = list(adata.obs["mCherry"])
adata.write_h5ad("adata.h5")



