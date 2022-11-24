import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata
import os


out_dir = "/mnt/Donald/tsuji/mouse_sc/221122/"
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)



bustools_d7 = sc.read_h5ad("/mnt/Daisy/kb_human_mouse/tsuji_day7/counts_unfiltered/adata.h5ad")
cellranger_d7 = sc.read_10x_h5("/mnt/Daisy/tsuji_sc_mouse/tsuji_day7/outs/filtered_feature_bc_matrix.h5")
mcherry_d7 = sc.read_10x_h5("/mnt/Donald/tsuji/mouse_sc/221122/tsuji_day7_mCherry/outs/filtered_feature_bc_matrix.h5")


sc.pp.normalize_total(cellranger_d7, target_sum=1e4)
sc.pp.log1p(cellranger_d7)
sc.pp.scale(cellranger_d7)
sc.tl.pca(cellranger_d7)
sc.pp.neighbors(cellranger_d7, n_neighbors=10, n_pcs=40)
sc.tl.umap(cellranger_d7)
cellranger_d7.obs_names = [str[:16] for str in cellranger_d7.obs_names.tolist()]

sc.pp.filter_cells(bustools_d7, min_genes=200)
sc.pp.filter_genes(bustools_d7, min_cells=3)
sc.pp.normalize_total(bustools_d7, target_sum=1e4)
sc.pp.log1p(bustools_d7)
sc.pp.scale(bustools_d7)

mcherry = pd.DataFrame.sparse.from_spmatrix(mcherry_d7.X)
mcherry.columns = ['test1', 'mCherry']
mcherry.index = [str[:16] for str in mcherry_d7.obs_names.tolist()]

common_cells = np.intersect1d(cellranger_d7.obs_names.tolist(), bustools_d7.obs_names.tolist())

bustools_d7 = bustools_d7[bustools_d7.obs_names.isin(common_cells),:]
cellranger_d7 = cellranger_d7[cellranger_d7.obs_names.isin(common_cells),:]

cellranger_d7.obs["kari"] = "kari"
bustools_d7.obsm = cellranger_d7.obsm

sc.pl.umap(cellranger_d7, color="GRCh38_EPCAM", vmax=0.01, save="cellranger.png")
sc.pl.umap(bustools_d7, color="EPCAM", vmax=0.01)

# mCherryを載せる
mcherry = pd.DataFrame.sparse.from_spmatrix(mcherry_d7.X)
mcherry.columns = ['test1', 'mCherry']
mcherry.index = [str[:16] for str in mcherry_d7.obs_names.tolist()]


cellranger_d7.obs = pd.merge(pd.DataFrame(cellranger_d7.obs), mcherry, how="left", left_index=True, right_index=True).fillna(0)
sc.pl.umap(cellranger_d7, color=["mm10___Tmem119","mm10___Ptprc","mm10___Epcam","GRCh38_NAPSA", "mCherry"], save="mCherry.png")

## 謎クラスタの解明
adata = cellranger_d7.copy()
sc.tl.leiden(adata, resolution=0.3)
sc.pl.umap(adata, color=['leiden'], legend_loc="on data", show=False); plt.savefig(f'{out_dir}/umap_nazo.png')
sc.tl.rank_genes_groups(adata, 'leiden', groups=[str(i) for i in range(0,24)], method='t-test')

result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
res_df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']})
res_df.to_csv('degs_nazo.csv')


sc.pl.umap(cellranger_d7, color=["mm10___Krt18","mm10___S100a6","mm10___Krt8","mm10___Wfdc2","mm10___Cx3cr1", "mCherry"], save="mCherry_cluster7.png")


## お渡し用
kari = pd.DataFrame(cellranger_d7.X)
kari.index=cellranger_d7.obs_names
kari.columns=cellranger_d7.var_names

kari.to_csv("gem_day7.tsv", sep="\t")


## どこでmCherry消えるか
adata = sc.read_10x_h5("/mnt/Daisy/tsuji_sc_mouse/tsuji_day7/outs/filtered_feature_bc_matrix.h5")
adata.var_names_make_unique()
mcherry = pd.DataFrame.sparse.from_spmatrix(mcherry_d7.X)
mcherry.columns = ['test1', 'mCherry']
mcherry.index = [str[:16] for str in mcherry_d7.obs_names.tolist()]

adata.obs_names = [str[:16] for str in adata.obs_names.tolist()]
adata.obs = pd.merge(pd.DataFrame(adata.obs), mcherry, how="left", left_index=True, right_index=True)
adata.obs = adata.obs.fillna({"mCherry":0})
sum(adata.obs["mCherry"]>0)


sc.pp.filter_cells(adata, min_genes=200); sum(adata.obs["mCherry"]>0)
sc.pp.filter_genes(adata, min_cells=3); sum(adata.obs["mCherry"]>0)
#
adata.var['mt'] = adata.var_names.str.startswith('mt-') 
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True); sum(adata.obs["mCherry"]>0)
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, show=True)#; plt.savefig(f'{out_dir}/violin_{sampleName}.png')
#

# adata = adata[adata.obs.n_genes_by_counts > 2500, :]; adata
adata = adata[adata.obs.n_genes_by_counts < 5000, :]; sum(adata.obs["mCherry"]>0)
adata = adata[adata.obs.pct_counts_mt < 5, :];       adata
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)




