import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

# make adata
adata= sc.read_h5ad("/mnt/Daisy/sc_KIDNEY_GSE157079/kidney.h5ad")
celltype = pd.read_csv("/mnt/Daisy/sc_KIDNEY_GSE157079/GSE157079_P0_adult_clusters.txt", sep="\t")
celltype = celltype[["barcodes"   ,    "clusters"]]
df1 = pd.DataFrame(adata.obs)
df1["barcodes"] = df1.index
adata.obs = pd.merge(df1, celltype, how="left", on="barcodes")

# basic preprocess
out_dir = "/mnt/Daisy/sc_KIDNEY_GSE157079/figures/"
adata.var_names_make_unique(); adata
adata.obs.index = adata.obs.index.astype(str)
adata.var.index = adata.var.index.astype(str)
adata.layers["count"] = adata.X.copy()
sc.pl.highest_expr_genes(adata, n_top=20, show=False); plt.savefig(f'{out_dir}/highest_expr_genes.png')
sc.pp.filter_cells(adata, min_genes=200); adata
sc.pp.filter_genes(adata, min_cells=3); adata

adata.var['mt'] = adata.var_names.str.startswith('mt-') 
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, show=False); plt.savefig(f'{out_dir}/violin.png')

sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', show=False);     plt.savefig(f'{out_dir}/scatter1.png');     # plt.show()
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', show=False); plt.savefig(f'{out_dir}/scatter2.png'); # plt.show()

adata = adata[adata.obs.n_genes_by_counts < 10000, :]
adata = adata[adata.obs.pct_counts_mt < 15, :]
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(adata, show=False); plt.savefig(f'{out_dir}/highly_variable_genes.png')
#
adata.raw = adata
adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])
sc.pp.scale(adata)
sc.tl.pca(adata)
sc.pl.pca(adata, show=False); plt.savefig(f'{out_dir}/pca.png')
sc.pl.pca_variance_ratio(adata, log=True, show=False); plt.savefig(f'{out_dir}/pca_variance_ratio.png')
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=14)
sc.tl.tsne(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata)
sc.pl.tsne(adata, color=['leiden', "clusters"], legend_loc="on data", show=False); plt.savefig(f'{out_dir}/tsne.png')
sc.pl.umap(adata, color=['leiden', "clusters"], show=False); plt.savefig(f'{out_dir}/umap.png')
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False, show=False); plt.savefig(f'{out_dir}/rank_genes_groups.png')
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']}).to_csv(f'{out_dir}/degs.csv')


