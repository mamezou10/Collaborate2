import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

wd = '/mnt/Donald/hoshino/2_kidney/Mouse_kidney/Visium/preprocess221031'
os.makedirs(wd, exist_ok=True)
os.chdir(wd)

import scanpy as sc
import anndata as an
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scanorama

# read 4 samples
adata_A = sc.read_visium("/mnt/Donald/hoshino/2_kidney/Mouse_kidney/Visium/SpaceRanger_Output/A_PBS/",
                         library_id = "PBS")
adata_B = sc.read_visium("/mnt/Donald/hoshino/2_kidney/Mouse_kidney/Visium/SpaceRanger_Output/B_CTL/",
                         library_id = "CTL")
adata_C = sc.read_visium("/mnt/Donald/hoshino/2_kidney/Mouse_kidney/Visium/SpaceRanger_Output/C_PE2/",
                         library_id = "PE2")
adata_D = sc.read_visium("/mnt/Donald/hoshino/2_kidney/Mouse_kidney/Visium/SpaceRanger_Output/D_PE1/",
                         library_id = "PE1")

adata_A.var_names_make_unique()
adata_B.var_names_make_unique()
adata_C.var_names_make_unique()
adata_D.var_names_make_unique()

# adata_A.layers["count"] = pd.DataFrame.sparse.from_spmatrix(adata_A.X)
# adata_B.layers["count"] = pd.DataFrame.sparse.from_spmatrix(adata_B.X)
# adata_C.layers["count"] = pd.DataFrame.sparse.from_spmatrix(adata_C.X)
# adata_D.layers["count"] = pd.DataFrame.sparse.from_spmatrix(adata_D.X)

library_names = ["PBS","CTL","PE2","PE1"]

adata = adata_A.concatenate(
    adata_B, adata_C, adata_D, 
    batch_key="library_id",
    uns_merge="unique",
    batch_categories=library_names
)

# filter
adata.var['mt'] = adata.var_names.str.startswith('mt-') 
adata.var['hb'] = adata.var_names.str.contains(("^Hb.*-"))
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','hb'], percent_top=None, log1p=False, inplace=True)
sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'pct_counts_hb'],
             jitter=0.4, groupby = 'library_id', rotation= 45, save="violin.png")

# need to plot the two sections separately and specify the library_id
fig = plt.figure(figsize=(30,20));count=1
for i, library in enumerate(library_names):
    for sig in ["total_counts", "n_genes_by_counts",'pct_counts_mt', 'pct_counts_hb']:
        axes = fig.add_subplot(4, 4, count)
        sc.pl.spatial(adata[adata.obs.library_id == library,:], library_id=library, show=False, color = sig, ax=axes)
        count += 1

plt.tight_layout()
plt.savefig("figures/qc.png")
plt.show()

# filter spots
keep = (adata.obs['pct_counts_hb'] < 20) & (adata.obs['pct_counts_mt'] < 30) & (adata.obs['n_genes_by_counts'] > 1000)
adata = adata[keep,:]

# replot
fig = plt.figure(figsize=(30,20));count=1
for i, library in enumerate(library_names):
    for sig in ["total_counts", "n_genes_by_counts",'pct_counts_mt', 'pct_counts_hb']:
        axes = fig.add_subplot(4, 4, count)
        sc.pl.spatial(adata[adata.obs.library_id == library,:], library_id=library, show=False, color = sig, ax=axes)
        count += 1

plt.tight_layout()
plt.savefig("figures/qc_after_filter_spot.png")
plt.show()

# highest genes
sc.pl.highest_expr_genes(adata, n_top=20, save="_after_filter_genes.png")

# filter genes
mito_genes = adata.var_names.str.startswith('mt-')
keep = np.invert(mito_genes)
adata = adata[:,keep]

### analysis
# save the counts to a separate object for later, we need the normalized counts in raw for DEG dete
adata.layers["count"] = pd.DataFrame.sparse.from_spmatrix(adata.X)

sc.pp.normalize_total(adata, inplace=True)
sc.pp.log1p(adata)
# take 1500 variable genes per batch and then use the union of them.
sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=1500, inplace=True, batch_key="library_id")

# subset for variable genes
adata.raw = adata
adata = adata[:,adata.var.highly_variable_nbatches > 0]

# scale data
sc.pp.scale(adata)

# Podocyte marker gene
fig, axs = plt.subplots(1, 4, figsize=(15, 10))
for i, library in enumerate(library_names):
    sc.pl.spatial(adata[adata.obs.library_id == library,:], library_id=library, color = ["Podxl"], show=False, ax=axs[i])

plt.tight_layout(); plt.savefig("figures/podocytes.png")
    

## dimension reduction
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added="clusters")
sc.pl.umap(adata, color=["clusters", "library_id"], palette=sc.pl.palettes.default_20, save="spot_cluster.png")

clusters_colors = dict(zip([str(i) for i in range(len(adata.obs.clusters.cat.categories))], adata.uns["clusters_colors"]))

fig, axs = plt.subplots(1, 4, figsize=(15, 10))
for i, library in enumerate(["PBS","CTL","PE2","PE1"]):
    ad = adata[adata.obs.library_id == library, :].copy()
    sc.pl.spatial(
        ad, img_key="hires", library_id=library, color="clusters", size=1.5,
        palette=[v for k, v in clusters_colors.items()
            if k in ad.obs.clusters.unique().tolist() ],
        legend_loc=None, show=False, ax=axs[i],)

plt.tight_layout()
plt.savefig("figures/spatial_cluster.png")


## integration
adatas = {}
for batch in library_names:
    adatas[batch] = adata[adata.obs['library_id'] == batch,]

# run scanorama.integrate
adatas = list(adatas.values())
scanorama.integrate_scanpy(adatas, dimred = 50)
# Get all the integrated matrices.
scanorama_int = [ad.obsm['X_scanorama'] for ad in adatas]
all_s = np.concatenate(scanorama_int)
# add to the AnnData object
adata.obsm["Scanorama"] = all_s

sc.pp.neighbors(adata, use_rep="Scanorama")
sc.tl.umap(adata)
sc.tl.leiden(adata, key_added="clusters")
sc.pl.umap(adata, color=["clusters", "library_id"], palette=sc.pl.palettes.default_20, save="spot_cluster_integrated.png")
clusters_colors = dict(zip([str(i) for i in range(len(adata.obs.clusters.cat.categories))], adata.uns["clusters_colors"]))


# replot after integrate
fig, axs = plt.subplots(1, 4, figsize=(15, 10))
for i, library in enumerate(["PBS","CTL","PE2","PE1"]):
    ad = adata[adata.obs.library_id == library, :].copy()
    sc.pl.spatial(
        ad, img_key="hires", library_id=library, color="clusters", size=1.5,
        palette=[v for k, v in clusters_colors.items()
            if k in ad.obs.clusters.unique().tolist() ],
        legend_loc="on data", show=False, ax=axs[i],)

plt.tight_layout(); 
plt.savefig("figures/spatial_cluster_integrated.png")

# legendゲット
sc.pl.spatial(adata[adata.obs.library_id == "PE2", :], img_key="hires", library_id="PE2", color="clusters", size=1.5,
        palette=[v for k, v in clusters_colors.items() if k in ad.obs.clusters.unique().tolist() ],
        save = "spatial_cluster_integrated_legend.png")


# run t-test 
sc.tl.rank_genes_groups(adata, "clusters", method="wilcoxon")
# plot as heatmap for cluster5 genes
sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, groupby="clusters", show_gene_labels=True, save=".png")
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
res_df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']})
res_df.to_csv('degs.csv')

marker_genes = annotate_geneset(res_df, pre="merker_gene")

marker_genes = parse_gmt('merker_gene_marker.gmt')
impla_gmt = parse_gmt('/home/hirose/Documents/main/gmts/impala.gmt')
msig_gmt = parse_gmt('/home/hirose/Documents/main/gmts/msigdb.v7.4.symbols.gmt')
Panglao_gmt = parse_gmt('/home/hirose/Documents/main/gmts/PanglaoDB_Augmented_2021.txt')
mouse2human = pd.read_csv("/home/hirose/Documents/main/gmts/mouse2human.txt", sep="\t")


for i in range(len(res_df.columns)//2):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-4 ").sort_values("pvals").human[:100]
    pval_df = gene_set_enrichment(cl_genes, total_genes, Panglao_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('annotation/enrich_impla_cluster_'+ str(i) + '.csv')


# Podocyte clusterを例示

sc.pl.spatial(adata[(adata.obs.library_id == "PE2") & (adata.obs.clusters == "9") , :], 
            img_key="hires", library_id="PE2", color="clusters", size=1.5,
            palette=[[v for k, v in clusters_colors.items() if k in ad.obs.clusters.unique().tolist() ][9]],
            save = "spatial_cluster_integrated_cluster9.png")

fig, axs = plt.subplots(1, 4, figsize=(15, 10))
for i, library in enumerate(["PBS","CTL","PE2","PE1"]):
    ad = adata[adata.obs.library_id == library, :].copy()
    sc.pl.spatial(
        ad[ad.obs.clusters=="9",:], img_key="hires", library_id=library, color="clusters", size=1.5,
        palette=[[v for k, v in clusters_colors.items() if k in ad.obs.clusters.unique().tolist() ][9]],
        legend_loc=None, show=False, ax=axs[i],)

plt.tight_layout(); 
plt.savefig("figures/spatial_cluster_integrated_cluster9.png")


# save
adata.write_h5ad("/mnt/Donald/hoshino/2_kidney/Mouse_kidney/Visium/preprocess221031/sp_adata.h5ad")