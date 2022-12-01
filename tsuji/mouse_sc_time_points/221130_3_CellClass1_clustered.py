import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata
import os
import seaborn as sns

out_dir = "/mnt/Donald/tsuji/mouse_sc/221130/"
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)

adata_total = sc.read_h5ad("adata_cellClass1.h5")

## microgliaを含みそうなクラスタに限る scanoramaで揃えてplotする
adata = adata_total[(adata_total.obs["CellClass1"]=="CellClass1-1") | (adata_total.obs["leiden"]=="6")].copy()
adata.X = adata.layers["count"].copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5, key_added="CellClass1_clusters" )
sc.pl.umap(adata, color=[ "sample"], alpha=0.3, save="microglia_and_others_sample.png")
sc.pl.umap(adata, color=['CellClass1_clusters'], legend_loc="on data", alpha=0.3, save="microglia_and_others_CellClass1_clusters.png")
sc.pl.umap(adata, color=['leiden'], legend_loc="on data", alpha=0.3, save="microglia_and_others_leiden.png")
sc.pl.umap(adata, color=['mCherry'], legend_loc="on data", save="microglia_and_others_mCherry.png")

sc.pl.umap(adata, color=["Tmem119", "Cx3cr1", "Ptprc", "Aif1", "Itgam", "Adgre1", "Cd68", "Cd40"], legend_loc="on data", save="microglia_markers.png")
sc.pl.stacked_violin(adata, ["Tmem119", "Cx3cr1", "Ptprc", "Aif1", "Itgam", "Adgre1", "Cd68"], groupby="CellClass1_clusters", dendrogram=True, save="microglia_markers.png", figsize=(8,10))


# クラスタごとのsamples割合を積み上げ棒グラフにする
adata.obs.groupby([ "sample"]).size()
# # sample 大体同じ
# sample
# Day4       2997
# Day7       3628
# Day10      3107
# KO_Day7    3237

df = pd.crosstab(adata.obs['CellClass1_clusters'], adata.obs['sample'], normalize='index')
df.plot.bar(stacked=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
plt.tight_layout()
plt.savefig(f'{out_dir}/stacked_bar.png')


df = adata.obs.groupby(["CellClass1_clusters", "sample"]).size().reset_index()
df.columns = ['CellClass1_clusters', 'sample', "count"]
df.CellClass1_clusters = df.CellClass1_clusters.cat.set_categories(["0","1","2","3","6","9","10","12","13","7","8","11","14","4","5","15"], ordered=True)
plt.figure(figsize=(20, 5))
sns.barplot(data=df, x="CellClass1_clusters", y="count", hue="sample")
plt.yscale('log')
plt.savefig(f'{out_dir}/lined_bar.png')


# クラスタごとのenrich
impla_gmt = parse_gmt('/home/hirose/Documents/main/gmts/impala.gmt')
msig_gmt = parse_gmt('/home/hirose/Documents/main/gmts/msigdb.v7.4.symbols.gmt')
msig_mouse_gmt = parse_gmt('/home/hirose/Documents/main/gmts/msigdb.v2022.1.Mm.symbols.gmt')
mouse2human = pd.read_csv("/home/hirose/Documents/main/gmts/mouse2human.txt", sep="\t")

sc.tl.rank_genes_groups(adata, 'CellClass1_clusters', method='t-test')
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
res_df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals', 'logfoldchanges']})
res_df.to_csv(f'{out_dir}/degs_CellClass1.csv')

# impala
for i in range(len(res_df.columns)//3):
    deg_df = res_df.iloc[:, (3*i):(3*i+3)]
    deg_df.columns =["gene", "pvals", 'logFC']
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-4 & logFC>1").sort_values("pvals").human[:100]
    pval_df = gene_set_enrichment(cl_genes, total_genes, impla_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["CellClass1_cluster"]] = "CellClass1_cluster"+str(i)
    pval_df.to_csv('annotation/enrich_impla_cluster_' + str(i) + '.csv')

df = pd.DataFrame(columns = ["Unnamed: 0", "pval", "CellClass1_cluster"])
for i in range(len(res_df.columns)//3):
    res = pd.read_csv('annotation/enrich_impla_cluster_' + str(i) + '.csv')
    df = pd.concat([df,res[res.pval<0.01]], axis=0)

df.to_csv('annotation_impla.csv')

# msig_mouse
for i in range(len(res_df.columns)//3):
    deg_df = res_df.iloc[:, (3*i):(3*i+3)]
    deg_df.columns =["gene", "pvals", 'logFC']
    # deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.gene
    cl_genes = deg_df.query("pvals < 10**-4 & logFC>1").sort_values("pvals").gene[:100]
    pval_df = gene_set_enrichment(cl_genes, total_genes, msig_mouse_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["CellClass1_cluster"]] = "CellClass1_cluster"+str(i)
    pval_df.to_csv('annotation/enrich_msig_mouse_cluster_' + str(i) + '.csv')

df = pd.DataFrame(columns = ["Unnamed: 0", "pval", "CellClass1_cluster"])
for i in range(len(res_df.columns)//3):
    res = pd.read_csv('annotation/enrich_msig_mouse_cluster_' + str(i) + '.csv')
    df = pd.concat([df,res[res.pval<0.01]], axis=0)

df.to_csv('annotation_msig_mouse.csv')



sc.pl.umap(adata,color=["Cd24a", "Cd47"], save="dontEAT.png")


adata.write_h5ad('adata_cellClass1_clustered.h5')


