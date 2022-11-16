
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import os

out_dir = "/mnt/Donald/tsuji/mouse_sc/221116/"
os.makedirs(out_dir)
os.chdir(out_dir)

adata=sc.read_h5ad(f'{out_dir}/adata_total_cellClass1.h5')

## microgliaを含みそうなクラスタに限る scanoramaで揃えてplotする
adata = adata[adata.obs["cellClass1"]=="cellClass1-1"].copy()
adata.X = adata.layers["count"].copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40, use_rep="Scanorama")
sc.tl.umap(adata)
sc.tl.leiden(adata )
sc.pl.umap(adata, color=[ "sample"], alpha=0.3, save="microglia_and_others_sample.png")
sc.pl.umap(adata, color=['leiden'], legend_loc="on data", alpha=0.3, save="microglia_and_others_leiden.png")

sc.pl.umap(adata, color=["Tmem119", "Ptprc", "Aif1", "Itgam", "Cx3cr1", "Adgre1", "Cd68", "Cd40"], legend_loc="on data", save="microglia_markers.png")
sc.pl.heatmap(adata, ["Tmem119", "Ptprc", "Aif1", "Itgam", "Cx3cr1", "Adgre1", "Cd68", "Cd40"], groupby="leiden", dendrogram=True, save="microglia.png", figsize=(30,20))
sc.pl.stacked_violin(adata, ["Tmem119", "Ptprc", "Aif1", "Itgam", "Cx3cr1", "Adgre1", "Cd68", "Cd40"], groupby="leiden", dendrogram=True, save="microglia_markers.png", figsize=(20,30))


adata = adata[(adata.obs.leiden!="17")&(adata.obs.leiden!="21")&(adata.obs.leiden!="12")&(adata.obs.leiden!="16")].copy()
sc.tl.dendrogram(adata,"leiden")
sc.pl.stacked_violin(adata, ["Tmem119", "Ptprc", "Aif1", "Itgam", "Cx3cr1", "Adgre1", "Cd68"], groupby="leiden", dendrogram=True, save="microglia_markers2.png", figsize=(20,30))


adata = adata[(adata.obs.leiden!="11")&(adata.obs.leiden!="13")&(adata.obs.leiden!="14")&(adata.obs.leiden!="15")&(adata.obs.leiden!="19")].copy()

# クラスタごとのsamples割合を積み上げ棒グラフにする
adata.obs.groupby([ "sample"]).size()
# sample 大体同じ
# Day4       2640
# Day7       2742
# Day10      2653
# KO_Day7    2909

df = pd.crosstab(adata.obs['leiden'], adata.obs['sample'], normalize='index')
df.plot.bar(stacked=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
plt.tight_layout()
plt.savefig(f'{out_dir}/stacked_bar.png')


df = adata.obs.groupby(["leiden", "sample"]).size().reset_index()
df.columns = ['leiden', 'sample', "count"]
plt.figure(figsize=(20, 5))
sns.barplot(data=df, x="leiden", y="count", hue="sample")
plt.savefig(f'{out_dir}/lined_bar.png')


# クラスタごとのenrich
impla_gmt = parse_gmt('/home/hirose/Documents/main/gmts/impala.gmt')
msig_gmt = parse_gmt('/home/hirose/Documents/main/gmts/msigdb.v7.4.symbols.gmt')
mouse2human = pd.read_csv("/home/hirose/Documents/main/gmts/mouse2human.txt", sep="\t")

result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
res_df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']})
res_df.to_csv(f'{out_dir}/degs_microglia.csv')

for i in range(len(res_df.columns)//2):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-4 ").sort_values("pvals").human[:100]
    pval_df = gene_set_enrichment(cl_genes, total_genes, impla_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('annotation/enrich_impla_cluster_' + str(i) + '.csv')

