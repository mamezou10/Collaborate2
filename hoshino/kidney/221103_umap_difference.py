
import os
import sys
wd = '/mnt/Donald/hoshino/2_kidney/221103_analysis/'
os.makedirs(wd, exist_ok=True)
os.chdir(wd)

from importlib import reload
import scipy as sp
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import anndata
import torch
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import deepcolor
np.random.seed(1)
torch.manual_seed(1)

sys.path.append("/home/hirose/Documents/main/Collaborate2/hoshino/kidney")
import deepcolor_mod.workflow as wf


sc_adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/221103/adata_total_annotated2.h5')

fig = plt.figure(figsize=(15,5))
axes = fig.add_subplot(1,3,1)
sc.pl.umap(sc_adata[sc_adata.obs["sample"]=="PBS", :], color="Annotation", legend_loc=None, show=False, ax=axes)
axes.set_title("PBS")
axes = fig.add_subplot(1,3,2)
sc.pl.umap(sc_adata[sc_adata.obs["sample"]=="CTL", :], color="Annotation", legend_loc=None, show=False, ax=axes)
axes.set_title("CTL")
axes = fig.add_subplot(1,3,3)
sc.pl.umap(sc_adata[(sc_adata.obs["sample"]=="PE")|(sc_adata.obs["sample"]=="PE"), :], color="Annotation", show=False, ax=axes)
axes.set_title("PE")
plt.savefig("figures/umap_each_sample.png")
plt.show()

plt.close("all")


## ratio の計算

adata_PBS = pd.DataFrame(sc_adata[(sc_adata.obs["sample"]=="PBS") , :].obs)
ratio_df = pd.DataFrame(adata_PBS.groupby(["Annotation"])["leiden"].value_counts(dropna=False, normalize=True))
ratio_df.columns = ["ratio_PBS"]
ratio_df = ratio_df.reset_index()
df_PBS = ratio_df[ratio_df.ratio_PBS!=0]

adata_CTL = pd.DataFrame(sc_adata[(sc_adata.obs["sample"]=="CTL") , :].obs)
ratio_df = pd.DataFrame(adata_CTL.groupby(["Annotation"])["leiden"].value_counts(dropna=False, normalize=True))
ratio_df.columns = ["ratio_CTL"]
ratio_df = ratio_df.reset_index()
df_CTL = ratio_df[ratio_df.ratio_CTL!=0]

adata_PE = pd.DataFrame(sc_adata[(sc_adata.obs["sample"]=="PE")|(sc_adata.obs["sample"]=="PE2") , :].obs)
ratio_df = pd.DataFrame(adata_PE.groupby(["Annotation"])["leiden"].value_counts(dropna=False, normalize=True))
ratio_df.columns = ["ratio_PE"]
ratio_df = ratio_df.reset_index()
df_PE = ratio_df[ratio_df.ratio_PE!=0]

df = pd.merge(df_PBS,df_CTL, how="outer", on=["Annotation", "level_1"]).merge(df_PE, how="outer", on=["Annotation", "level_1"])

for celltype in np.unique(df.Annotation):
    fig_df = df[df.Annotation==celltype].iloc[:,2:].T
    fig_df.columns = df[df.Annotation==celltype].level_1
    axes = fig_df.plot.bar(stacked=True, rot=0)
    axes.legend(loc=4) 
    axes.set_title(celltype)
    plt.savefig(f'figures/ratio_{celltype}.png')


# Endotherial cluster0同士の比較
kari_adata = sc_adata[sc_adata.obs["leiden"]=="0",:].copy()
kari_adata.X = kari_adata.layers["count"].copy()
sc.pp.normalize_total(kari_adata)
sc.pp.log1p(kari_adata)
sc.tl.rank_genes_groups(kari_adata, "sample")
result = kari_adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
res_df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']})
res_df.to_csv('degs_cluster0.csv')

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
    cl_genes = deg_df.query("pvals < 10**-2 ").sort_values("pvals").human[:100]
    pval_df = gene_set_enrichment(cl_genes, total_genes, impla_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('annotation/enrich_impla_cluster_'+ res_df.columns[2*i] + '.csv')

for i in range(len(res_df.columns)//2):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-2").sort_values("pvals").human[:100]
    pval_df = gene_set_enrichment(cl_genes, total_genes, msig_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('annotation/enrich_msig_gmt_cluster_'+ res_df.columns[2*i] + '.csv')


# M-phage cluster6同士の比較
kari_adata = sc_adata[sc_adata.obs["leiden"]=="6",:].copy()
kari_adata.X = kari_adata.layers["count"].copy()
sc.pp.normalize_total(kari_adata)
sc.pp.log1p(kari_adata)
sc.tl.rank_genes_groups(kari_adata, "sample")
result = kari_adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
res_df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']})
res_df.to_csv('degs_cluster6.csv')

for i in range(len(res_df.columns)//2):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-2 ").sort_values("pvals").human[:100]
    pval_df = gene_set_enrichment(cl_genes, total_genes, impla_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('annotation/enrich_impla_cluster6_'+ res_df.columns[2*i] + '.csv')

for i in range(len(res_df.columns)//2):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-2").sort_values("pvals").human[:100]
    pval_df = gene_set_enrichment(cl_genes, total_genes, msig_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('annotation/enrich_msig_gmt_cluster6_'+ res_df.columns[2*i] + '.csv')

# M-phage cluster17同士の比較
kari_adata = sc_adata[sc_adata.obs["leiden"]=="17",:].copy()
kari_adata.X = kari_adata.layers["count"].copy()
sc.pp.normalize_total(kari_adata)
sc.pp.log1p(kari_adata)
sc.tl.rank_genes_groups(kari_adata, "sample")
result = kari_adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
res_df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']})
res_df.to_csv('degs_cluster17.csv')

for i in range(len(res_df.columns)//2):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-2 ").sort_values("pvals").human[:100]
    pval_df = gene_set_enrichment(cl_genes, total_genes, impla_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('annotation/enrich_impla_cluster17_'+ res_df.columns[2*i] + '.csv')

for i in range(len(res_df.columns)//2):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-2").sort_values("pvals").human[:100]
    pval_df = gene_set_enrichment(cl_genes, total_genes, msig_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('annotation/enrich_msig_gmt_cluster17_'+ res_df.columns[2*i] + '.csv')



# M-phage cluster17同士の比較
kari_adata = sc_adata[sc_adata.obs["leiden"]=="17",:].copy()
kari_adata.X = kari_adata.layers["count"].copy()
sc.pp.normalize_total(kari_adata)
sc.pp.log1p(kari_adata)
sc.tl.rank_genes_groups(kari_adata, "sample")
result = kari_adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
res_df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']})
res_df.to_csv('degs_cluster17.csv')

for i in range(len(res_df.columns)//2):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-2 ").sort_values("pvals").human[:100]
    pval_df = gene_set_enrichment(cl_genes, total_genes, impla_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('annotation/enrich_impla_cluster17_'+ res_df.columns[2*i] + '.csv')

for i in range(len(res_df.columns)//2):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-2").sort_values("pvals").human[:100]
    pval_df = gene_set_enrichment(cl_genes, total_genes, msig_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('annotation/enrich_msig_gmt_cluster17_'+ res_df.columns[2*i] + '.csv')








## interaction
lt_df = pd.read_csv('/home/hirose/Documents/main/gmts/ligand_target_df.csv', index_col=0)
mouse2human = pd.read_csv("/home/hirose/Documents/main/gmts/mouse2human.txt", sep="\t")
human_genes = pd.merge(pd.DataFrame(sc_adata.var_names), mouse2human, how="left", left_on=0, right_on="mouse")
sc_adata_human = sc_adata.copy()
sc_adata_human.var_names = human_genes.human
sc_adata_human = sc_adata_human[:,[sc_adata_human.var_names[i] is not np.nan for i in list(range(len(sc_adata_human.var_names)))]]
sc_adata_human.var_names_make_unique()
sc_adata_human.var_names_make_unique()
# sc_adata_human.obs_names_make_unique()
np.random.seed(1)
sc_adata_human.X = sc_adata_human.layers["count"]
sc.pp.normalize_total(sc_adata_human, target_sum=1e4)
sc.pp.log1p(sc_adata_human)
# sc_adata_human.uns['log1p']["base"] = None
# del sc_adata_human.uns['log1p']['base']
# reload(wf)
# reload(sc.preprocessing._highly_variable_genes)
fig, coexp_cc_df = wf.calculate_proximal_cell_communications(sc_adata_human, 
                            'Annotation', lt_df, ["Podocytes"], 
                            celltype_sample_num=100, ntop_genes=500, each_display_num=3, role="sender", edge_thresh=0.3)
fig.show()

np.random.seed(1)
fig, coexp_cc_df = wf.calculate_proximal_cell_communications(sc_adata_human, 
                            'leiden', lt_df, ["0"], 
                            celltype_sample_num=100, ntop_genes=500, each_display_num=3, role="sender", edge_thresh=0.3)
fig.show()


