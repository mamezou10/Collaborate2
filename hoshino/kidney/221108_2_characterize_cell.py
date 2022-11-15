
import os
import sys
wd = '/mnt/Donald/hoshino/2_kidney/221108/'
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
import seaborn as sns
# import deepcolor
np.random.seed(1)
torch.manual_seed(1)

sys.path.append("/home/hirose/Documents/main/Collaborate2/hoshino/kidney")
import deepcolor_mod.workflow as wf

lt_df = pd.read_csv('/home/hirose/Documents/main/gmts/shimam_ligand_target_matrix.csv', index_col=0)
lt_df = lt_df.drop(index=lt_df.index[[14725]]) ## ligand名がNanになっている行
mouse2human = pd.read_csv("/home/hirose/Documents/main/gmts/mouse2human.txt", sep="\t")

sp_adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/221103/sp_adata_train.h5')
sc_adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/221103/sc_adata_train.h5')

sp_adata = wf.calculate_clusterwise_distribution(sc_adata, sp_adata, 'leiden')


df = pd.DataFrame(sp_adata.obs)
df = df[df.clusters=="9"].sort_values("library_id")
samples = df["library_id"].tolist()
sample_colors=[]
for item in samples:
    if item=="PBS":
        item_mod = "b"
    elif item=="CTL":
        item_mod = "k"
    else:
        item_mod = "g"
    sample_colors.append(item_mod)

g = sns.clustermap(df[["6","11","17"]], row_colors = sample_colors, row_cluster=True)

## cluster９の中で　6 highのspot確認
sns.clustermap(df[["6","11","17"]].iloc[g.dendrogram_row.reordered_ind[357:],:], row_cluster=False)
plt.show()


#cluster９の中で　6 highのspotにおける6との共局在
coloc_cell = sp_adata[g.dendrogram_row.reordered_ind[357:],:].obsm["map2sc"].T @ sp_adata[g.dendrogram_row.reordered_ind[357:],:].obsm["map2sc"] 
no_coloc_cell = sp_adata[g.dendrogram_row.reordered_ind[:357],:].obsm["map2sc"].T @ sp_adata[g.dendrogram_row.reordered_ind[:357],:].obsm["map2sc"] 

coloc_with_cluster6 = coloc_cell[sc_adata.obs.leiden=="6"].mean(axis=0)
coloc_without_cluster6 = no_coloc_cell[sc_adata.obs.leiden=="6"].mean(axis=0)

obs = pd.DataFrame(sc_adata.obs)
obs["coloc_with_cluster6"] = coloc_with_cluster6.tolist()
obs["coloc_without_cluster6"] = coloc_without_cluster6.tolist()
sc_adata.obs = obs

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

sc.pl.umap(sc_adata, color=["coloc_with_cluster6", "coloc_without_cluster6"], vmax=0.00001, save="coloc_with_cluster6.png")


## cluster6がMphageの中でどのようなさぶ細胞か
kari_adata = sc_adata.copy()
kari_adata.X = kari_adata.layers["count"].copy()
sc.pp.normalize_total(kari_adata)
sc.pp.log1p(kari_adata)
sc.tl.rank_genes_groups(kari_adata, "leiden", groups=["6", "11","17"])
result = kari_adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
res_df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']})
res_df.to_csv('degs_cluster6_in_macrophage.csv')

impla_gmt = parse_gmt('/home/hirose/Documents/main/gmts/impala.gmt')
msig_gmt = parse_gmt('/home/hirose/Documents/main/gmts/msigdb.v7.4.symbols.gmt')
mouse2human = pd.read_csv("/home/hirose/Documents/main/gmts/mouse2human.txt", sep="\t")

for i in range(len(res_df.columns)//2):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-2 ").sort_values("pvals").human[:200]
    pval_df = gene_set_enrichment(cl_genes, total_genes, impla_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('enrich_impla_cluster6_in_macrophage_'+ res_df.columns[2*i] + '.csv')

for i in range(len(res_df.columns)//2):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-2").sort_values("pvals").human[:300]
    pval_df = gene_set_enrichment(cl_genes, total_genes, msig_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('enrich_msig_gmt_cluster6_in_macrophage_'+ res_df.columns[2*i] + '.csv')



## cluster10がProximal tubule cellの中でどのようなさぶ細胞か
kari_adata = sc_adata.copy()
kari_adata.X = kari_adata.layers["count"].copy()
sc.pp.normalize_total(kari_adata)
sc.pp.log1p(kari_adata)
sc.tl.rank_genes_groups(kari_adata, "leiden", groups=["7", "10", "14"])
result = kari_adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
res_df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']})
res_df.to_csv('degs_cluster10_in_macrophage.csv')

for i in range(len(res_df.columns)//2):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-2 ").sort_values("pvals").human[:300]
    pval_df = gene_set_enrichment(cl_genes, total_genes, impla_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('enrich_impla_cluster10_in_proximal_tubule_cell_'+ res_df.columns[2*i] + '.csv')

for i in range(len(res_df.columns)//2):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-2").sort_values("pvals").human[:300]
    pval_df = gene_set_enrichment(cl_genes, total_genes, msig_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('enrich_msig_gmt_cluster10_in_proximal_tubule_cell_'+ res_df.columns[2*i] + '.csv')





sc_adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/221103/sc_adata_train.h5')


# sc.pl.spatial(sp_adata[sp_adata.obs["library_id"]=="CTL", :], library_id="CTL", color=["0", "2","13","19"])

# boolean_l = (((sp_adata.obs["library_id"]=="PE1") | (sp_adata.obs["library_id"]=="PE2")) & (sp_adata.obs["clusters"]=="9")).tolist()
boolean_l = ( (sp_adata.obs["clusters"]=="9")).tolist()
# boolean_l = ((sp_adata.obs["library_id"]=="PBS") & (sp_adata.obs["clusters"]=="9")).tolist()
# boolean_l = ((sp_adata.obs["library_id"]=="CTL") & (sp_adata.obs["clusters"]=="9")).tolist()

sc_adata.obsm["map2sp"] = np.array(pd.DataFrame(sc_adata.obsm["map2sp"]).loc[:, boolean_l].copy())
sc_adata.uns['log1p']["base"] = None

human_genes = pd.merge(pd.DataFrame(sc_adata.var_names), mouse2human, how="left", left_on=0, right_on="mouse")
sc_adata_human = sc_adata.copy()
sc_adata_human.var_names = human_genes.human
sc_adata_human = sc_adata_human[:,[sc_adata_human.var_names[i] is not np.nan for i in list(range(len(sc_adata_human.var_names)))]]
sc_adata_human.var_names_make_unique()
np.random.seed(1)

fig, coexp_cc_df = wf.calculate_proximal_cell_communications(sc_adata_human, 'leiden', lt_df, ["6", "11", "17"], ["10"],       # macrophages
                                                            celltype_sample_num=100, ntop_genes=4000,  
                                                            each_display_num=3, 
                                                            role="sender", edge_thresh=1)
fig

#こじま先生由来
# sc_adata.var_names[sc_adata.var_names.str.startswith('Ifn')]

