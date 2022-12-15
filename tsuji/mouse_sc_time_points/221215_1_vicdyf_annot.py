
import copy
import enum
import itertools
import os
from re import A
from tkinter import N

data_dir = "/mnt/Donald/tsuji/mouse_sc/221207/"

out_dir = "/mnt/Donald/tsuji/mouse_sc/221215_2/"
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)

np.random.seed(1)
torch.manual_seed(1)

from sklearn.neighbors import NearestNeighbors
import math
import numpy as np
import scvelo as scv
import joblib
import os
import anndata
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import sys

# sys.path.append('/home/hirose/Documents/main/Collaborate2/tsuji/mouse_sc_time_points/envdyn')
# import envdyn
sys.path.append('/home/hirose/Documents/main/Collaborate2/tsuji/mouse_sc_time_points/scripts')
sys.path.append('/home/hirose/Documents/main/Collaborate2/tsuji/mouse_sc_time_points/envdyn')
import utils 
import commons
import basic
import workflow
import condiff

import umap
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import colors
import importlib
import torch

import json
import pytorch_lightning as pl
import dataset
import modules
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from functorch import jacfwd, jvp, jacrev, vmap
# from scripts import hiroseflow
import importlib

impla_gmt = parse_gmt('/home/hirose/Documents/main/gmts/impala.gmt')
msig_gmt = parse_gmt('/home/hirose/Documents/main/gmts/msigdb.v7.4.symbols.gmt')
msig_mouse_gmt = parse_gmt('/home/hirose/Documents/main/gmts/msigdb.v2022.1.Mm.symbols.gmt')
mouse2human = pd.read_csv("/home/hirose/Documents/main/gmts/mouse2human.txt", sep="\t")

cd24 = pd.Series(['Cd24a', 'CD24'], index=['mouse', 'human'])
mouse2human = mouse2human.append(cd24, ignore_index=True)


est_adata = sc.read_h5ad(f'{data_dir}/est_adata.h5ad')
cdiff_adata = sc.read_h5ad(f'{data_dir}/cdiff_adata.h5ad')
cond1 = 'Day7' 
cond2 = 'KO_Day7' 

# 
commons.visuazlie_condiff_dynamics(est_adata, cdiff_adata, cond1, cond2, f'{out_dir}/condiff_flows.png', plot_num=30, plot_top_prop=0.8)
commons.plot_vicdyf_umap(est_adata, ['condition', 'leiden', 'CellClass1_clusters'], show=False)
plt.savefig(f'{out_dir}/condition_umap.png');plt.close('all')


## top_adata
top_adata = cdiff_adata[cdiff_adata.obs.total_dyn_diff > cdiff_adata.obs.total_dyn_diff.quantile(0.8)]

sc.pp.neighbors(top_adata, use_rep='X_vicdyf_zl')
sc.tl.leiden(top_adata, key_added='cdiff_cluster', resolution=0.5)
top_adata.obs.cdiff_cluster.unique()
top_adata_tmp = top_adata.copy()
top_adata_tmp.obsm['X_umap'] = top_adata.obsm['X_vicdyf_umap']
sc.pl.umap(top_adata_tmp, color=['cdiff_cluster', 'total_dyn_diff'], show=False)
plt.savefig(f'{out_dir}/cdiff_clusters.png');plt.close('all')

## 4から分岐を見たい

# visualize cdiff target scores
if not os.path.exists(f'{out_dir}/target_scores/'):
    os.makedirs(f'{out_dir}/target_scores/')

clusters = np.sort(top_adata.obs.cdiff_cluster.unique())
for cluster in clusters:
    est_adata.obsm['X_umap'] = est_adata.obsm['X_vicdyf_umap']
    cluster_adata = top_adata[top_adata.obs.cdiff_cluster == cluster]
    target_label = f'cdiff{cluster}_target'
    est_adata.obs[target_label] = condiff.calculate_cdiff_target_scores(cluster_adata, est_adata)
    sc.pl.umap(est_adata, color=target_label, color_map='OrRd', vcenter=0, show=False)
    # plt.show()
    plt.savefig(f'{out_dir}/target_scores/{cluster}.png');plt.close('all')


# targetの細胞を潜在空間上で再クラスタリング
est_adata_tmp = est_adata.copy()
sc.pp.neighbors(est_adata_tmp, use_rep='X_vicdyf_zl')
sc.tl.leiden(est_adata_tmp, key_added='coarse_leiden', resolution=0.6)
est_adata_tmp.obs.coarse_leiden.unique()
est_adata_tmp_tmp = est_adata_tmp.copy()
est_adata_tmp.obsm['X_umap'] = est_adata_tmp.obsm['X_vicdyf_umap']

sc.pl.umap(est_adata_tmp, color=['coarse_leiden'], legend_loc="on data", alpha=0.8, s=30, show=False)
plt.tight_layout(); plt.savefig(f'{out_dir}/coarse_leiden.png');plt.close('all')

sc.pl.violin(est_adata_tmp, "cdiff5_target", groupby="coarse_leiden", show=False)
plt.savefig(f'{out_dir}/target_scores/cdiff5_target_violin.png');plt.close('all')


est_adata_tmp.obs["cdiff5_target_score"] = "low"
est_adata_tmp.obs["cdiff5_target_score"][est_adata_tmp.obs["coarse_leiden"].isin(["4", "5", "6", "7"])] = "high"
# sc.pl.umap(est_adata_tmp, color=['cdiff5_target_score'], layer="lambda", s=30)

# cdijj5のtargetが高い細胞たちのenrichment
est_adata_tmp.uns['log1p']["base"] = None
sc.tl.rank_genes_groups(est_adata_tmp, 'cdiff5_target_score',  method='wilcoxon', key_added = "cdiff5_target_wilcoxon")
sc.pl.rank_genes_groups(est_adata_tmp, key="cdiff5_target_wilcoxon", show=False)
plt.savefig(f'{out_dir}/target_scores/cdiff5_target_deg.png');plt.close('all')


sc.tl.rank_genes_groups(est_adata_tmp, 'coarse_leiden', method='t-test')
result = est_adata_tmp.uns['rank_genes_groups']
groups = result['names'].dtype.names
res_df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals', 'logfoldchanges']})
res_df.to_csv(f'{out_dir}/degs_coarse_leiden.csv')

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

#
utils.plot_mean_flow(est_adata_tmp, cluster_key='coarse_leiden')
plt.savefig(f'{out_dir}/mean_vel_flow.png');plt.close('all')

ax = sc.pl.umap(est_adata_tmp[~est_adata_tmp.obs["CellClass1_clusters"].isin(["0", "1", "2", "6", "13"]),:], size=100, show=False)
sc.pl.umap(
    est_adata_tmp[est_adata_tmp.obs["CellClass1_clusters"].isin(["0", "1", "2", "6", "13"]),:],
    size=20,
    na_color="green", 
    # color="CellClass1_clusters",
    alpha=0.2,
    # edges_color="red",
    ax=ax,
    save="_phago_cells.png"
)


# cdiff cluster4から条件によって違いの大きそうな遺伝子
cluster="4"
est_adata_tmp = est_adata.copy()
est_adata_tmp.X = est_adata_tmp.layers["count"]
sc.pp.normalize_total(est_adata_tmp)
sc.pp.log1p(est_adata_tmp)

cluster_adata = top_adata[top_adata.obs.cdiff_cluster == cluster]
gene_scores = pd.Series(cluster_adata.layers['norm_cond_vel_diff'].mean(axis=0), index=cluster_adata.var_names)
source_cells = cluster_adata.obs_names
importlib.reload(condiff)
deg_cdiff_df, jac_adata = condiff.make_deg_cdiff_df(source_cells, est_adata_tmp, cluster, est_adata_tmp.obs[f'cdiff{cluster}_target'], gene_scores, q=0.8)
sc.tl.rank_genes_groups(est_adata_tmp, 'sample', groups=['KO_Day7'], reference='Day7')
deg_cdiff_df = commons.extract_deg_df(est_adata_tmp, 'KO_Day7')
#deg_cdiff_df['score'] = gene_scores[deg_cdiff_df.index]
deg_cdiff_df['score'] = gene_scores[np.intersect1d(deg_cdiff_df.index, gene_scores.index)]
deg_cdiff_df = deg_cdiff_df.query('pvals_adj < 0.01')
deg_cdiff_df.sort_values('score', ascending=False).to_csv(f'{out_dir}/{cluster}_score_for_bar.csv')
top_genes = deg_cdiff_df.sort_values('score', ascending=False).index[:10]
bottom_genes = deg_cdiff_df.sort_values('score', ascending=True).index[:10]
top_genes = np.concatenate([bottom_genes, top_genes[::-1]])
# deg_cdiff_df = deg_cdiff_df.sort_values("score")
fig, axes = plt.subplots(1, 1, figsize=(6 * 1, 5 * 1))
# axes.bar(deg_cdiff_df['names'], deg_cdiff_df['score'])
axes.bar(top_genes, deg_cdiff_df['score'][top_genes])
plt.xticks(rotation= 90)
fig.subplots_adjust(bottom=0.2)
fig.subplots_adjust(left=0.2)
axes.set_title(f'{cluster} difference genes')
axes.set_ylabel('cdiff difference KO_Day7/Day7')
# axes.set_ylim(-1,1)
#plt.show()
plt.savefig(f'{out_dir}/{cluster}_bar.png');plt.close('all')


## それをGSEA
import gseapy
deg_cdiff_df = pd.merge(deg_cdiff_df, mouse2human, left_on="names", right_on="mouse", how="left")
cdiff_df = pd.DataFrame({'gene': deg_cdiff_df.human, 'scores': deg_cdiff_df.score}).dropna().sort_values("scores", ascending=False)
pre_res = gseapy.prerank(cdiff_df, gene_sets='/home/hirose/Documents/main/gmts/impala.gmt', outdir=f'{out_dir}/gsea_{cluster}', graph_num=10, permutation_num=1000, format='png', seed=6, processes=12)

pre_res.res2d.sort_values('pval').head(n=10)

plt.close("all")

for term in pre_res.res2d.sort_values('pval').head(n=10).index:
    print(term)
    gseapy.gseaplot(rank_metric=pre_res.ranking,
         term=term,
         **pre_res.results[term],
         ofname=f'{out_dir}/gsea_{cluster}/{term}_gsea.png')



# ligand_targetバージョン、それぞれリガンドのターゲットスコア上位１００でGSEA
lt_df = pd.read_csv("/home/hirose/Documents/main/gmts/shimam_ligand_target_matrix.csv", index_col=0)

targ_genes_list = []
for gene in lt_df.columns:
    targ_genes = list(lt_df.sort_values(gene, ascending=False)[[gene]][:100].index)
    targ_genes[0:1] = [gene, str(len(targ_genes))]
    targ_genes_list.append("\t".join(targ_genes))

pd.Series(targ_genes_list).to_csv(f'{out_dir}/targ_gmt.gmt', header=False, index=False)

pre_res = gseapy.prerank(cdiff_df, gene_sets='/mnt/Donald/tsuji/mouse_sc/221215/targ_gmt.gmt', outdir=f'{out_dir}/gsea_lt_{cluster}', graph_num=10, permutation_num=1000, format='png', seed=6, processes=12)

pre_res.res2d.sort_values('pval').head(n=10)

plt.close("all")

for term in pre_res.res2d.sort_values('pval').head(n=10).index:
    print(term)
    gseapy.gseaplot(rank_metric=pre_res.ranking,
         term=term,
         **pre_res.results[term],
         ofname=f'{out_dir}/gsea_lt_{cluster}/{term}_gsea.png')







# from typing import  List
# rank_metric=pre_res.ranking
# term="Phagosome - Homo sapiens (human)|KEGG"
# hits= List[int]
# nes= float
# pval=float
# fdr=float
# RES=float
# pheno_pos= "po"
# pheno_neg= "ne",
# figsize = (6, 5.5)
# cmap= "seismic"
# ofname= None

# g = gseapy.GSEAPlot(
#     rank_metric,
#     term,
#     hits,
#     nes,
#     pval,
#     fdr,
#     RES,
#     pheno_pos,
#     pheno_neg,
#     figsize,
#     cmap,
#     ofname,
# )





res_list = []
jac_adata.varm['top_cdiff'] = pd.DataFrame({}, index=jac_adata.var_names)
top_cdiff_df = jac_adata.varm['top_cdiff']
clusters = np.sort(top_cdiff_df.columns)
for cluster in clusters:
    cdiff_df = pd.DataFrame({'gene': top_cdiff_df.index, 'scores': top_cdiff_df[cluster]})
    pre_res = gseapy.prerank(cdiff_df, gene_sets='db/meta/impala.gmt', outdir=f'{out_dir}/gsea/{cluster}', permutation_num=1000, format='png', seed=6, processes=12)
    pre_res.res2d.sort_values('pval').head(n=10)
    res_list.append(pre_res.res2d)







# ## barplot
# clusters = np.sort(top_adata.obs.cdiff_cluster.unique())
# for cluster in clusters:
#     est_adata_tmp = est_adata.copy()
#     est_adata_tmp.X = est_adata_tmp.layers["matrix"]
#     sc.pp.normalize_total(est_adata_tmp)
#     sc.pp.log1p(est_adata_tmp)
#     cluster_adata = top_adata[top_adata.obs.cdiff_cluster == cluster]
#     gene_scores = pd.Series(cluster_adata.layers['norm_cond_vel_diff'].mean(axis=0), index=cluster_adata.var_names)
#     source_cells = cluster_adata.obs_names
#     deg_cdiff_df = condiff.make_deg_cdiff_df(source_cells, est_adata_tmp, cluster, est_adata_tmp.obs[f'cdiff{cluster}_target'], gene_scores, q=0.8)
#     sc.tl.rank_genes_groups(est_adata_tmp, 'sample', groups=['sample5'], reference='sample4')
#     deg_cdiff_df = commons.extract_deg_df(est_adata_tmp, 'sample5')
#     deg_cdiff_df['score'] = gene_scores[deg_cdiff_df.index]
#     deg_cdiff_df = deg_cdiff_df.query('pvals_adj < 0.01')
#     deg_cdiff_df.to_csv(f'{out_dir}/{cluster}_score_for_bar.csv')
#     top_genes = deg_cdiff_df.sort_values('score', ascending=False).index[:10]
#     bottom_genes = deg_cdiff_df.sort_values('score', ascending=True).index[:10]
#     top_genes = np.concatenate([bottom_genes, top_genes[::-1]])
#     # deg_cdiff_df = deg_cdiff_df.sort_values("score")
#     fig, axes = plt.subplots(1, 1, figsize=(6 * 1, 5 * 1))
#     # axes.bar(deg_cdiff_df['names'], deg_cdiff_df['score'])
#     axes.bar(top_genes, deg_cdiff_df['score'][top_genes])
#     plt.xticks(rotation= 90)
#     fig.subplots_adjust(bottom=0.2)
#     fig.subplots_adjust(left=0.2)
#     axes.set_title(f'{cluster} difference genes')
#     axes.set_ylabel('cdiff difference mouse5/mouse4')
#     # axes.set_ylim(-1,1)
#     #plt.show()
#     plt.savefig(f'{out_dir}/cdiff_deg/{cluster}_bar.png');plt.close('all')




