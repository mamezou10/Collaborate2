import copy
import enum
import itertools
import os
from re import A
from tkinter import N
out_dir = '/mnt/Donald/ito/220907_mf_2'
#os.chdir(out_dir)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

os.chdir("/home/hirose/Documents/Collaborate/")

from sklearn.neighbors import NearestNeighbors
import math
import numpy as np
import scvelo as scv
import joblib
import os
import anndata
import pandas as pd
import seaborn as sns
from scipy.sparse import csr_matrix
from matplotlib import pyplot as plt
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from scipy import stats
import sys

sys.path.append('envdyn/')
import envdyn
sys.path.append('scripts/')
from scripts import commons
from scripts import utils
from scripts import basic
from scripts import workflow
from scripts import condiff

import umap
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib import colors
from adjustText import adjust_text
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
import importlib

np.random.seed(1)
torch.manual_seed(1)

importlib.reload(workflow)
importlib.reload(commons)
importlib.reload(utils)
importlib.reload(modules)
importlib.reload(envdyn)
importlib.reload(condiff)


## Macrophage
adata = sc.read_h5ad('/mnt/Donald/ito/220812/preprocessed_adata.h5ad')
adata = adata[adata.obs.annotate=="macrophage"]

scv.pp.moments(adata, n_neighbors=300)
corrs = pd.Series(scv.utils.vcorrcoef(adata.layers['Ms'], adata.layers['Mu'], axis=0), index=adata.var_names)
top_corr_genes = corrs.index[corrs > 0.6]
adata.var['highly_variable'] = False
adata.var['highly_variable'][top_corr_genes] = True
model_params = {
            'x_dim': 100,
            'z_dim': 10,
            'enc_z_h_dim': 128, 'enc_d_h_dim': 128, 'dec_z_h_dim': 128,
            'num_enc_z_layers': 2, 'num_enc_d_layers': 2,
            'num_dec_z_layers': 2, 'use_ambient': False, 'use_vamp': False, 'no_d_kld': False, 'decreasing_temp': False, 'dec_temp_steps': 30000, 'loss_mode': 'poisson'
}
checkpoint_dirname = f'{out_dir}/checkpoint_mf'
# adata.obs['sample'] =    np.array("mouse" + pd.Series(adata.obs_names).str.split("_", expand=True).iloc[:,2])
adata.obs['condition'] = adata.obs["sample"]

est_adata, lit_envdyn = workflow.conduct_cvicdyf_inference(adata, model_params, checkpoint_dirname, 
                                                            batch_size=50, two_step=False, dyn_mode=False, 
                                                            epoch=1000, patience=20, module=modules.Cvicdyf)

sc.pp.normalize_total(est_adata)
sc.pp.log1p(est_adata)
est_adata.write_h5ad(f'{out_dir}/mf_est_adata.h5ad')
est_adata = sc.read_h5ad(f'{out_dir}/mf_est_adata.h5ad')
torch.save(lit_envdyn.state_dict, f'{out_dir}/mf_lit_envdyn.pt')
importlib.reload(utils)
utils.plot_mean_flow(est_adata, cluster_key='leiden')
plt.savefig(f'{out_dir}/mf_mean_vel_flow.png');plt.close('all')

conds = np.unique(est_adata.obs['condition'].unique())
cond1 = 'sample1' 
cond2 = 'sample2' 
cond4 = 'sample4' 
cond5 = 'sample5' 

cdiff_adata = commons.estimate_two_cond_dynamics(est_adata, cond4, cond5, lit_envdyn)
cdiff_adata.write_h5ad(f'{out_dir}/mf_cdiff_adata.h5ad')
# cdiff_adata = sc.read_h5ad(f'{out_dir}/mf_cdiff_adata.h5ad')

grid_df = commons.select_grid_values(est_adata.obsm['X_vicdyf_umap'], 30, est_adata.obs['norm_vicdyf_mean_velocity'])
commons.plot_vicdyf_umap(est_adata[grid_df.num_idx.values], 'norm_vicdyf_mean_velocity', show=False)
plt.savefig(f'{out_dir}/mf_gird_umap_vel.png');plt.close('all')

commons.visuazlie_condiff_dynamics(est_adata, cdiff_adata, cond4, cond5, f'{out_dir}/mf_condiff_flows.png', plot_num=30, plot_top_prop=0.8)
commons.plot_vicdyf_umap(est_adata, ['condition', 'leiden'], show=False)
plt.savefig(f'{out_dir}/mf_condition_umap.png');plt.close('all')


top_adata = cdiff_adata[cdiff_adata.obs.total_dyn_diff > cdiff_adata.obs.total_dyn_diff.quantile(0.8)]

sc.pp.neighbors(top_adata, use_rep='X_vicdyf_zl')
sc.tl.leiden(top_adata, key_added='cdiff_cluster', resolution=0.5)
top_adata.obs.cdiff_cluster.unique()
top_adata_tmp = top_adata.copy()
top_adata_tmp.obsm['X_umap'] = top_adata.obsm['X_vicdyf_umap']
sc.pl.umap(top_adata_tmp, color=['cdiff_cluster', 'total_dyn_diff'], show=False)
plt.savefig(f'{out_dir}/cdiff_clusters.png');plt.close('all')

# visualize cdiff target scores
if not os.path.exists(f'{out_dir}/target_scores/'):
    os.makedirs(f'{out_dir}/target_scores/')

clusters = np.sort(top_adata.obs.cdiff_cluster.unique())
for cluster in clusters:
    est_adata.obsm['X_umap'] = est_adata.obsm['X_vicdyf_umap']
    cluster_adata = top_adata[top_adata.obs.cdiff_cluster == cluster]
    target_label = f'cdiff{cluster}_target'
    est_adata.obs[target_label] = condiff.calculate_cdiff_target_scores(cluster_adata, est_adata)
    sc.pl.umap(est_adata, color=target_label, color_map='coolwarm', vcenter=0, show=False)
    # plt.show()
    plt.savefig(f'{out_dir}/target_scores/{cluster}.png');plt.close('all')


importlib.reload(condiff)
est_adata.write_h5ad(f'{out_dir}/estset_adata.h5ad')
# cluster = '2'
if not os.path.exists(f'{out_dir}/cdiff_deg/'):
    os.makedirs(f'{out_dir}/cdiff_deg/')

est_adata = sc.read_h5ad(f'{out_dir}/estset_adata.h5ad')

clusters = np.sort(top_adata.obs.cdiff_cluster.unique())
for cluster in clusters:
    est_adata_tmp = est_adata.copy()
    est_adata_tmp.X = est_adata_tmp.layers["matrix"]
    sc.pp.normalize_total(est_adata_tmp)
    sc.pp.log1p(est_adata_tmp)
    cluster_adata = top_adata[top_adata.obs.cdiff_cluster == cluster]
    gene_scores = pd.Series(cluster_adata.layers['norm_cond_vel_diff'].mean(axis=0), index=cluster_adata.var_names)
    source_cells = cluster_adata.obs_names
    deg_cdiff_df = condiff.make_deg_cdiff_df(source_cells, est_adata_tmp, cluster, est_adata_tmp.obs[f'cdiff{cluster}_target'], gene_scores, q=0.8)
    
    # def make_deg_cdiff_df(source_cells, jac_adata, cluster, target_scores, gene_scores, q=0.8, method='wilcoxon'):
    # top_diff_cells = target_scores.index[target_scores > target_scores.quantile(q)]
    # jac_adata.obs['diff_pop'] = 'None'
    # jac_adata.obs['diff_pop'][source_cells] = 'Source'
    # jac_adata.obs['diff_pop'][top_diff_cells] = 'Target'
    sc.tl.rank_genes_groups(est_adata_tmp, 'sample', groups=['sample5'], reference='sample4')
    deg_cdiff_df = commons.extract_deg_df(est_adata_tmp, 'sample5')
    deg_cdiff_df['score'] = gene_scores[deg_cdiff_df.index]
    
    deg_cdiff_df = deg_cdiff_df.query('pvals_adj < 0.1')
    # deg_cdiff_df = deg_cdiff_df.dropna(subset="logfoldchanges")
    top_genes = deg_cdiff_df.sort_values('score', ascending=False).index[:5]
    bottom_genes = deg_cdiff_df.sort_values('score', ascending=True).index[:5]
    top_genes = np.concatenate([top_genes, bottom_genes])
    fig, axes = plt.subplots(1, 1, figsize=(6 * 1, 5 * 1))
    axes.scatter(deg_cdiff_df['score'], deg_cdiff_df['logfoldchanges'], s=10)
    axes.scatter(deg_cdiff_df['score'][top_genes], deg_cdiff_df['logfoldchanges'][top_genes], s=10)
    texts = []
    # for gene in top_genes:
    #     texts.append(axes.annotate(gene, (deg_cdiff_df['score'][gene], deg_cdiff_df['logfoldchanges'][gene]), size=15, va="center", arrowprops=dict(arrowstyle = "->", color = "black")))
    for gene in top_genes:
        texts.append(axes.annotate(gene, (deg_cdiff_df['score'][gene], deg_cdiff_df['logfoldchanges'][gene]), size=8, arrowprops=dict(arrowstyle = "-", color = "black")))
    adjust_text(texts)
    axes.set_xlabel('Difference scores')
    axes.set_ylabel('log2 fold changes')
    axes.set_ylim(-3,3)
    # plt.subplots_adjust(top=-3)
    # plt.show()
    plt.savefig(f'{out_dir}/cdiff_deg/{cluster}_2.png');plt.close('all')

from scipy import stats
intersect_cells = np.intersect1d(est_adata_tmp.obs_names, cdiff_adata.obs_names)
corr = [stats.spearmanr(cdiff_adata[intersect_cells,:].obs["cond2_ratio"], 
                        est_adata_tmp[intersect_cells, i ].layers["lambda"]).correlation for i in est_adata_tmp.var_names]
corrs = pd.DataFrame({"corr": corr, "genes": est_adata_tmp.var_names})


for cluster in clusters:
    est_adata_tmp = est_adata.copy()
    est_adata_tmp.X = est_adata_tmp.layers["matrix"]
    sc.pp.normalize_total(est_adata_tmp)
    sc.pp.log1p(est_adata_tmp)
    cluster_adata = top_adata[top_adata.obs.cdiff_cluster == cluster]
    gene_scores = pd.Series(cluster_adata.layers['norm_cond_vel_diff'].mean(axis=0), index=cluster_adata.var_names)
    source_cells = cluster_adata.obs_names
    deg_cdiff_df = condiff.make_deg_cdiff_df(source_cells, est_adata_tmp, cluster, est_adata_tmp.obs[f'cdiff{cluster}_target'], gene_scores, q=0.8)
    sc.tl.rank_genes_groups(est_adata_tmp, 'sample', groups=['sample5'], reference='sample4')
    deg_cdiff_df = commons.extract_deg_df(est_adata_tmp, 'sample5')
    deg_cdiff_df['score'] = gene_scores[deg_cdiff_df.index]
    deg_cdiff_df = deg_cdiff_df.query('pvals_adj < 0.01')
    # intersect_cells = np.intersect1d(est_adata_tmp.obs_names, cluster_adata.obs_names)
    # corr = [stats.spearmanr(cluster_adata[source_cells, :].obs["cond2_ratio"], 
    #                         est_adata_tmp[source_cells, i ].layers["lambda"]).correlation for i in est_adata_tmp.var_names]
    # corrs = pd.DataFrame({"corr": corr, "genes": est_adata_tmp.var_names})
    deg_cdiff_df = pd.merge(deg_cdiff_df, corrs, left_on="names", right_on="genes", how="inner")
    deg_cdiff_df = deg_cdiff_df.set_index("names")
    top_genes = deg_cdiff_df.sort_values('score', ascending=False).index[:10]
    bottom_genes = deg_cdiff_df.sort_values('score', ascending=True).index[:10]
    top_genes = np.concatenate([top_genes, bottom_genes])
    fig, axes = plt.subplots(1, 1, figsize=(6 * 1, 5 * 1))
    axes.scatter(deg_cdiff_df['score'], deg_cdiff_df['corr'], s=10)
    axes.scatter(deg_cdiff_df['score'][top_genes], deg_cdiff_df['corr'][top_genes], s=10)
    texts = []
    for gene in top_genes:
        texts.append(axes.annotate(gene, (deg_cdiff_df['score'][gene], deg_cdiff_df['corr'][gene]), size=8, arrowprops=dict(arrowstyle = "-", color = "black")))
    adjust_text(texts)
    axes.set_xlabel('Difference scores')
    axes.set_ylabel('Correlation lambda vs ratio')
    axes.set_ylim(-1,1)
    # plt.show()
    plt.savefig(f'{out_dir}/cdiff_deg/{cluster}_corr_total.png');plt.close('all')


for cluster in clusters:
    est_adata_tmp = est_adata.copy()
    est_adata_tmp.X = est_adata_tmp.layers["matrix"]
    sc.pp.normalize_total(est_adata_tmp)
    sc.pp.log1p(est_adata_tmp)
    cluster_adata = top_adata[top_adata.obs.cdiff_cluster == cluster]
    gene_scores = pd.Series(cluster_adata.layers['norm_cond_vel_diff'].mean(axis=0), index=cluster_adata.var_names)
    source_cells = cluster_adata.obs_names
    deg_cdiff_df = condiff.make_deg_cdiff_df(source_cells, est_adata_tmp, cluster, est_adata_tmp.obs[f'cdiff{cluster}_target'], gene_scores, q=0.95)
    sc.tl.rank_genes_groups(est_adata_tmp, 'sample', groups=['sample5'], reference='sample4')
    deg_cdiff_df = commons.extract_deg_df(est_adata_tmp, 'sample5')
    deg_cdiff_df['score'] = gene_scores[deg_cdiff_df.index]
    deg_cdiff_df = deg_cdiff_df.query('pvals_adj < 0.01')
    # intersect_cells = np.intersect1d(est_adata_tmp.obs_names, cluster_adata.obs_names)
    corr = [stats.spearmanr(cluster_adata[source_cells, :].obs["cond2_ratio"], 
                            est_adata_tmp[source_cells, i ].layers["lambda"]).correlation for i in est_adata_tmp.var_names]
    corrs = pd.DataFrame({"corr": corr, "genes": est_adata_tmp.var_names})
    deg_cdiff_df = pd.merge(deg_cdiff_df, corrs, left_on="names", right_on="genes", how="inner")
    deg_cdiff_df = deg_cdiff_df.set_index("names")
    top_genes = deg_cdiff_df.sort_values('score', ascending=False).index[:10]
    bottom_genes = deg_cdiff_df.sort_values('score', ascending=True).index[:10]
    top_genes = np.concatenate([top_genes, bottom_genes])
    fig, axes = plt.subplots(1, 1, figsize=(6 * 1, 5 * 1))
    axes.scatter(deg_cdiff_df['score'], deg_cdiff_df['corr'], s=10)
    axes.scatter(deg_cdiff_df['score'][top_genes], deg_cdiff_df['corr'][top_genes], s=10)
    texts = []
    for gene in top_genes:
        texts.append(axes.annotate(gene, (deg_cdiff_df['score'][gene], deg_cdiff_df['corr'][gene]), size=8, arrowprops=dict(arrowstyle = "-", color = "black")))
    adjust_text(texts)
    axes.set_xlabel('Difference scores')
    axes.set_ylabel('Correlation lambda vs ratio')
    axes.set_ylim(-1,1)
    # plt.show()
    plt.savefig(f'{out_dir}/cdiff_deg/{cluster}_corr_targCluster.png');plt.close('all')


from adjustText import adjust_text





for cluster in clusters:
    est_adata_tmp = est_adata.copy()
    est_adata_tmp.X = est_adata_tmp.layers["matrix"]
    sc.pp.normalize_total(est_adata_tmp)
    sc.pp.log1p(est_adata_tmp)
    cluster_adata = top_adata[top_adata.obs.cdiff_cluster == cluster]
    gene_scores = pd.Series(cluster_adata.layers['norm_cond_vel_diff'].mean(axis=0), index=cluster_adata.var_names)
    source_cells = cluster_adata.obs_names
    deg_cdiff_df = condiff.make_deg_cdiff_df(source_cells, est_adata_tmp, cluster, est_adata_tmp.obs[f'cdiff{cluster}_target'], gene_scores, q=0.95)
    sc.tl.rank_genes_groups(est_adata_tmp, 'sample', groups=['sample5'], reference='sample4')
    deg_cdiff_df = commons.extract_deg_df(est_adata_tmp, 'sample5')
    deg_cdiff_df['score'] = gene_scores[deg_cdiff_df.index]
    deg_cdiff_df = deg_cdiff_df.query('pvals_adj < 0.01')
    deg_cdiff_df = pd.merge(deg_cdiff_df, corrs, left_on="names", right_on="genes", how="inner")
    deg_cdiff_df = deg_cdiff_df.set_index("names")
    top_genes = deg_cdiff_df.sort_values('score', ascending=False).index[:10]
    bottom_genes = deg_cdiff_df.sort_values('score', ascending=True).index[:10]
    top_genes = np.concatenate([top_genes, bottom_genes])
    fig, axes = plt.subplots(1, 1, figsize=(6 * 1, 5 * 1))
    axes.scatter(deg_cdiff_df['score'], deg_cdiff_df['corr'], s=10)
    axes.scatter(deg_cdiff_df['score'][top_genes], deg_cdiff_df['corr'][top_genes], s=10)
    texts = []
    for gene in top_genes:
        texts.append(axes.annotate(gene, (deg_cdiff_df['score'][gene], deg_cdiff_df['corr'][gene]), 
                        size=8, arrowprops=dict(arrowstyle = "-", color = "black")))
    adjust_text(texts)
    axes.set_xlabel('Difference scores')
    axes.set_ylabel('Correlation lambda vs ratio')
    axes.set_ylim(-1,1)
    # plt.show()
    plt.savefig(f'{out_dir}/cdiff_deg/{cluster}_corr_total_q095.png');plt.close('all')





for cluster in clusters:
    est_adata_tmp = est_adata.copy()
    est_adata_tmp.X = est_adata_tmp.layers["matrix"]
    sc.pp.normalize_total(est_adata_tmp)
    sc.pp.log1p(est_adata_tmp)
    cluster_adata = top_adata[top_adata.obs.cdiff_cluster == cluster]
    gene_scores = pd.Series(cluster_adata.layers['norm_cond_vel_diff'].mean(axis=0), index=cluster_adata.var_names)
    source_cells = cluster_adata.obs_names
    deg_cdiff_df = condiff.make_deg_cdiff_df(source_cells, est_adata_tmp, cluster, est_adata_tmp.obs[f'cdiff{cluster}_target'], gene_scores, q=0.8)
    sc.tl.rank_genes_groups(est_adata_tmp, 'sample', groups=['sample5'], reference='sample4')
    deg_cdiff_df = commons.extract_deg_df(est_adata_tmp, 'sample5')
    deg_cdiff_df['score'] = gene_scores[deg_cdiff_df.index]
    deg_cdiff_df = deg_cdiff_df.query('pvals_adj < 0.01')
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
    axes.set_ylabel('cdiff difference mouse5/mouse4')
    # axes.set_ylim(-1,1)
    #plt.show()
    plt.savefig(f'{out_dir}/cdiff_deg/{cluster}_bar.png');plt.close('all')

## output for excel
for cluster in clusters:
    # cluster="2"
    est_adata_tmp = est_adata.copy()
    est_adata_tmp.X = est_adata_tmp.layers["matrix"]
    sc.pp.normalize_total(est_adata_tmp)
    sc.pp.log1p(est_adata_tmp)
    cluster_adata = top_adata[top_adata.obs.cdiff_cluster == cluster]
    gene_scores = pd.Series(cluster_adata.layers['norm_cond_vel_diff'].mean(axis=0), index=cluster_adata.var_names)
    source_cells = cluster_adata.obs_names
    deg_cdiff_df = condiff.make_deg_cdiff_df(source_cells, est_adata_tmp, cluster, est_adata_tmp.obs[f'cdiff{cluster}_target'], gene_scores, q=0.8)
    sc.tl.rank_genes_groups(est_adata_tmp, 'sample', groups=['sample5'], reference='sample4')
    deg_cdiff_df = commons.extract_deg_df(est_adata_tmp, 'sample5')
    deg_cdiff_df['score'] = gene_scores[deg_cdiff_df.index]
    deg_cdiff_df = deg_cdiff_df.query('pvals_adj < 0.01')
    # top_genes = deg_cdiff_df.sort_values('score', ascending=False).index[:10]
    # bottom_genes = deg_cdiff_df.sort_values('score', ascending=True).index[:10]
    # top_genes = np.concatenate([bottom_genes, top_genes[::-1]])
    deg_cdiff_df = deg_cdiff_df.sort_values("score", ascending=False).head(10)
    deg_cdiff_df.to_csv(f'{out_dir}/cdiff_deg/{cluster}_score.csv')





# os.environ['PYTHONHASHSEED'] = '0'


## cdiff2?????????????????????????????????
est_adata_tmp = est_adata.copy()
est_adata_tmp.X = est_adata_tmp.layers["matrix"]
sc.pp.normalize_total(est_adata_tmp)
sc.pp.log1p(est_adata_tmp)
sc.tl.pca(est_adata_tmp, svd_solver='arpack')
sc.pp.neighbors(est_adata_tmp, n_neighbors=10, n_pcs=40)
sc.tl.leiden(est_adata_tmp, key_added="fine_leiden", resolution=0.8)

sc.pl.umap(est_adata_tmp, color=['fine_leiden'], layer="lambda", s=30, show=False)
plt.savefig(f'{out_dir}/fine_leiden2.png');plt.close('all')


sc.pl.violin(est_adata_tmp, "cdiff2_target", groupby="fine_leiden", show=False)
plt.savefig(f'{out_dir}/cdiff2_target_violin2.png');plt.close('all')

sc.tl.rank_genes_groups(est_adata_tmp, 'fine_leiden', method='wilcoxon', key_added = "wilcoxon")
sc.pl.rank_genes_groups_matrixplot(est_adata_tmp, n_genes=5, key="wilcoxon", groupby="fine_leiden", show=False)
plt.savefig(f'{out_dir}/cdiff2_target_heatmap2.png');plt.close('all')

est_adata_tmp.obs["cdiff_cluster"] = "none"
est_adata_tmp.obs["cdiff_cluster"][top_adata.obs_names]= top_adata.obs["cdiff_cluster"]
sc.tl.rank_genes_groups(est_adata_tmp, 'cdiff_cluster', method='wilcoxon', key_added = "cdiff_wilcoxon")
sc.pl.rank_genes_groups_matrixplot(est_adata_tmp, n_genes=5, key="cdiff_wilcoxon", groupby="cdiff_cluster", show=False)
plt.savefig(f'{out_dir}/cdiff_heatmap.png');plt.close('all')


est_adata_tmp.write_h5ad(f'{out_dir}/est_adata_tmp.h5ad')
top_adata.write_h5ad(f'{out_dir}/top_adata.h5ad')
cluster_adata.write_h5ad(f'{out_dir}/cluster_adata.h5ad')
cdiff_adata.write_h5ad(f'{out_dir}/cdiff_adata.h5ad')




# impla_gmt = commons.parse_gmt('gmts/impala.gmt')
impla_gmt = commons.parse_gmt_500('gmts/impala.gmt')
# msig_gmt = commons.parse_gmt('gmts/msigdb.v7.4.symbols.gmt')
msig_gmt = commons.parse_gmt_500('gmts/msigdb.v7.4.symbols.gmt')
mouse2human = pd.read_csv("gmts/mouse2human.txt", sep="\t")

#cdiff_cl2
# keys = ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']
# deg_df = pd.DataFrame({
#     key: est_adata_tmp.uns['cdiff_wilcoxon'][key]["2"]
#     for key in keys
# }, index=est_adata_tmp.uns['cdiff_wilcoxon']['names']["2"])

# deg_df = pd.merge(deg_df, mouse2human, left_on="names", right_on="mouse")
# cl_genes = deg_df.query("scores > 10 & pvals_adj < 10**-4 ").human
# total_genes = pd.merge(pd.DataFrame(est_adata.var_names), mouse2human, left_on="Gene", right_on="mouse").human
# pval_df = commons.gene_set_enrichment(cl_genes, total_genes, impla_gmt)
# pval_df.to_csv(f'{out_dir}/enrich_impla_cdiff_cl2.csv')
# pval_df = commons.gene_set_enrichment(cl_genes, total_genes, msig_gmt)
# pval_df.to_csv(f'{out_dir}/enrich_msig_cdiff_cl2.csv')

# cdiff_cl2_genes = cl_genes.copy()

## pattern3 ??????cdiff2???cl5,6????????????DEG
adata_plt = est_adata_tmp.copy()
adata_plt.obs["cdiff2_and_fineClusters"] = adata_plt.obs["fine_leiden"]
adata_plt[adata_plt.obs["cdiff_cluster"]=="2"].obs["cdiff2_and_fineClusters"]= "cdiff2" 

i=1
for i in range(10):
    targ_fine_cluster = str(i)
    sc.tl.rank_genes_groups(adata_plt, 'cdiff2_and_fineClusters', groups=[targ_fine_cluster], references="cdiff2", method='wilcoxon', key_added = "cdiff2_and_fineClusters_wilcoxon")
    #sc.pl.rank_genes_groups(adata_plt, key="cdiff2_and_fineClusters_wilcoxon")
    deg_df = pd.DataFrame({
        key: adata_plt.uns['cdiff2_and_fineClusters_wilcoxon'][key][targ_fine_cluster]
        for key in keys
    }, index=adata_plt.uns['cdiff2_and_fineClusters_wilcoxon']['names'][targ_fine_cluster])
    deg_df = pd.merge(deg_df, mouse2human, left_on="names", right_on="mouse")
    cl_genes = deg_df.sort_values("scores", ascending=False).human[:200]
    deg_df.sort_values("scores", ascending=False).to_csv(f'{out_dir}/cl_genes_clu{targ_fine_cluster}.csv')
    # cl_genes = deg_df.query("scores > 5 & pvals_adj < 10**-4 ").human
    # cl_genes = deg_df.query("logfoldchanges > 1.5 & pvals_adj < 10**-4 ").human
    #import pdb; pdb.set_trace()
    pval_df = commons.gene_set_enrichment(cl_genes, deg_df.human, impla_gmt)
    pval_df.to_csv(f'{out_dir}/enrich_impla_direct_cl{targ_fine_cluster}.csv')
    pval_df = commons.gene_set_enrichment(cl_genes, deg_df.human, msig_gmt)
    pval_df.to_csv(f'{out_dir}/enrich_msig_direct_cl{targ_fine_cluster}_2.csv')
    ## fig from pattern3
    msig = pd.read_csv(f'{out_dir}/enrich_msig_direct_cl{targ_fine_cluster}_2.csv')
    msig.columns = ["gene_set", "padj"]
    df_bar = pd.DataFrame({
        "group":  [f'clu{targ_fine_cluster}' for i in range(len(msig))] ,
        "gene_set": list(msig.gene_set)  ,
        "padj": list(msig.padj)  ,
    })
    df_bar["-log10_padj"] = -np.log10(df_bar.padj)
    # df_bar = df_bar[(df_bar["gene_set"].str.contains("^GOBP")) & (df_bar["gene_set"].str.contains("IMMUN")) & (df_bar["padj"] < 0.01)]
    df_bar = df_bar[(df_bar["gene_set"].str.contains("^GOBP")) & (df_bar["padj"] < 0.01)]
    df_bar = df_bar.reset_index()
    idx = df_bar.index[(df_bar["gene_set"].str.contains("INNATE")) & (df_bar["gene_set"].str.contains("NEGATIVE")) ].tolist()
    cl = ["gray" for i in range(len(df_bar))]
    for i in idx:
        cl[i] ="c"
    df_bar["cl"] = cl
    # plt.figure(figsize=(15,40))
    sns.barplot(data=df_bar, y='gene_set', x='-log10_padj', palette=df_bar["cl"])
    plt.tick_params(labelsize=5)
    plt.subplots_adjust(left=0.7)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/cdiff2_direct_clu{targ_fine_cluster}_GO_2.png');plt.close('all')






