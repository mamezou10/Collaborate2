import copy
import enum
import itertools
import os
from re import A
from tkinter import N
out_dir = '/mnt/Donald/ito/220829_2_mf'
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
import numpy as np
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


est_adata_tmp = sc.read_h5ad('/mnt/Donald/ito/220822_mf/est_adata_tmp.h5ad')


## pattern3 直接cdiff2とcl5, 6を比べてDEG
# est_adata_tmp.obs["cdiff2_and_fineClusters"] = est_adata_tmp.obs["fine_leiden"]
# est_adata_tmp[est_adata_tmp.obs["cdiff_cluster"]=="2"].obs["cdiff2_and_fineClusters"]= "cdiff2" 

# est_adata_tmp.uns['log1p']["base"] = None
# sc.tl.rank_genes_groups(est_adata_tmp, 'cdiff2_and_fineClusters', groups=["5", "6"], references="cdiff2", method='wilcoxon', key_added = "cdiff2_and_fineClusters_wilcoxon")

# sc.pl.rank_genes_groups(est_adata_tmp, key="cdiff2_and_fineClusters_wilcoxon")

impla_gmt = commons.parse_gmt('gmts/impala.gmt')
msig_gmt = commons.parse_gmt('gmts/msigdb.v7.4.symbols.gmt')
gobp_gmt = commons.parse_gmt('gmts/c5.go.bp.v7.5.1.symbols.gmt')
mouse2human = pd.read_csv("gmts/mouse2human.txt", sep="\t")
total_genes = pd.merge(pd.DataFrame(est_adata_tmp.var_names), mouse2human, left_on="Gene", right_on="mouse").human


#cl6
keys = ['names', 'scores', 'pvals', 'pvals_adj', 'logfoldchanges']
deg_df = pd.DataFrame({
    key: est_adata_tmp.uns['cdiff2_and_fineClusters_wilcoxon'][key]["6"]
    for key in keys
}, index=est_adata_tmp.uns['cdiff2_and_fineClusters_wilcoxon']['names']["6"])

deg_df = pd.merge(deg_df, mouse2human, left_on="names", right_on="mouse")
cl_genes = deg_df.query("scores > 8 & pvals_adj < 10**-4 ").human
pval_df = commons.gene_set_enrichment(cl_genes, deg_df.human, impla_gmt)
pval_df.to_csv(f'{out_dir}/enrich_impla_direct_cl6.csv')
pval_df = commons.gene_set_enrichment(cl_genes, deg_df.human, msig_gmt)
pval_df.to_csv(f'{out_dir}/enrich_msig_direct_cl6_2.csv')


#cl5
deg_df = pd.DataFrame({
    key: est_adata_tmp.uns['cdiff2_and_fineClusters_wilcoxon'][key]["5"]
    for key in keys
}, index=est_adata_tmp.uns['cdiff2_and_fineClusters_wilcoxon']['names']["5"])

deg_df = pd.merge(deg_df, mouse2human, left_on="names", right_on="mouse")
cl_genes = deg_df.query("scores > 8 & pvals_adj < 10**-4 ").human
pval_df = commons.gene_set_enrichment(cl_genes, deg_df.human, impla_gmt)
pval_df.to_csv(f'{out_dir}/enrich_impla_direct_cl5.csv')
pval_df = commons.gene_set_enrichment(cl_genes, deg_df.human, msig_gmt)
pval_df.to_csv(f'{out_dir}/enrich_msig_direct_cl5_2.csv')



import seaborn as sns



## fig from pattern3 ver2
msig5 = pd.read_csv(f'{out_dir}/enrich_msig_direct_cl5_2.csv')
msig6 = pd.read_csv(f'{out_dir}/enrich_msig_direct_cl6_2.csv')
msig5.columns = ["gene_set", "padj"]
msig6.columns = ["gene_set", "padj"]

df_bar = pd.DataFrame({
    "group":  ["clu5" for i in range(len(msig5))] + ["clu6" for i in range(len(msig6))],
    "gene_set": list(msig5.gene_set) + list(msig6.gene_set) ,
    "padj": list(msig5.padj) + list(msig6.padj) ,
})
df_bar["-log10_padj"] = -np.log10(df_bar.padj)
df_bar = df_bar[(df_bar["gene_set"].str.contains("^GOBP")) & (df_bar["padj"] < 0.05)]
df_bar = df_bar.reset_index()
idx = df_bar.index[(df_bar["gene_set"].str.contains("INNATE"))].tolist()
cl = ["gray" for i in range(len(df_bar))]
for i in idx:
    cl[i] ="c"

df_bar["cl"] = cl

plt.figure(figsize=(15,5))
sns.barplot(data=df_bar[df_bar["group"]=="clu5"], y='gene_set', x='-log10_padj', palette=df_bar[df_bar["group"]=="clu5"]["cl"])
plt.tick_params(labelsize=10)
plt.subplots_adjust(left=0.7)
plt.savefig(f'{out_dir}/cdiff2_direct_clu5_GO_2.png');plt.close('all')

plt.figure(figsize=(15,30))
sns.barplot(data=df_bar[df_bar["group"]=="clu6"], y='gene_set', x='-log10_padj', palette=df_bar[df_bar["group"]=="clu6"]["cl"])
plt.tick_params(labelsize=10)
plt.subplots_adjust(left=0.7)
plt.savefig(f'{out_dir}/cdiff2_direct_clu6_GO_2.png');plt.close('all')



sc.pl.violin(est_adata_tmp, "cdiff2_target", groupby="fine_leiden", show=True)










