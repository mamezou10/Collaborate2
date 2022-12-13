import copy
import enum
import itertools
import os
from re import A
from tkinter import N

data_dir = "/mnt/Donald/tsuji/mouse_sc/221130/"

out_dir = "/mnt/Donald/tsuji/mouse_sc/221207/"
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

adata = sc.read_h5ad(f'{data_dir}/adata_cellClass1_clustered.h5')

adata = adata[adata.obs["CellClass1_clusters"].isin(["0", "1", "2", "3", "6","9","10","12", "13"])].copy()

# importlib.reload(workflow)
importlib.reload(modules)

scv.pp.moments(adata, n_neighbors=300)
corrs = pd.Series(scv.utils.vcorrcoef(adata.layers['Ms'], adata.layers['Mu'], axis=0), index=adata.var_names)
# adata = sc.read_h5ad('/mnt/Donald/ito/220812/preprocessed_adata.h5ad')
top_corr_genes = corrs.index[corrs > 0.6]
# adata = sc.read_h5ad('/mnt/Donald/ito/220812/preprocessed_adata.h5ad')
adata.var['highly_variable'] = False
adata.var['highly_variable'][top_corr_genes] = True
model_params = {
            'x_dim': 100,
            'z_dim': 10,
            'enc_z_h_dim': 128, 'enc_d_h_dim': 128, 'dec_z_h_dim': 128,
            'num_enc_z_layers': 2, 'num_enc_d_layers': 2,
            'num_dec_z_layers': 2, 'use_ambient': False, 'use_vamp': False, 'no_d_kld': False, 'decreasing_temp': False, 'dec_temp_steps': 30000, 'loss_mode': 'poisson'
}
checkpoint_dirname = f'{out_dir}/checkpoint'
# adata.obs['sample'] =    np.array("mouse" + pd.Series(adata.obs_names).str.split("_", expand=True).iloc[:,2])
adata.obs['condition'] = adata.obs["sample"]
# importlib.reload(workflow)
# importlib.reload(commons)
# importlib.reload(utils)
# importlib.reload(modules)
# importlib.reload(envdyn)
est_adata, lit_envdyn = workflow.conduct_cvicdyf_inference(adata, model_params, checkpoint_dirname, batch_size=50, two_step=True, dyn_mode=False, epoch=1000, patience=20, module=modules.Cvicdyf)




sc.pp.normalize_total(est_adata)
sc.pp.log1p(est_adata)
est_adata.write_h5ad(f'{out_dir}/est_adata.h5ad')
torch.save(lit_envdyn.state_dict, f'{out_dir}/lit_envdyn.pt')
importlib.reload(utils)
utils.plot_mean_flow(est_adata, cluster_key='CellClass1_clusters')
plt.savefig(f'{out_dir}/mean_vel_flow.png');plt.close('all')


conds = np.unique(est_adata.obs['condition'].unique())
cond1 = 'Day7' 
cond2 = 'KO_Day7' 
importlib.reload(commons)
cdiff_adata = commons.estimate_two_cond_dynamics(est_adata, cond1, cond2, lit_envdyn)
cdiff_adata.write_h5ad(f'{out_dir}/cdiff_adata.h5ad')

grid_df = commons.select_grid_values(est_adata.obsm['X_vicdyf_umap'], 30, est_adata.obs['norm_vicdyf_mean_velocity'])
commons.plot_vicdyf_umap(est_adata[grid_df.num_idx.values], 'norm_vicdyf_mean_velocity', show=False)
plt.savefig(f'{out_dir}/gird_umap_vel.png');plt.close('all')

grid_df = commons.select_grid_values(est_adata.obsm['X_vicdyf_umap'], 30, est_adata.obs['norm_vicdyf_fluctuation'])
commons.plot_vicdyf_umap(est_adata[grid_df.num_idx.values], 'norm_vicdyf_fluctuation', show=False)
plt.savefig(f'{out_dir}/gird_umap_fluc.png');plt.close('all')

commons.plot_vicdyf_umap(est_adata[grid_df.num_idx.values], 'CellClass1_clusters', show=False)
plt.savefig(f'{out_dir}/gird_umap_cluster.png');plt.close('all')

commons.visuazlie_condiff_dynamics(est_adata, cdiff_adata, cond1, cond2, f'{out_dir}/condiff_flows1.png', plot_num=30, plot_top_prop=0.8)


importlib.reload(commons)
commons.visuazlie_condiff_dynamics_ones(est_adata, cdiff_adata, cond1, cond2, f'{out_dir}/condiff_flows_ones.png', plot_num=30, plot_top_prop=0.8)






commons.plot_vicdyf_umap(est_adata, ['condition', 'CellClass1_clusters'], show=False)
plt.savefig(f'{out_dir}/condition_umap.png');plt.close('all')

pos_cdiff_mat = copy.deepcopy(cdiff_adata.layers['norm_cond_vel_diff'])
pos_cdiff_mat[pos_cdiff_mat < 0] = 0
pos_cdiff_mat
pos_vel_diffs = pd.Series(np.linalg.norm(pos_cdiff_mat, axis=0), index=cdiff_adata.var_names).sort_values(ascending=False)
neg_cdiff_mat = copy.deepcopy(cdiff_adata.layers['norm_cond_vel_diff'])
neg_cdiff_mat[neg_cdiff_mat > 0] = 0
neg_vel_diffs = pd.Series(np.linalg.norm(neg_cdiff_mat, axis=0), index=cdiff_adata.var_names).sort_values(ascending=False)
commons.plot_vicdyf_umap(est_adata, pos_vel_diffs.iloc[:10].index, show=False)
plt.savefig(f'{out_dir}/exp_pos_tops_exp.png');plt.close('all')

commons.plot_vicdyf_umap(est_adata, neg_vel_diffs.iloc[:10].index,show=False)
plt.savefig(f'{out_dir}/exp_neg_tops_exp.png');plt.close('all')
for cond in [cond1, cond2]:
    commons.plot_vicdyf_umap(cdiff_adata, pos_vel_diffs.iloc[:10].index, layer_name=f'cond_vel_{cond}',show=False)
    plt.savefig(f'{out_dir}/cond_vel_pos_tops_{cond}.png');plt.close('all')
    commons.plot_vicdyf_umap(cdiff_adata, neg_vel_diffs.iloc[:10].index, layer_name=f'cond_vel_{cond}',show=False)
    plt.savefig(f'{out_dir}/cond_vel_neg_tops_{cond}.png');plt.close('all')

commons.plot_vicdyf_umap(cdiff_adata, pos_vel_diffs.iloc[:10].index, layer_name=f'norm_cond_vel_diff')
plt.savefig(f'{out_dir}/cond_vel_pos_tops_cdiff.png');plt.close('all')






importlib.reload(hiroseflow)
# annotationする
sc.tl.rank_genes_groups(jac_adata, "leiden", method="wilcoxon")
marker_genes = hiroseflow.rank_genes_groups_df(jac_adata)
marker_genes["names"] = [i.upper() for i in marker_genes.names]
anno_df = hiroseflow.annotate_geneset(marker_genes, "leiden_anno", geneset_db="/mnt/244hirose/Scripts/gmts/PanglaoDB_Augmented_2021.txt")

