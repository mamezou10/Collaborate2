import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.neighbors import NearestNeighbors


# read base expression
proj_dir = '/Users/kojimayasuhiro/Projects/envdynamics/'
sample_dir = proj_dir + 'data/immune_celldiff/'
input_dir = sample_dir + 'abci/'
shima_dir = sample_dir + 'from_shimamura/'


def initialize_vicdfy_adata(adata, envdyn_exp, orig_dir):
    ld_mat_fname = orig_dir + 'ld_mat'
    ld_mat = np.loadtxt(ld_mat_fname)
    z_mat_fname = orig_dir + 'z_mat'
    z_mat = np.loadtxt(z_mat_fname)
    zl_mat_fname = orig_dir + 'zl_mat'
    zl_mat = np.loadtxt(zl_mat_fname)
    z_embed_fname = orig_dir + 'z_embed'
    z_embed = np.loadtxt(z_embed_fname)
    # gene_vel_fname = orig_dir + 'gene_vel'
    # gene_vel = np.loadtxt(gene_vel_fname)
    mean_gene_vel_fname = orig_dir + 'mean_gene_vel'
    mean_gene_vel = np.loadtxt(mean_gene_vel_fname)
    batch_std_mat_fname = orig_dir + 'batch_std_mat'
    batch_std_mat = np.loadtxt(batch_std_mat_fname)
    mean_d_embed_fname = orig_dir + 'mean_d_embed'
    mean_d_embed = np.loadtxt(mean_d_embed_fname)
    stoc_d_embed_fname = orig_dir + 'stoc_d_embed'
    stoc_d_embed = np.loadtxt(stoc_d_embed_fname)
    adata = adata[:, adata.var.highly_variable]
    adata = adata[:, np.sum(adata.layers['spliced'], axis=0) > 50]
    adata.layers['lambda'] = ld_mat
    # adata.layers['vicdyf_velocity'] = gene_vel
    adata.layers['vicdyf_mean_velocity'] = mean_gene_vel
    adata.layers['vicdyf_fluctuation'] = batch_std_mat
    adata.obsm['X_vicdyf_z'] = z_mat
    adata.obsm['X_vicdyf_zl'] = zl_mat
    adata.obsm['X_vicdyf_umap'] = z_embed
    return(adata)
    

def obj_path(analysis_id, obj_name):
    path = sample_dir + f'analysis/{analysis_id}/{obj_name}'
    return(path)


def plot_scatter_with_marker_txt(x, y, genes, markers, ax):
    x = pd.Series(x, index=genes)
    y = pd.Series(y, index=genes)
    ax.scatter(x, y, s=5)
    for gene in markers:
        txt = f'{gene},PC1@{x.rank(ascending=False)},PC2@{y.rank(ascending=False)}'
        ax.text(x[gene], y[gene], gene)
    return(ax)

def trancate_ext_val(vec, q=0.01):
    low_val = np.quantile(vec, q)
    high_val = np.quantile(vec, 1 - q)
    vec[vec < low_val] = low_val
    vec[vec > high_val] = high_val
    return(vec)


def calculate_neighbor_ratio(X, val_vec, nn=30):
    nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    val_mat = val_vec[indices]
    val_ratio_vec = val_mat.mean(axis=1)
    return(val_ratio_vec)
