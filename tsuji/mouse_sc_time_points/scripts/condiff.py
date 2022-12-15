import torch
import scanpy as sc
import commons, workflow
import pandas as pd
import numpy as np

def make_diff_mat(cluster_adata, tot_adata):
    max_vec = tot_adata.layers['lambda'].max(axis=0)
    norm_ld_mat = tot_adata.layers['lambda'] / max_vec
    mean_lds = (cluster_adata.layers['lambda'] / max_vec).mean(axis=0)
    diff_mat = (norm_ld_mat - mean_lds) / np.linalg.norm(norm_ld_mat - mean_lds, axis=1, keepdims=True)
    return diff_mat

def calculate_cdiff_target_scores(cluster_adata, tot_adata, cdiff_key='norm_cond_vel_diff'):
    max_vec = tot_adata.layers['lambda'].max(axis=0)
    cdiff_vec = cluster_adata.layers[cdiff_key].mean(axis=0) 
    cdiff_vec = cdiff_vec / np.linalg.norm(cdiff_vec)
    diff_mat = make_diff_mat(cluster_adata, tot_adata)
    cdiff_target_scores = pd.Series(diff_mat @ cdiff_vec, index=tot_adata.obs_names)
    return cdiff_target_scores
    

def make_deg_cdiff_df(source_cells, jac_adata, cluster, target_scores, gene_scores, q=0.8, method='wilcoxon'):
    top_diff_cells = target_scores.index[target_scores > target_scores.quantile(q)]
    jac_adata.obs['diff_pop'] = 'None'
    jac_adata.obs['diff_pop'][source_cells] = 'Source'
    jac_adata.obs['diff_pop'][top_diff_cells] = 'Target'
    sc.tl.rank_genes_groups(jac_adata, 'diff_pop', groups=['Target'], reference='Source', method=method)
    #import pdb; pdb.set_trace()
    deg_df = commons.extract_deg_df(jac_adata, 'Target')
    deg_df['score'] = gene_scores[np.intersect1d(deg_df.index, gene_scores.index)]
    return deg_df, jac_adata


def make_specifc_cond_tensor(adata, cond):
    c = torch.zeros_like(torch.tensor(adata.obsm['condition'].values)).float()
    c[:, adata.obsm['condition'].columns == cond] = 1.0
    return c


@torch.no_grad()
def estimate_cond_dynamics(adata, cond, model, condition_key='condition', batch_key='sample', cond_in_obsm=False):
    orig_device = model.device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    s, u, snorm_mat, unorm_mat, b, t, adata = workflow.make_datasets(adata, condition_key=condition_key, batch_key=batch_key, cond_in_obsm=cond_in_obsm)
    one_t = make_specifc_cond_tensor(adata, cond).to(device)
    batch = (s.to(device), u.to(device), snorm_mat.to(device), unorm_mat.to(device), b.to(device), one_t.to(device))
    z, qz = model.encode_z(batch)
    d, qd = model.encode_d(qz.loc, batch)
    diff_px_zd_ld = model.calculate_diff_x_grad(qz.loc, qd.loc, batch)
    model.to(orig_device)
    return diff_px_zd_ld.detach().cpu().numpy(), qd.loc.detach().cpu().numpy(), qd.scale.detach().cpu().numpy()


@torch.no_grad()
def estimate_stochastic_condiff(adata, cond1, cond2, model, condition_key='condition', batch_key='sample', cond_in_obsm=False):
    orig_device = model.device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    b = torch.tensor(adata.obsm['batch'].values).float()
    t = torch.tensor(adata.obsm['condition'].values).float()
    s = torch.tensor(adata.layers['spliced'].toarray())
    u = torch.tensor(adata.layers['unspliced'].toarray())
    snorm_mat = s
    unorm_mat = u
    t1 = make_specifc_cond_tensor(adata, cond1).to(device)
    t2 = make_specifc_cond_tensor(adata, cond2).to(device)
    batch1 = (s.to(device), u.to(device), snorm_mat.to(device), unorm_mat.to(device), b.to(device), t1.to(device))
    batch2 = (s.to(device), u.to(device), snorm_mat.to(device), unorm_mat.to(device), b.to(device), t2.to(device))
    z, qz = model.encode_z(batch1)
    d1, qd1 = model.encode_d(z, batch1)
    d2, qd2 = model.encode_d(z, batch2)
    v1 = model.calculate_diff_x_grad(z, d1, batch1)
    v2 = model.calculate_diff_x_grad(z, d2, batch1)
    diff_v = v2 - v1
    model.to(orig_device)
    return diff_v.detach().cpu().numpy()


def calculate_cdiff_bf(adata, cond1, cond2, model, n=10, condition_key='condition', batch_key='sample', cond_in_obsm=False, cellwise=False):
    if cellwise:
        aggregate = np.stack
    else:
        aggregate = np.concatenate
    diff_v = aggregate([
        estimate_stochastic_condiff(adata, cond1, cond2, model, condition_key=condition_key, batch_key=batch_key, cond_in_obsm=cond_in_obsm)
        for _ in range(n)], axis=0)
    eps = 1.0e-16
    p = (diff_v > 0).mean(axis=0) + 1.0e-16
    bfs = np.log2((p + eps) / (1 - p + eps))
    if not cellwise:
        bfs = pd.Series(bfs, index=adata.var_names)
    return bfs


def estimate_two_cond_dynamics(adata, cond1, cond2, lit_envdyn):
    sub_adata = adata
    for cond in [cond1, cond2]:
        cond_vel, cond_d, connd_dscale = estimate_cond_dynamics(sub_adata, cond, lit_envdyn)
        cond_dumap = commons.calc_int_dembed(sub_adata.obsm['X_vicdyf_umap'], sub_adata.obsm['X_vicdyf_zl'], cond_d, 1)
        sub_adata.obsm[f'cond_d_{cond}'] = cond_d
        sub_adata.obsm[f'cond_dscale_{cond}'] = connd_dscale
        sub_adata.obsm[f'cond_dumap_{cond}'] = cond_dumap
        sub_adata.layers[f'cond_vel_{cond}'] = cond_vel
    sub_adata.layers['cond_vel_diff'] = sub_adata.layers[f'cond_vel_{cond2}'] -  sub_adata.layers[f'cond_vel_{cond1}']
    max_vec = np.max(sub_adata.layers['lambda'], axis=0)
    sub_adata = norm_condiff(sub_adata, cond1, cond2, max_vec)
    return sub_adata


def norm_condiff(adata, cond1, cond2, norm_vec):
    layers = [f'cond_vel_{cond}' for cond in [cond1, cond2]] + ['cond_vel_diff']
    for layer in layers:
        adata.layers[f'norm_{layer}'] = adata.layers[layer] / norm_vec
    return adata

def condiff_clustering(adata, q, res):
    top_adata = adata[adata.obs.total_condiff > adata.obs.total_condiff.quantile(q)]
    sc.pp.neighbors(top_adata, use_rep='X_vicdyf_zl')
    sc.tl.leiden(top_adata, key_added='cdiff_cluster', resolution=res)
    return top_adata


def visualize_population_dyndiff(adata, cells, conds, ax, scale=5, width=0.0125):
    color_dict = {conds[0]: '#1F72AA', conds[1]: '#EF7D21'}
    top_adata = adata[cells]
    u_mean = top_adata.obsm['X_umap'].mean(axis=0)
    du_mean_dict = {}
    for cond in conds:
        du_mean_dict[cond] = top_adata.obsm[f'cond_dumap_{cond}'].mean(axis=0)
    for cond in conds:
        du_mean = du_mean_dict[cond]
        color = color_dict[cond]
        ax.quiver(np.array(u_mean[0]), np.array(u_mean[1]), np.array(du_mean[0]), np.array(du_mean[1]), color=color, label=cond, scale=scale, width=width, alpha=0.5)
