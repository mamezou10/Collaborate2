import anndata 
import torch
import scanpy as sc
import pandas as pd
import numpy as np
from .exp import VaeSmExperiment
from plotnine import *
import plotly.graph_objects as go
import plotly.io as pio
from .commons import make_edge_df
import matplotlib
from matplotlib import colors
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from matplotlib.offsetbox import AnchoredText
from matplotlib.patches import Patch
import subprocess
import statsmodels.api as sm
import copy
from scipy.sparse import coo_matrix
import datetime
now = datetime.datetime.now().strftime('%Y%m%d%H%M')

def safe_toarray(x):
    if type(x) != np.ndarray:
        return x.toarray()
    else:
        return x

def make_inputs(sc_adata, sp_adata, layer_name):
    x = torch.tensor(safe_toarray(sc_adata.layers[layer_name]))
    s = torch.tensor(safe_toarray(sp_adata.layers[layer_name]))
    if (x - x.int()).norm() > 0:
        try:
            raise ValueError('target layer of sc_adata should be raw count')
        except ValueError as e:
           print(e) 
    if (s - s.int()).norm() > 0:
        try:
            raise ValueError('target layer of sp_adata should be raw count')
        except ValueError as e:
           print(e) 
    return x, s

def optimize_deepcolor(vaesm_exp, lr, x_batch_size, s_batch_size, first_epoch, second_epoch):
    print(f'Loss: {vaesm_exp.evaluate()}')
    print('Start first opt')
    vaesm_exp.mode_change('sc')
    vaesm_exp.initialize_optimizer(lr)
    vaesm_exp.initialize_loader(x_batch_size, s_batch_size)
    vaesm_exp.train_total(first_epoch)
    print('Done first opt')
    print(f'Loss: {vaesm_exp.evaluate()}')
    print('Start second opt')
    vaesm_exp.mode_change('sp')
    vaesm_exp.initialize_optimizer(lr)
    vaesm_exp.initialize_loader(x_batch_size, s_batch_size)
    vaesm_exp.train_total(second_epoch)
    print('Done second opt')
    print(f'Loss: {vaesm_exp.evaluate()}')
    return vaesm_exp


def conduct_umap(adata, key):
    sc.pp.neighbors(adata, use_rep=key, n_neighbors=30)
    sc.tl.umap(adata)
    return(adata)


def extract_mapping_info(vaesm_exp, sc_adata, sp_adata):
    with torch.no_grad():
        xz, qxz, xld, p, sld, theta_x, theta_s = vaesm_exp.vaesm(vaesm_exp.xedm.x.to(vaesm_exp.device))
    sc_adata.obsm['X_zl'] = qxz.loc.detach().cpu().numpy()
    sc_adata.obsm['lambda'] = xld.detach().cpu().numpy()
    p_df = pd.DataFrame(p.detach().cpu().numpy().transpose(), index=sc_adata.obs_names, columns=sp_adata.obs_names)
    sc_adata.obsm['map2sp'] = p_df.values
    sp_adata.obsm['map2sc'] = p_df.transpose().values
    sc_adata.obsm['p_mat'] = sc_adata.obsm['map2sp'] / np.sum(sc_adata.obsm['map2sp'], axis=1).reshape((-1, 1))
    return sc_adata, sp_adata


def estimate_spatial_distribution(
        sc_adata, sp_adata, param_save_path, layer_name='count', first_epoch=500, second_epoch=500, lr=0.001, val_ratio=0.01, test_ratio=0.01, device=None, num_workers=1,
        x_batch_size=300, s_batch_size=300, 
        model_params = {
            "x_dim": 100,
            "s_dim": 100,
            "xz_dim": 10, "sz_dim": 10,
            "enc_z_h_dim": 50, "dec_z_h_dim": 50, "map_h_dim": 50,
            "num_enc_z_layers": 2, "num_dec_z_layers": 2
        }
    ):
    if device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # make data set 
    losss1=[]
    losss2=[]
    sp_adata.obs_names_make_unique()
    sc_adata.obs_names_make_unique()
    x, s = make_inputs(sc_adata, sp_adata, layer_name)
    model_params['x_dim'] = x.size()[1]
    model_params['s_dim'] = s.size()[1]
    vaesm_exp = VaeSmExperiment(model_params, lr, x, s, test_ratio, x_batch_size, s_batch_size, num_workers, validation_ratio=val_ratio, device=device)
    vaesm_exp = optimize_deepcolor(vaesm_exp, lr, x_batch_size, s_batch_size, first_epoch, second_epoch)
    torch.save(vaesm_exp.vaesm.state_dict(), param_save_path)
    sc_adata.uns['param_save_path'] = param_save_path
    sp_adata.uns['param_save_path'] = param_save_path
    sp_adata.uns['layer_name'] = layer_name
    sc_adata, sp_adata = extract_mapping_info(vaesm_exp, sc_adata, sp_adata)
    return sc_adata, sp_adata


def calculate_clusterwise_distribution(sc_adata, sp_adata, cluster_label):
    p_mat = sp_adata.obsm['map2sc']
    celltypes = np.sort(np.unique(sc_adata.obs[cluster_label].astype(str)))
    try:
        raise ValueError('some of cluster names in `cluster_label` is overlapped with `sp_adata.obs.columns`')
    except ValueError as e:
        print(e)
    cp_map_df = pd.DataFrame({
        celltype: np.sum(p_mat[:, sc_adata.obs[cluster_label] == celltype], axis=1)
        for celltype in celltypes}, index=sp_adata.obs_names)
    cp_map_df_max = cp_map_df.idxmax(axis=1)
    cp_map_df_max.name = "major_cluster_" + cluster_label
    sp_adata.obs = pd.concat([sp_adata.obs, cp_map_df, cp_map_df_max], axis=1)
    return sp_adata

def calculate_imputed_spatial_expression(sc_adata, sp_adata):
    sc_norm_mat = sc_adata.layers['count'].toarray() / np.sum(sc_adata.layers['count'].toarray(), axis=1).reshape((-1, 1))
    sp_adata.layers['imputed_exp'] = np.matmul(
        sp_adata.obsm['map2sc'], sc_norm_mat)
    return sp_adata

def estimate_colocalization(sc_adata):
    p_mat = sc_adata.obsm['p_mat']
    coloc_mat = p_mat @ p_mat.transpose()
    coloc_mat = np.log2(coloc_mat) + np.log2(p_mat.shape[1])
    sc_adata.obsp['colocalization'] = coloc_mat
    return sc_adata

def make_coloc_mat(sc_adata):
    p_mat = sc_adata.obsm['p_mat']
    coloc_mat = p_mat @ p_mat.transpose()
    coloc_mat = np.log2(coloc_mat) + np.log2(p_mat.shape[1])
    return coloc_mat


def make_high_coloc_index(sc_adata, celltype_label):
    coloc_mat = make_coloc_mat(sc_adata) 
    thresh = 1
    high_coloc_index = np.argwhere(coloc_mat > thresh)
    high_coloc_index = high_coloc_index[high_coloc_index[:, 0] < high_coloc_index[:, 1]]
    ocell1_types = sc_adata.obs[celltype_label].iloc[high_coloc_index[:, 0]].values
    ocell2_types = sc_adata.obs[celltype_label].iloc[high_coloc_index[:, 1]].values
    high_coloc_index = high_coloc_index[ocell1_types != ocell2_types]
    return high_coloc_index

def make_cell_umap_df(cell, edge_df, sc_adata):
    cell_adata = sc_adata[edge_df[cell]]
    cell_umap_df = pd.DataFrame(cell_adata.obsm['position'] * 0.9, columns=['X', 'Y'], index=edge_df.index)
    cell_umap_df['edge'] = edge_df.index
    return(cell_umap_df)

def make_edge_vis_df(sc_adata, celltype_label, total_edge_num, edge_thresh=1):
    orig_edge_df = make_edge_df(sc_adata, celltype_label, edge_thresh=edge_thresh)
    sub_edge_df = orig_edge_df.loc[np.random.choice(orig_edge_df.index, total_edge_num)]
    tot_edge_df = pd.concat([
        make_cell_umap_df(cell, sub_edge_df, sc_adata)
        for cell in ['cell1', 'cell2']], axis=0)
    return tot_edge_df

def visualize_colocalization_network(sc_adata, sp_adata, celltype_label, spatial_cluster, celltype_sample_num=500, total_edge_num=5000,  color_dict=None, edge_thresh=1):
    # resample celltypes
    sc_adata = sc_adata[sc_adata.obs.groupby(celltype_label).sample(celltype_sample_num, replace=True).index]
    sc_adata.obs_names_make_unique()
    # determine cell thetas and positions
    thetas = 2 * np.pi * np.arange(sc_adata.shape[0]) / sc_adata.shape[0]
    x = np.cos(thetas)
    y = np.sin(thetas)
    pos_mat = np.column_stack((x, y))
    sc_adata.obsm['position'] = pos_mat
    # map max leiden
    p_mat = sc_adata.obsm['p_mat']
    sc_adata.obs['max_map'] = sp_adata[p_mat.argmax(axis=1)].obs[spatial_cluster].astype(str).values
    # sample cell pairs based on colocalization
    tot_edge_df = make_edge_vis_df(sc_adata, celltype_label, total_edge_num, edge_thresh=edge_thresh)
    # visualize
    cells_df = pd.DataFrame({
        'X': sc_adata.obsm['position'][:, 0] * np.random.uniform(0.9, 1.1, size=sc_adata.shape[0]), 
        'Y': sc_adata.obsm['position'][:, 1] * np.random.uniform(0.9, 1.1, size=sc_adata.shape[0]), 
        'celltype': sc_adata.obs[celltype_label]})
    # determine even odds groups
    groups = sc_adata.obs[celltype_label].unique()
    gidxs = np.arange(groups.shape[0])
    even_groups = groups[gidxs % 2 == 0]
    odd_groups = groups[gidxs % 2 == 1]
    even_cells_df = cells_df.query('celltype in @even_groups')
    odd_cells_df = cells_df.query('celltype in @odd_groups')
    celltype_df = cells_df.groupby('celltype', as_index=False).mean()
    add_df = pd.DataFrame({
        'X': sc_adata.obsm['position'][:, 0] * np.random.uniform(1.1, 1.3, size=sc_adata.shape[0]), 
        'Y': sc_adata.obsm['position'][:, 1] * np.random.uniform(1.1, 1.3, size=sc_adata.shape[0]),
        'celltype': sc_adata.obs['max_map'].astype(str)})
    g = ggplot(add_df, aes(x='X', y='Y', color='celltype')) + geom_point(size=0.5)  +\
         geom_point(even_cells_df, size=0.1, color='#60C2CB') + \
          geom_point(odd_cells_df, size=0.1, color='#D05C54')  + \
          geom_line(tot_edge_df, aes(group='edge'), color='black', size=0.1, alpha=0.05) + \
              geom_text(celltype_df, aes(label='celltype'), color='black')  
    if not color_dict == None:
        g = g + scale_color_manual(color_dict)
    return g


# spcify top expression
def make_top_values(mat, top_fraction = 0.1, axis=0):
    top_mat = mat > np.quantile(mat, 1 - top_fraction, axis=axis, keepdims=True)
    return(top_mat)


def make_top_act_ligands(cell_type, coexp_count_df, topn=3):
    d = coexp_count_df.loc[cell_type].max(axis=0).sort_values(ascending=False)[:topn]
    return(d.index)


def make_coexp_cc_df(ligand_adata, edge_df, role):
    sender = edge_df.cell1 if role == "sender" else edge_df.cell2
    receiver = edge_df.cell2 if role == "sender" else edge_df.cell1
    coexp_df = pd.DataFrame(
        ligand_adata[sender].X *
        ligand_adata[receiver].layers['activity'],
        columns=ligand_adata.var_names, index=edge_df.index
    )
    coexp_df['cell2_type'] = edge_df['cell2_type']
    coexp_df['cell1_type'] = edge_df['cell1_type']
    coexp_cc_df = coexp_df.groupby(['cell2_type', 'cell1_type']).sum()
    coexp_cc_df = coexp_cc_df.reset_index().melt(id_vars=['cell1_type', 'cell2_type'], var_name='ligand', value_name='coactivity')
    return coexp_cc_df

def calculate_proximal_cell_communications(sc_adata, celltype_label, lt_df, 
                                           target_cellset_from, target_cellset_to=None, 
                                           celltype_sample_num=500, ntop_genes=4000, each_display_num=3, 
                                           role="sender", edge_thresh=1):
    # subsample data
    sc_adata = sc_adata[sc_adata.obs.groupby(celltype_label).sample(celltype_sample_num, replace=True).index]
    sc_adata.obs_names_make_unique()
    celltypes = sc_adata.obs.loc[:, celltype_label].unique()
    # make edge_df
    edge_df = make_edge_df(sc_adata, celltype_label, sub_sample=False, exclude_reverse=False, edge_thresh=edge_thresh)
    # select edge df with cell1 as target
    edge_df = edge_df.loc[edge_df.cell1_type.isin(target_cellset_from)]
    if target_cellset_to == None:
        edge_df = edge_df.loc[~edge_df.cell2_type.isin(target_cellset_from)]
    else:
        edge_df = edge_df.loc[edge_df.cell2_type.isin(target_cellset_to)]
    # select genes
    sc.pp.highly_variable_genes(sc_adata, n_top_genes=ntop_genes)
    sc_adata = sc_adata[:, sc_adata.var.highly_variable]
    common_genes = np.intersect1d(lt_df.index, sc_adata.var_names)
    lt_df = lt_df.loc[common_genes]
    sc_adata = sc_adata[:, common_genes]
    lt_df = lt_df.loc[:, lt_df.columns.isin(sc_adata.var_names)]
    # import pdb; pdb.set_trace()
    # make normalization
    ligands = lt_df.columns
    lt_df = lt_df.div(lt_df.sum(axis=0), axis=1)
    ligand_adata = sc_adata[:, ligands]
    top_exps = make_top_values(sc_adata.X.toarray(), axis=1, top_fraction=0.01)
    ligand_adata.layers['activity'] = make_top_values(top_exps @ lt_df)
    ligand_adata.X = make_top_values(safe_toarray(ligand_adata.X))
    # make base coexp df
    coexp_cc_df = make_coexp_cc_df(ligand_adata, edge_df, role)
    sub_coexp_cc_df = coexp_cc_df.sort_values('coactivity', ascending=False).groupby('cell2_type', as_index=False).head(n=each_display_num)
    # plotting configurrations
    tot_list = list(sub_coexp_cc_df.ligand.unique()) + list(celltypes)
    ligand_pos_dict = pd.Series({
        ligand: i
        for i, ligand in enumerate(sub_coexp_cc_df.ligand.unique())
    })
    celltype_pos_dict = pd.Series({
        celltype: i + sub_coexp_cc_df.ligand.unique().shape[0]
        for i, celltype in enumerate(celltypes)
    })
    senders = sub_coexp_cc_df.cell1_type.values if role == "sender" else sub_coexp_cc_df.cell2_type.values
    receivers = sub_coexp_cc_df.cell2_type.values if role == "sender" else sub_coexp_cc_df.cell1_type.values
    sources = pd.concat([ligand_pos_dict.loc[sub_coexp_cc_df.ligand.values], celltype_pos_dict.loc[senders]])
    targets = pd.concat([celltype_pos_dict.loc[receivers], ligand_pos_dict.loc[sub_coexp_cc_df.ligand.values]])
    values = pd.concat([sub_coexp_cc_df['coactivity'], sub_coexp_cc_df['coactivity']])
    labels = pd.concat([sub_coexp_cc_df['cell1_type'], sub_coexp_cc_df['cell1_type']])
    # colors = pd.Series(target_color_dict)[labels]
    fig = go.Figure(data=[go.Sankey(node=dict(label=tot_list),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            # color=colors,
            label=labels))])
    fig.update_layout(font_family="Courier New")
    return fig, coexp_cc_df







def make_dual_zl(sc_adata, high_coloc_index):
    dual_zl_add = sc_adata.obsm['X_zl'][high_coloc_index[:, 0]] + sc_adata.obsm['X_zl'][high_coloc_index[:, 1]]
    dual_zl_prod = sc_adata.obsm['X_zl'][high_coloc_index[:, 0]] * sc_adata.obsm['X_zl'][high_coloc_index[:, 1]]
    dual_zl_prod = np.sign(dual_zl_prod) * np.sqrt(np.abs(dual_zl_prod))
    dual_zl = np.concatenate([dual_zl_add, dual_zl_prod], axis=1)
    return dual_zl_add


def setup_dual_adata(dual_zl, sc_adata, high_coloc_index):
    dual_adata = anndata.AnnData(dual_zl)
    dual_adata.obsm['X_zl'] = dual_zl
    dual_adata.obs['cell1_celltype'] = sc_adata.obs["large_class"].values[high_coloc_index[:, 0]]
    dual_adata.obs['cell2_celltype'] = sc_adata.obs["large_class"].values[high_coloc_index[:, 1]]
    dual_adata.obs['cell1_obsname'] = sc_adata.obs_names[high_coloc_index[:, 0]]
    dual_adata.obs['cell2_obsname'] = sc_adata.obs_names[high_coloc_index[:, 1]]
    dual_adata.obs_names = dual_adata.obs['cell1_obsname'] + dual_adata.obs['cell2_obsname']
    cell_min = dual_adata.obs[['cell1_celltype', 'cell2_celltype']].astype(str).values.min(axis=1)
    cell_max = dual_adata.obs[['cell1_celltype', 'cell2_celltype']].astype(str).values.max(axis=1)
    dual_adata.obs['dual_celltype'] = cell_min + '/' + cell_max
    return dual_adata


def analyze_pair_cluster(sc_adata, sp_adata, cellset1, cellset2, celltype_label, max_pair_num=30000):
    contributions = np.sum(sc_adata.obsm['map2sp'], axis=1)
    sc_adata.obs['large_class'] = 'None'
    annot1 = ','.join(cellset1)
    annot2 = ','.join(cellset2)
    sc_adata.obs['large_class'][sc_adata.obs[celltype_label].isin(cellset1)] = annot1
    sc_adata.obs['large_class'][~sc_adata.obs[celltype_label].isin(cellset1)] = annot2
    sc_adata = sc_adata[sc_adata.obs['large_class'].isin([annot1, annot2])]
    high_coloc_index = make_high_coloc_index(sc_adata, "large_class")
    if high_coloc_index.shape[0] > max_pair_num:
        high_coloc_index = high_coloc_index[np.random.randint(0, high_coloc_index.shape[0], size=max_pair_num)]
    dual_zl = make_dual_zl(sc_adata, high_coloc_index)
    dual_adata = setup_dual_adata(dual_zl, sc_adata, high_coloc_index)
    dual_adata = conduct_umap(dual_adata, 'X_zl')
    sc.tl.leiden(dual_adata, resolution=0.1)
    dual_adata.uns['large_class1'] = annot1
    dual_adata.uns['large_class2'] = annot2
    return dual_adata

def calculate_norm_expression(sp_adata):
    sp_adata.layers['norm_counts'] = sp_adata.layers['count'].toarray() / np.sum(sp_adata.layers['count'].toarray(), axis=1).reshape(-1, 1)
    return sp_adata

def sp_ligand_activity(sp_adata, lt_df, top_fraction=0.01 ):        ## sp_ligand_activity represented in binary
    # sp_adata = sc.read_h5ad('../220311/sp_tmp_trained.h5ad')
    # lt_df = pd.read_csv('../220304/ligand_target_df.csv', index_col=0)
    common_genes = np.intersect1d(lt_df.index, sp_adata.var_names)
    lt_df = lt_df.loc[common_genes]
    sp_adata = sp_adata[:, common_genes]
    lt_df = lt_df.loc[:, lt_df.columns.isin(sp_adata.var_names)]
    # make normalization
    ligands = lt_df.columns
    lt_df = lt_df.div(lt_df.sum(axis=0), axis=1)
    ligand_adata = sp_adata[:, ligands]
    top_exps = make_top_values(sp_adata.X.toarray(), axis=1, top_fraction=top_fraction)
    ligand_adata.layers['activity'] = make_top_values(top_exps @ lt_df)
    ligand_adata.X = make_top_values(safe_toarray(ligand_adata.X))
    ligand_adata.layers["activity_n"] = ligand_adata.layers["activity"].astype(int)
    return ligand_adata

def plot_ligand_activity(ligand_adata, gene):
    fig, axs = plt.subplots(1, 2, figsize=(30, 10))
    sc.pl.spatial(ligand_adata, color=[gene], layer="imputed_exp", ax=axs[0], show=False) 
    sc.pl.spatial(ligand_adata, color=[gene], layer="activity_n", ax=axs[1], show=False) 
    plt.tight_layout()
    plt.show()
    #fig.savefig("img.png")
    return fig


def get_top_ligand_activity(sp_adata, sc_adata, lt_df, role, cluster="3", top_n=3, show=False):
    common_genes = np.intersect1d(lt_df.index, sp_adata.var_names)
    lt_df = lt_df.loc[common_genes]
    sp_adata = sp_adata[:, common_genes]
    lt_df = lt_df.loc[:, lt_df.columns.isin(sp_adata.var_names)]
    # make normalization
    ligands = lt_df.columns
    lt_df = lt_df.div(lt_df.sum(axis=0), axis=1)
    ligand_adata = sp_adata[:, ligands]
    top_exps = make_top_values(sp_adata.X.toarray(), axis=1, top_fraction=1)
    ligand_adata.layers['activity'] = make_top_values(top_exps @ lt_df)
    ligand_adata.X = make_top_values(safe_toarray(ligand_adata.X))
    nn, coexp_cc_df = calculate_proximal_cell_communications(sc_adata, 'leiden', lt_df, [cluster], celltype_sample_num=1000, ntop_genes=4000, each_display_num=3, role=role, edge_thresh=1)
    top_ligands = coexp_cc_df[coexp_cc_df.cell1_type==cluster].groupby("cell2_type")[["ligand", "coactivity"]].apply(lambda df: df.nlargest(top_n, "coactivity"))
    return top_ligands


def add_sp_ligand_activity(sp_adata, lt_df, ntop_genes = 4000, top_fraction = 0.01):  ## sp_ligand_activity represented in binary
    # sp_adata = sc.read_h5ad('../220311/sp_tmp_trained.h5ad')
    # sc_adata = sc.read_h5ad('../220311/sc_tmp_trained.h5ad')
    # sp_adata = calculate_imputed_spatial_expression(sc_adata, sp_adata)
    # lt_df = pd.read_csv('../220304/ligand_target_df.csv', index_col=0)
    #sc.pp.scale(sp_adata, max_value=10)
    sc.pp.highly_variable_genes(sp_adata, n_top_genes=ntop_genes)
    sp_adata = sp_adata[:, sp_adata.var.highly_variable]
    common_genes = np.intersect1d(lt_df.index, sp_adata.var_names)
    lt_df = lt_df.loc[common_genes,:]
    sp_adata = sp_adata[:, common_genes]
    lt_df = lt_df.loc[:, lt_df.columns.isin(sp_adata.var_names)]
    lt_df = lt_df * (lt_df > np.quantile(lt_df, 1 - top_fraction, axis=0, keepdims=True))
    lt_df = lt_df.div(lt_df.sum(axis=0), axis=1)

    exp_tmp = sp_adata.layers["imputed_exp"]
    exp_tmp = sp.stats.zscore(pd.DataFrame(exp_tmp))

    act = np.dot(exp_tmp,lt_df)
    act = ((act - act.min()) / (act.max() - act.min()))
    sp_adata = sp_adata[:, lt_df.columns]
    sp_adata.layers["sp_ligand_activity"] = act
    
    ligact_adata = sp_adata
    return ligact_adata


def plot_sp_ligand_activity(ligact_adata, gene = "gene", library_id = "library_id"):
    fig, axs = plt.subplots(1, 3, figsize=(15,5))
    sc.pl.spatial(ligact_adata, color=[gene], library_id=library_id, ax=axs[0], show=False)
    sc.pl.spatial(ligact_adata, color=[gene], library_id=library_id, layer='imputed_exp', vmax=0.0001, ax=axs[1], show=False)
    sc.pl.spatial(ligact_adata, color=[gene], library_id=library_id, layer='sp_ligand_activity', ax=axs[2], show=False)#
    fig.tight_layout()
    fig.show()
    fig.savefig(gene + "_ligand_activity_spatial.png")
    return fig


def plot_lig_correlation(ligact_adata, gene = "gene", plot=True):
    y = pd.DataFrame(ligact_adata[:,gene].layers["sp_ligand_activity"])
    x = pd.DataFrame(ligact_adata[:,gene].layers["imputed_exp"])
    # x = pd.DataFrame(ligact_adata[:,gene].X)
    df = pd.concat([x,y], axis=1)
    df.columns=["x","y"]
    r, p = sp.stats.pearsonr(x=df.x, y=df.y)
    g = sns.regplot(x=x, y=y, data=df)
    anc = AnchoredText("r: "+str(round(r,3))+", p: "+str(p), loc="upper left", frameon=False, prop=dict(size=15))
    g.axes.add_artist(anc)
    g.set_title(gene)
    if plot:
        plt.show()
    return g, r, p

def plot_lig_correlation2(ligact_adata, ga_coor, gene = "gene", sigma="sigma", plot=True):
    y = pd.DataFrame(ligact_adata[:,gene].layers["sp_ligand_activity"])
    x = pd.DataFrame(ga_coor)
    # x = pd.DataFrame(ligact_adata[:,gene].X)
    df = pd.concat([x,y], axis=1)
    df.columns=["x","y"]
    r, p = sp.stats.pearsonr(x=list(pd.DataFrame(ga_coor).iloc[:,0]), 
                            y=list(pd.DataFrame(ligact_adata[:,gene].layers["sp_ligand_activity"]).iloc[:,0]))
    g = sns.regplot(x=x, y=y, data=df)
    anc = AnchoredText("r: " +str(round(r,3)) + ",\n p: " +str(p)  +  ",\n sigma: " +sigma  , loc="upper left", frameon=False, prop=dict(size=15))
    g.axes.add_artist(anc)
    g.set_title(gene + " + diffusion")
    if plot:
        plt.show()
    return g, r, p

def get_lig_correlation(ligact_adata, genes):
    gene_list=[]
    r_list=[]
    p_list=[]
    for gene in genes:
        try:
            _, r, p = plot_lig_correlation(ligact_adata, gene = gene, plot=False)
            #import pdb; pdb.set_trace()
            gene_list.append(gene)
            r_list.append(r)
            p_list.append(p)  
        except KeyError:
            pass  
    res = pd.DataFrame(list(zip(gene_list, r_list, p_list)), columns = ['gene','r', 'p'])
    return res


def rank_genes_groups_df(adata, key='rank_genes_groups'):
    dd = []
    groupby = adata.uns['rank_genes_groups']['params']['groupby']
    for group in adata.obs[groupby].cat.categories:
        cols = []
        for col in adata.uns[key].keys():
            if col != 'params':
                   cols.append(pd.DataFrame(adata.uns[key][col][group], columns=[col]))
        df = pd.concat(cols,axis=1)
        df['group'] = group
        dd.append(df)
    rgg = pd.concat(dd)
    rgg['group'] = rgg['group'].astype('category')
    return rgg.set_index('group')


def annotate_geneset(marker_genes, pre="test", geneset_db="/home/hirose/Scripts/gmt/PanglaoDB_Augmented_2021.txt"):
    karis = list()
    # marker_genes = marker_genes[(marker_genes.pvals_adj < 10**-3) & (marker_genes.scores > 0)]
    # marker_genes = marker_genes[(marker_genes.pvals_adj < 10**-5) & (marker_genes.logfoldchanges > 0)]
    for i in np.unique(pd.DataFrame(marker_genes.index)):
        #import pdb; pdb.set_trace()
        try:
            marker_genes_ = marker_genes[(marker_genes.pvals_adj < 10**-5)].loc[str(i)].sort_values("scores", ascending=False).iloc[:100,:]
        except (ValueError, KeyError):
            continue
        num = marker_genes_.loc[str(i)].shape[0]
        gene = '\t'.join(marker_genes_.loc[str(i)]["names"].tolist())
        kari = "\t".join(["leiden"+str(i), str(num), gene]) 
        karis.append(kari)
    s = "\n".join(karis)
    with open(pre + "_marker.gmt", "w", encoding="utf-8") as f:
        f.write(s)
    command = [
        "Rscript", 
        "/home/hirose/Scripts/gmt/for_fisher.r", 
        "/home/hirose/Scripts/gmt/my_fisher.R", 
        pre + "_marker.gmt", geneset_db, pre, 
        "5"
    ]
    subprocess.run(command)
    enrichment = pd.read_table(pre + "_enrichment.tsv")
    anno_df = pd.DataFrame()
    for i in list(np.unique(marker_genes.index)):
        data = enrichment[enrichment.geneset=="leiden"+str(i)]['gene_ontology'].str.cat(enrichment[enrichment.geneset=="leiden"+str(i)]['minus_log10_pvalue'].round(2).astype(str), sep='; -log10(p)=')
        anno_df = pd.concat([anno_df, pd.DataFrame({"leiden"+str(i): data}).reset_index(drop=True)], axis=1)
    anno_df.to_csv(pre + "_anno.tsv", sep="\t")
    return anno_df

def enrichment_geneset(gene_list, pre="test", geneset_db="/home/hirose/Scripts/gmt/impala.gmt"):
    gene_list = np.unique(gene_list)
    num = len(gene_list)
    gene_list = '\t'.join(gene_list)
    
    s = "\t".join([pre, str(num), gene_list])
    with open(pre + "_marker.gmt", "w", encoding="utf-8") as f:
        f.write(s)
    command = [
        "Rscript", 
        "/home/hirose/Scripts/gmt/for_fisher.r", 
        "/home/hirose/Scripts/gmt/my_fisher.R", 
        pre + "_marker.gmt", geneset_db, pre, 
        "5"
    ]
    subprocess.run(command)
    enrichment = pd.read_table(pre + "_enrichment.tsv")









def make_heatmap(sc_adata, sp_adata, pre, type="logN", cluster="", vmax=None, vmin=None):
    # colors = [list(matplotlib.colors.cnames.values())[col] for col in range(4,90,4)]
    # color_key = dict(zip(sc_adata.obs.leiden.unique(), 
    #                     [colors[i] for i in range(len(sc_adata.obs.leiden.unique()))]))
    # color_key_list = sorted(color_key.items(), key=lambda x:x[0])
    # color_key.clear()
    # color_key.update(color_key_list)
    color_key = dict(zip([str(i) for i in list(range(13))],sc_adata.uns["leiden_colors"]) )
    clust_cols = sc_adata.obs.leiden.map(color_key)
    key_list = list(color_key.keys())
    for i in key_list:
        color_key["cluster "+str(i)] = color_key.pop(i)
    handles = [Patch(facecolor=color_key[name]) for name in color_key]
    
    if cluster=="":
        pmat = sc_adata.obsm['p_mat']
    else:
        spots = sp_adata.obs.major_cluster==cluster
        pmat = sc_adata.obsm['p_mat'][:,spots]   
         
    coloc_mat = pmat @ pmat.transpose()
    coloc_mat_L = np.log2(coloc_mat) + np.log2(pmat.shape[1])
    if type=="logN":
        sns.clustermap(coloc_mat_L, col_colors=[clust_cols], row_colors=[clust_cols], 
                       row_cluster=True, col_cluster=True, yticklabels=False, xticklabels=False, 
                       cmap="Reds", vmax=vmax, vmin=vmin)
    else:
        sns.clustermap(coloc_mat, col_colors=[clust_cols], row_colors=[clust_cols], 
                       row_cluster=True, col_cluster=True, yticklabels=False, xticklabels=False, 
                       cmap="Reds", vmax=0.004, vmin=vmin)
    #plt.figure()
    plt.legend(handles, color_key, #title='leiden',
            bbox_to_anchor=(1.1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.savefig(pre + "coloc_heatmap.png", bbox_inches='tight')
    plt.show()


def make_heatmap_normalized(sc_adata, sp_adata, pre, type="logN", cluster="", vmax=None, vmin=None):
    color_key = dict(zip([str(i) for i in list(range(13))],sc_adata.uns["leiden_colors"]) )
    clust_cols = sc_adata.obs.leiden.map(color_key)
    key_list = list(color_key.keys())
    for i in key_list:
        color_key["cluster "+str(i)] = color_key.pop(i)
    handles = [Patch(facecolor=color_key[name]) for name in color_key]
    
    if cluster=="":
        pmat = sc_adata.obsm['p_mat']
    else:
        spots = sp_adata.obs.major_cluster==cluster
        pmat = sc_adata.obsm['p_mat'][:,spots]   
         
    coloc_mat = pmat @ pmat.transpose()
    coloc_mat_L = np.log2(coloc_mat) + np.log2(pmat.shape[1])
    coloc_mat = coloc_mat/np.sum(coloc_mat, axis=1)
    coloc_mat_L = coloc_mat_L/np.sum(coloc_mat_L, axis=1)
    
    if type=="logN":
        sns.clustermap(coloc_mat_L, col_colors=[clust_cols], row_colors=[clust_cols], 
                       row_cluster=True, col_cluster=True, yticklabels=False, xticklabels=False, 
                       cmap="Reds", vmax=vmax, vmin=vmin)
    else:
        sns.clustermap(coloc_mat, col_colors=[clust_cols], row_colors=[clust_cols], 
                       row_cluster=True, col_cluster=True, yticklabels=False, xticklabels=False, 
                       cmap="Reds", vmax=vmax, vmin=vmin)
    plt.legend(handles, color_key, #title='leiden',
            bbox_to_anchor=(1.1, 1), bbox_transform=plt.gcf().transFigure, loc='upper right')
    plt.savefig(pre + "norm_coloc_heatmap.png", bbox_inches='tight')
    plt.show()


## 林先生から
def make_boxes(x, thres):
    epsilon = 1.0e-6
    thres = thres * (1.0 + epsilon)
    mins = x.min(axis = 0)
    maxs = x.max(axis = 0)
    nx, ny = [ int(diff) + 3 for diff in (maxs - mins) / thres ]
    # nx, ny, nz = [ int(diff) + 3 for diff in (maxs - mins) / thres ]
    boxes =  [ [ [] for _ in range(ny) ]
                     for _ in range(nx) ] 
                     #for _ in range(nx) ]
    for inode in range(x.shape[0]):
        ix, iy = [ int(diff) + 1 for diff in (x[inode] - mins) / thres ]
        # boxes[ix][iy][iz].append(inode)
        boxes[ix][iy].append(inode)
    return boxes, nx, ny#, nz

def make_edges_dist(adata, thres, gene):
    # kari = pd.concat([pd.DataFrame(adata.obsm["spatial"]),pd.DataFrame(adata[:,gene].X)], axis=1)
    kari = pd.concat([pd.DataFrame(adata.obsm["spatial"]),pd.DataFrame(adata[:,gene].layers["imputed_exp"])], axis=1)
    kari.columns = ["x","y","exp"]
    x = kari[["x","y"]].values
    exp = kari[["exp"]].values
    edges = [ [], [], [], [] ]
    boxes, nx, ny = make_boxes(x, thres)
    # boxes, nx, ny, nz = make_boxes(x, thres)
    for ix1 in range(1, nx - 1):
        for iy1 in range(1, ny - 1):
            ineighbors = np.array([   inode for ix2 in range(ix1 - 1, ix1 + 2)
                                            for iy2 in range(iy1 - 1, iy1 + 2)
                                            for inode in boxes[ix2][iy2] ])
            for inode1 in boxes[ix1][iy1]:
                inodes2 = ineighbors[ inode1 < ineighbors ]
                d = np.linalg.norm(x[inode1] - x[inodes2], axis = 1)
                inodes2 = inodes2[ d < thres ]
                edges[0] += [ inode1 ] * len(inodes2)
                edges[1] += list(inodes2)
                edges[2] += list(d[ d < thres ])
                edges[3] += list(exp[inodes2].flatten())
    return edges

def add_gaussian(edges, sig, adata):
    epsilon = 1.0e-6
    shape = adata.shape[0]
    row = np.array(edges[0])
    col = np.array(edges[1])
    data_d = np.array(edges[2])
    data_exp = np.array(edges[3])
    coo_exp = coo_matrix((data_exp, (row, col)), shape=(shape,shape))
    data_sigma = np.exp(-np.power(data_d, 2)/ (2 * (sig**2)))
    coo_sigma = coo_matrix((data_sigma, (row, col)), shape=(shape,shape))
    coo_gaussian = coo_exp.multiply(coo_sigma).sum(axis=1) /(coo_sigma.sum(axis=1) + epsilon)
    return coo_gaussian

def gaussian_corr_out(x, y, plot=True):
    X = sm.add_constant(x)
    model = sm.OLS(y, X)
    model = model.fit()
    # plt.scatter(x, y)
    # if plot:
    #     plt.show()
    model_params = np.array(model.params)
    return model.aic, model.bic, model_params[1], model_params[0]

def get_gaussian_aic(adata, genes, pre=now, thres=1000):
    gene_list=[]; thre_list=[]; sig_list=[]; aic_list=[]; bic_list=[]; a_list=[]; b_const_list=[]
    for thre in thres:
        sigs = list(range(5, thre, thre//10))
        for gene in genes:
            edges = make_edges_dist(adata=adata, thres=thre, gene=gene)    
            lig_act = adata[:,gene].layers["sp_ligand_activity"]
            #　no X
            x = [0 for _ in range(adata.shape[0])]
            aic, bic, a, b_const = gaussian_corr_out(x, lig_act, plot=False)
            gene_list.append(gene)
            sig_list.append("no X")
            aic_list.append(aic)
            bic_list.append(bic)
            thre_list.append(thre)
            a_list.append(a)
            b_const_list.append(b_const)
            # no gaussian
            x = pd.DataFrame(adata[:,gene].layers["imputed_exp"])
            aic, bic, a, b_const = gaussian_corr_out(x, lig_act, plot=False)
            gene_list.append(gene)
            sig_list.append("no gaussian")
            aic_list.append(aic)
            bic_list.append(bic)  
            thre_list.append(thre)  
            a_list.append(a)
            b_const_list.append(b_const)            
            # with gaussian
            for sig in sigs:
                print(sig)
                gau_expr = add_gaussian(edges=edges, sig=sig, adata=adata)
                aic, bic, a, b_const = gaussian_corr_out(gau_expr, lig_act, plot=False)
                gene_list.append(gene)
                sig_list.append(sig)
                aic_list.append(aic)
                bic_list.append(bic)
                thre_list.append(thre)
                a_list.append(a)
                b_const_list.append(b_const)
    sig_res = pd.DataFrame(list(zip(gene_list, thre_list, sig_list, aic_list, bic_list, a_list, b_const_list)), columns = ['gene','thre', 'sig', 'aic', 'bic', "a", "b_const"]) 
    sig_res.to_csv(pre + "_sig_res.tsv", sep="\t", index=False)
    return sig_res


def plot_coloc_spots(sp_adata, sc_adata, coloc_clusters=["0","1"]):
    fig, axs = plt.subplots(1, len(coloc_clusters), figsize=(15,5))
    i=0
    for coloc_cluster in coloc_clusters:
        mg = sc_adata[(sc_adata.obs["coloc_cluster"]==coloc_cluster) & ((sc_adata.obs["leiden"]=="1") | (sc_adata.obs["leiden"]=="7") | (sc_adata.obs["leiden"]=="10"))].obsm['map2sp'].sum(axis=0)
        ca = sc_adata[(sc_adata.obs["coloc_cluster"]==coloc_cluster) & (sc_adata.obs["leiden"]=="3") ].obsm['map2sp'].sum(axis=0).shape
        col_name = "coloc_cluster " + coloc_cluster
        sp_adata.obs[col_name] = mg*ca
        #sc.pl.spatial(sp_adata, color=col_name)
        sc.pl.spatial(sp_adata, color=col_name, ax=axs[i], show=False)
        i += 1
    fig.tight_layout()
    fig.show()
    fig.savefig("coloc_spatial.png")
    return fig



def annotate_cluster(sc_adata, cluster="coloc_cluster", pre="coloc"):
    anno_adata = sc_adata
    #import pdb; pdb.set_trace()
    sc.tl.rank_genes_groups(anno_adata, cluster, method="wilcoxon")
    # sc.pl.rank_genes_groups_heatmap(anno_adata, n_genes=3, show_gene_labels=True, 
    #                                 save="sc_cluster_heatmap.png")
    marker_genes = rank_genes_groups_df(anno_adata)
    anno_df = annotate_geneset(marker_genes, pre, geneset_db="/home/hirose/Scripts/gmt/impala.gmt")
    return anno_df

