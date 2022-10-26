import vicdyf
import scvelo as scv
import scanpy as sc
from matplotlib import pyplot as plt
import os
import pandas as pd
import sys
from importlib import reload
sys.path.append("/mnt/244hirose/Scripts/")

import vicdyf_mod.src.vicdyf
reload(vicdyf_mod.src.vicdyf)
from vicdyf_mod.src.vicdyf import workflow as wf

wd = "/mnt/Donald/ito/220812"
os.makedirs(wd, exist_ok=True)
os.chdir(wd)


adata_1 = scv.read("../190918_Novaseq6000_1_Microglia70_Brain20_Immune10/velocyto/190918_Novaseq6000_1_Microglia70_Brain20_Immune10.loom", validate=False)
adata_2 = scv.read("../190918_Novaseq6000_2_Microglia70_Brain20_Immune10/velocyto/190918_Novaseq6000_2_Microglia70_Brain20_Immune10.loom", validate=False)
adata_3 = scv.read("../190918_Novaseq6000_3_Microglia10_Brain5_Immune70/velocyto/190918_Novaseq6000_3_Microglia10_Brain5_Immune70.loom", validate=False)
adata_4 = scv.read("../190918_Novaseq6000_4_Microglia15_Immune80_ManyMacrophages/velocyto/190918_Novaseq6000_4_Microglia15_Immune80_ManyMacrophages.loom", validate=False)
adata_5 = scv.read("../190918_Novaseq6000_5_Microglia15_Immune80_ManyMacrophages/velocyto/190918_Novaseq6000_5_Microglia15_Immune80_ManyMacrophages.loom", validate=False)
adata_1.var_names_make_unique()
adata_2.var_names_make_unique()
adata_3.var_names_make_unique()
adata_4.var_names_make_unique()
adata_5.var_names_make_unique()

adata = adata_1.concatenate([adata_2, adata_3, adata_4, adata_5], 
                            uns_merge="same", index_unique=None, batch_key="sample", 
                            batch_categories=["sample1", "sample2", "sample3", "sample4", "sample5"])
raw_adata = adata.copy()

scv.pp.filter_and_normalize(adata, min_shared_counts=20)#, n_top_genes=4000, retain_genes=adata.var_names)
# scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
sc.pp.scale(adata)

adata
# scv.tl.velocity(adata)
scv.pp.neighbors(adata)
# scv.tl.velocity_graph(adata)
scv.tl.umap(adata)
scv.tl.louvain(adata)
sc.tl.leiden(adata)
# scv.pl.velocity_embedding_stream(adata, basis='umap', color=["louvain"], save="velocity.pdf")
sc.pl.umap(adata)

adata_ref = sc.read_h5ad("/mnt/Daisy/sc_BRAIN_GSE129788/adata.h5ad")
var_names = adata_ref.var_names.intersection(adata.var_names)
adata_ref = adata_ref[:, var_names]
adata = adata[:, var_names]
sc.pl.umap(adata_ref, color='cluster')

kari_adata = sc.tl.ingest(adata, adata_ref, obs='cluster', inplace=False)
adata.obs = pd.DataFrame(adata.obs).join(pd.DataFrame(kari_adata.obs["cluster"]))
sc.pl.umap(adata, color='cluster', wspace=0.4)

adata.uns['cluster_colors'] = adata_ref.uns['cluster_colors']
sc.pl.umap(adata, color=['cluster'], wspace=0.5)
adata_concat = adata_ref.concatenate(adata, batch_categories=['ref', 'new'])
adata_concat.obs.cluster = adata_concat.obs.cluster.astype('category')
adata_concat.obs.cluster.cat.reorder_categories(adata_ref.obs.cluster.cat.categories, inplace=True)  # fix category ordering
adata_concat.uns['cluster_colors'] = adata_ref.uns['cluster_colors']  # fix category colors
sc.pl.umap(adata_concat, color=['batch', 'cluster'])

sc.tl.pca(adata_concat)
sc.external.pp.bbknn(adata_concat, batch_key='batch')
sc.tl.umap(adata_concat)
sc.pl.umap(adata_concat, color=['batch', 'cluster'])


adata_concat = adata_ref.concatenate(adata, batch_categories=['ref', 'new'])
sc.tl.pca(adata_concat)
sc.external.pp.bbknn(adata_concat, batch_key='batch') 
sc.tl.umap(adata_concat)
sc.pl.umap(adata_concat, color=['batch', 'cluster'])

collections.MutableSet = collections.abc.MutableSet
import scanorama

adatas = [ adata_ref, adata ]
corrected = scanorama.correct_scanpy(adatas, return_dimred=True)

corr_adata = sc.concat(
    corrected,
    label="dataset",
    keys=["this", "ref"],
    join="outer",
    uns_merge="first",
)
from sklearn.metrics.pairwise import cosine_distances

distances_anterior = 1 - cosine_distances(
    corr_adata[corr_adata.obs.dataset == "ref"].obsm[
        "X_scanorama"
    ],
    corr_adata[corr_adata.obs.dataset == "this"].obsm[
        "X_scanorama"
    ],
)
def label_transfer(dist, labels):
    lab = pd.get_dummies(labels).to_numpy().T
    class_prob = lab @ dist.T
    norm = np.linalg.norm(class_prob, 2, axis=0)
    class_prob = class_prob / norm
    class_prob = (class_prob.T - class_prob.min(1)) / class_prob.ptp(1)
    return class_prob

class_prob_anterior = label_transfer(distances_anterior, adata_ref.obs.cluster)

cp_anterior_df = pd.DataFrame(
    class_prob_anterior, columns=np.sort(adata_ref.obs.cluster.unique())
)
cp_anterior_df.index = adata.obs.index

adata_corr_transfer = adata.copy()
adata_corr_transfer.obs = pd.concat(
    [adata.obs, cp_anterior_df], axis=1
)

sc.pl.umap(adata_corr_transfer, color=['ABC', 'ARP', 'ASC', 'CPC', 'DC', 'EC', 'EPC', 'Hb_VC', 'HypEPC', 'ImmN', 'MAC', 'MG', 'MNC', 'NEUT', 'NRP', 'NSC', 'NendC', 'OEG', 'OLG', 'OPC', 'PC', 'TNC', 'VLMC', 'VSMC', 'mNEUR'], hspace=0.5, wspace=0.5)



## copy from DrKojima
'''
def transform_exp(adata):
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.raw = adata
    sc.pp.scale(adata)
    return adata

m_adata = adata
importlib.reload(commons)
m_adata = transform_exp(m_adata)
m_adata.var_names_make_unique()

# m_meta_df = pd.read_csv('exps/cd10sp/atac_integration_2022_04_26/signac_meta_df.csv', index_col=0)
# m_adata = m_adata[m_meta_df.index]

adata = sc.read_h5ad('/mnt/Daisy/sc_BRAIN_GSE129788/adata.h5ad')
sc.pp.scale(adata)

importlib.reload(commons)
# m_adata, adata = commons.ingets_integration(m_adata, adata)

q_adata = m_adata
r_adata = adata
import numpy as np
genes = np.intersect1d(q_adata.var_names, r_adata.var_names)
q_adata = q_adata[:, genes]
r_adata = r_adata[:, genes]
sc.pp.pca(r_adata)
sc.pp.neighbors(r_adata)
sc.tl.umap(r_adata)
sc.tl.ingest(q_adata, r_adata, 'cluster', embedding_method=('pca'))
reg = neighbors.KNeighborsRegressor().fit(r_adata.obsm['X_pca'], r_adata.obsm['X_umap'])
# q_adata.obsm['X_vicdyf_umap'] = reg.predict(q_adata.obsm['X_pca'])
q_adata.obsm['X_umap'] = reg.predict(q_adata.obsm['X_pca'])
q_adata.write_h5ad(f'{out_dir}/mult_trans_adata.h5ad')
q_adata = sc.read_h5ad(f'{out_dir}/mult_trans_adata.h5ad')
# m_meta_df.loc[m_adata.obs_names]
# common_cells = np.intersect1d(m_meta_df.index.astype(str), q_adata.obs_names)
common_cells = np.intersect1d(m_adata.obs_names, q_adata.obs_names)
q_adata = q_adata[common_cells]
q_adata.obs = pd.concat([q_adata.obs.loc[common_cells], m_adata.obs.loc[common_cells]], axis=1)
sc.pl.tsne(q_adata, color=['cluster'], use_raw=False)
plt.savefig(f'{out_dir}/q_adata_markers.png');plt.close()

q_adata.obsm
m_adata = adata
common_cells = np.intersect1d(m_adata.obs_names, q_adata.obs_names)
m_adata = m_adata[common_cells]
#m_adata.obs = pd.concat([m_adata.obs.loc[common_cells], m_meta_df.loc[common_cells]], axis=1)
m_adata = transform_exp(m_adata)
m_adata.var_names_make_unique()
# sc.pp.highly_variable_genes(m_adata)
sc.pp.pca(m_adata)
sc.pp.neighbors(m_adata)
sc.tl.umap(m_adata)
sc.pl.umap(m_adata, color=['cluster'], use_raw=False)
plt.savefig(f'{out_dir}/m_adata_markers.png');plt.close()

'''


















#https://www.rndsystems.com/resources/cell-markers/immune-cells/regulatory-t-cell/regulatory-t-cell-markers
marker_genes = ["CD3G", "CD4", "CD5", "IL2RA", "ENTPD1", "ITGAE", "TNFRSF18", "LAP3", "LRRC32", "TNFRSF4" ]
sc.pl.heatmap(adata, marker_genes, groupby="louvain", swap_axes=True, show_gene_labels=True, vmax=0.2, dendrogram=True)


adata.write("adata.h5ad")
# adata.write_loom('adata.loom')

adata.layers['spliced'] = adata.layers['spliced'].astype(int)
adata.layers['unspliced'] = adata.layers['unspliced'].astype(int)
# adata.layers['count'] = adata.layers['count'].astype(float)


reload(wf)
adata = wf.estimate_dynamics(adata,lr=0.0001)

adata = vicdyf.utils.change_visualization(adata, n_neighbors=30)
scv.pl.velocity_embedding_grid(adata,X=adata.obsm['X_vicdyf_umap'], V=adata.obsm['X_vicdyf_mdumap'], color='vicdyf_fluctuation', show=False, basis='X_vicdyf_umap', density=0.3)
plt.savefig('test_flow.png')

clust_adata = adata.copy()
clust_adata.X = adata.layers['vicdyf_fluctuation'].copy()
scv.tl.louvain(clust_adata, key_added="fluct_louvain")
scv.pl.velocity_embedding_grid(clust_adata,X=adata.obsm['X_vicdyf_umap'], V=adata.obsm['X_vicdyf_mdumap'], color='fluct_louvain', show=False, basis='X_vicdyf_umap', density=0.3, legend_loc="on data")
plt.savefig('test_flow_cluster.png')
plt.show()

scv.pl.velocity_embedding_grid(clust_adata,X=adata.obsm['X_vicdyf_umap'], V=adata.obsm['X_vicdyf_mdumap'], color='EPCAM', show=False, basis='X_vicdyf_umap', density=0.3, legend_loc="on data")
plt.savefig('test_flow_epcam.png')




bdata = adatas.copy()
bdata.layers['spliced'] = pd.DataFrame.sparse.from_spmatrix(raw_adata[:, adata.var_names].layers['spliced']).astype(float)
bdata.layers['unspliced'] = pd.DataFrame.sparse.from_spmatrix(raw_adata[:, adata.var_names].layers['unspliced']).astype(float)
bdata.layers['count'] = pd.DataFrame.sparse.from_spmatrix(raw_adata[:, adata.var_names].X).astype(float)
bdata.obs["fluc_cluster"] = clust_adata.obs.fluct_louvain
bdata.write_loom('adata.loom', write_obsm_varm=True)


