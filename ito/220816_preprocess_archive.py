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

wd = "/mnt/Donald/ito/220816"
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
sc.tl.leiden(adata, resolution=0.5, key_added="coarse_leiden")
# scv.pl.velocity_embedding_stream(adata, basis='umap', color=["louvain"], save="velocity.pdf")
sc.pl.umap(adata)
from sklearn.preprocessing import LabelEncoder

adata.write_h5ad("adata.h5ad")


adata = sc.read_h5ad("adata.h5ad")

adata_ref = sc.read_h5ad("/mnt/Daisy/sc_BRAIN_GSE128855/adata_wholebrain.h5ad")
var_names = adata_ref.var_names.intersection(adata.var_names)
adata_ref = adata_ref[:, var_names]
adata = adata[:, var_names]
sc.pl.umap(adata_ref, color='cluster', save="ref.pdf")

class_le = LabelEncoder()
adata_ref.obs["cluster_num"] = class_le.fit_transform(adata_ref.obs['cluster'])

kari_adata = sc.tl.ingest(adata, adata_ref, obs='cluster_num', inplace=False)

adata.obs["cluster_num"] = kari_adata.obs["cluster_num"]
adata.obs["cluster_num_inv"] = class_le.inverse_transform(adata.obs['cluster_num'])

sc.pl.umap(adata, color='cluster_num_inv', wspace=0.4, save="total.pdf")
sc.pl.umap(adata, color=['Tmem119', 'P2ry12'], wspace=0.4, save="microglia.pdf")
sc.pl.umap(adata, color=['Nes','Hes1', 'Notch1'], wspace=0.4, save="NSC.pdf", vmax=5)
sc.pl.umap(adata, color=['Cd44', 'Itgam'], wspace=0.4, save="macrophage.pdf") # ITGAM=CD11b
sc.pl.umap(adata, color=['Csf1r', 'Itgam', 'Ly6'], wspace=0.4, save="neutrophil.pdf")
sc.pl.umap(adata, color=['Map2', 'Sypl'], wspace=0.4, save="neuron.pdf")
sc.pl.umap(adata, save="total_nocolor.pdf")

adata.layers['count'] = pd.DataFrame.sparse.from_spmatrix(raw_adata[:, adata.var_names].X).astype(float)


adata.write_h5ad("preprocessed_adata.h5ad")

adata = sc.read_h5ad("/mnt/Donald/ito/220812/preprocessed_adata.h5ad")


sc.tl.leiden(adata, resolution=0.2, key_added="coarse_leiden")
sc.pl.umap(adata, color="coarse_leiden")
sc.tl.rank_genes_groups(adata, groupby="coarse_leiden")
pd.DataFrame(adata.uns["rank_genes_groups"]["names"]).loc[:10, 3]