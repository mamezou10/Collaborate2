import vicdyf
import scvelo as scv
import scanpy as sc
from matplotlib import pyplot as plt
import os
import pandas as pd
import sys
from importlib import reload
from sklearn.preprocessing import LabelEncoder
sys.path.append("/mnt/244hirose/Scripts/")

import vicdyf_mod.src.vicdyf
reload(vicdyf_mod.src.vicdyf)
from vicdyf_mod.src.vicdyf import workflow as wf

wd = "/mnt/Donald/ito/220816_2"
os.makedirs(wd, exist_ok=True)
os.chdir(wd)


adata = sc.read_h5ad("/mnt/Donald/ito/220816/adata.h5ad")

adata_ref = sc.read_h5ad("/mnt/Daisy/sc_BRAIN_GSE129788/adata.h5ad")
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

