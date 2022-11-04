
import os
import sys
wd = '/mnt/Donald/hoshino/2_kidney/221103/'
os.makedirs(wd, exist_ok=True)
os.chdir(wd)

from importlib import reload
import scipy as sp
from matplotlib.offsetbox import AnchoredText
import seaborn as sns
import anndata
import torch
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import deepcolor
np.random.seed(1)
torch.manual_seed(1)

sys.path.append("/home/hirose/Documents/main/Collaborate2/hoshino/kidney")
import deepcolor_mod.workflow as wf


sc_adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/221103/adata_total_annotated2.h5')
sp_adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/Mouse_kidney/Visium/preprocess221031/sp_adata.h5ad')

sc_adata = sc_adata[:, sc_adata.layers['count'].toarray().sum(axis=0) > 10]
sp_adata = sp_adata[:, sp_adata.layers['count'].toarray().sum(axis=0) > 10]
common_genes = np.intersect1d(sc_adata.var_names, sp_adata.var_names)
sc_adata = sc_adata[:, common_genes]
sp_adata = sp_adata[:, common_genes]

reload(wf)
sc_adata, sp_adata = wf.estimate_spatial_distribution(sc_adata, sp_adata, 
                            x_batch_size=300, s_batch_size=300, 
                            param_save_path='/mnt/Donald/hoshino/2_kidney/Mouse_kidney/Visium/preprocess221031/opt_params.pt', 
                            first_epoch=1000, second_epoch=500)

# zでumap与え直し
sc.pp.neighbors(sc_adata, use_rep="X_zl")
sc.tl.umap(sc_adata)
sc.pl.umap(sc_adata, color="Annotation")

sp_adata = wf.calculate_imputed_spatial_expression(sc_adata, sp_adata)

sc.pl.spatial(sp_adata[sp_adata.obs["library_id"]=="CTL", :], library_id="CTL", color="Wif1", use_raw=False)
sc.pl.spatial(sp_adata[sp_adata.obs["library_id"]=="CTL", :], library_id="CTL", color="Wif1", layer="imputed_exp")
sc.pl.spatial(sp_adata[sp_adata.obs["library_id"]=="CTL", :], library_id="CTL", color="Wif1", layer="count")

"Dock5" in sp_adata.var_names



sp_adata = wf.calculate_clusterwise_distribution(sc_adata, sp_adata, 'Annotation')

celltypes = np.unique(sc_adata.obs["Annotation"])
library_names = ["PBS","CTL","PE2","PE1"]
for celltype in celltypes:
    fig = plt.figure(figsize=(20,5))
    for i, library in enumerate(library_names):
        axes = fig.add_subplot(1,4,i+1)
        sc.pl.spatial(sp_adata[sp_adata.obs["library_id"]==library, :], library_id=library, 
                    color=celltype, ax=axes, show=False, title="")
    plt.suptitle(celltype)
    plt.savefig(f"figures/spatial_{celltype}.png")


sc_adata.write_h5ad("sc_adata_train.h5")
sp_adata.write_h5ad("sp_adata_train.h5")
