
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


sc_adata = sc.read_h5ad('sc_adata_train.h5')
sp_adata = sc.read_h5ad('sp_adata_train.h5')


## interaction
lt_df = pd.read_csv('/home/hirose/Documents/main/gmts/ligand_target_df.csv', index_col=0)
mouse2human = pd.read_csv("/home/hirose/Documents/main/gmts/mouse2human.txt", sep="\t")
human_genes = pd.merge(pd.DataFrame(sc_adata.var_names), mouse2human, how="left", left_on=0, right_on="mouse")
sc_adata_human = sc_adata.copy()
sc_adata_human.var_names = human_genes.human
sc_adata_human = sc_adata_human[:,[sc_adata_human.var_names[i] is not np.nan for i in list(range(len(sc_adata_human.var_names)))]]
sc_adata_human.var_names_make_unique()
sc_adata_human.var_names_make_unique()
# sc_adata_human.obs_names_make_unique()
np.random.seed(1)
sc_adata_human.uns['log1p']["base"] = None
# del sc_adata_human.uns['log1p']['base']
reload(wf)
reload(sc.preprocessing._highly_variable_genes)
fig, coexp_cc_df = wf.calculate_proximal_cell_communications(sc_adata_human, 
                            'Annotation', lt_df, ["Podocytes"], 
                            celltype_sample_num=100, ntop_genes=500, each_display_num=3, role="sender", edge_thresh=0.3)
fig








