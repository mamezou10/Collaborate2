
import os
import sys
wd = '/mnt/Donald/hoshino/2_kidney/221108/'
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
import seaborn as sns
# import deepcolor
np.random.seed(1)
torch.manual_seed(1)

sys.path.append("/home/hirose/Documents/main/Collaborate2/hoshino/kidney")
import deepcolor_mod.workflow as wf

lt_df = pd.read_csv('/home/hirose/Documents/main/gmts/shimam_ligand_target_matrix.csv', index_col=0)
lt_df = lt_df.drop(index=lt_df.index[[14725]]) ## ligand名がNanになっている行
mouse2human = pd.read_csv("/home/hirose/Documents/main/gmts/mouse2human.txt", sep="\t")

sp_adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/221103/sp_adata_train.h5')
sc_adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/221103/sc_adata_train.h5')

# all_sp_adata = sc.read_h5ad("/mnt/Donald/hoshino/2_kidney/Mouse_kidney/Visium/preprocess221031/sp_adata.h5ad")

sp_adata = wf.calculate_clusterwise_distribution(sc_adata, sp_adata, 'leiden')

# sc.pl.spatial(sp_adata[sp_adata.obs["library_id"]=="CTL", :], library_id="CTL", color=["0", "2","13","19"])

boolean_l = (((sp_adata.obs["library_id"]=="PE1") | (sp_adata.obs["library_id"]=="PE2")) & (sp_adata.obs["clusters"]=="9")).tolist()
# boolean_l = ((sp_adata.obs["library_id"]=="PBS") & (sp_adata.obs["clusters"]=="9")).tolist()
# boolean_l = ((sp_adata.obs["library_id"]=="CTL") & (sp_adata.obs["clusters"]=="9")).tolist()

sc_adata.obsm["map2sp"] = np.array(pd.DataFrame(sc_adata.obsm["map2sp"]).loc[:, boolean_l].copy())
sc_adata.uns['log1p']["base"] = None

human_genes = pd.merge(pd.DataFrame(sc_adata.var_names), mouse2human, how="left", left_on=0, right_on="mouse")
sc_adata_human = sc_adata.copy()
sc_adata_human.var_names = human_genes.human
sc_adata_human = sc_adata_human[:,[sc_adata_human.var_names[i] is not np.nan for i in list(range(len(sc_adata_human.var_names)))]]
sc_adata_human.var_names_make_unique()
np.random.seed(1)

fig, coexp_cc_df = wf.calculate_proximal_cell_communications(sc_adata_human, 'leiden', lt_df, ["6","11","17"],        # macrophages
                                                            celltype_sample_num=100, ntop_genes=4000,  
                                                            each_display_num=3, 
                                                            role="sender", edge_thresh=1)
fig




kari = pd.DataFrame(sc_adata.obsm["map2sp"])
kari = kari.apply(lambda x: x/ x.sum(), axis=1)

ctl = kari.loc[:, ((sp_adata.obs["library_id"]=="CTL") & (sp_adata.obs["clusters"]=="9")).tolist()].copy().mean(axis=1)
pbs = kari.loc[:, ((sp_adata.obs["library_id"]=="PBS") & (sp_adata.obs["clusters"]=="9")).tolist()].copy().mean(axis=1)
pe = kari.loc[:, (((sp_adata.obs["library_id"]=="PE1") | (sp_adata.obs["library_id"]=="PE2")) & (sp_adata.obs["clusters"]=="9")).tolist()].copy().mean(axis=1)

obs = pd.DataFrame(sc_adata.obs)
obs["ctl"] = ctl.tolist()
obs["pbs"] = pbs.tolist()
obs["pe"] = pe.tolist()
sc_adata.obs = obs


sc.pl.umap(sc_adata, color=["pbs","ctl","pe"], vmax=0.001, save="cluster9_sample.png")

[(sc_adata.obs["leiden"]=="6")|(sc_adata.obs["leiden"]=="11")|(sc_adata.obs["leiden"]=="17")]



sc.pl.spatial(sp_adata[sp_adata.obs["library_id"]=="CTL", :], library_id="CTL", color=["6"])





