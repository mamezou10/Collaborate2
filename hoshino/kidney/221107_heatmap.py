
import os
import sys
wd = '/mnt/Donald/hoshino/2_kidney/221107/'
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


sp_adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/221103/sp_adata_train.h5')
sc_adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/221103/sc_adata_train.h5')

# all_sp_adata = sc.read_h5ad("/mnt/Donald/hoshino/2_kidney/Mouse_kidney/Visium/preprocess221031/sp_adata.h5ad")

sp_adata = wf.calculate_clusterwise_distribution(sc_adata, sp_adata, 'leiden')

df = pd.DataFrame(sp_adata.obs)
df = df[df.clusters=="9"].sort_values("library_id")


samples = df["library_id"].tolist()
sample_colors=[]
for item in samples:
    if item=="PBS":
        item_mod = "b"
    elif item=="CTL":
        item_mod = "k"
    else:
        item_mod = "g"
    sample_colors.append(item_mod)

sns.clustermap(df.iloc[:,26:52], row_colors = sample_colors, row_cluster=False)
plt.savefig("figures/heatmap_cluster9_2.png")
plt.show()


sns.clustermap(df[["6","11","17"]], row_colors = sample_colors, row_cluster=True)
plt.savefig("figures/heatmap_cluster9_macrophage.png")
plt.show()


df = df[["library_id", "6","11","17"]].melt(id_vars=["library_id"])
df.library_id = df.library_id.str.replace("PE1", "PE").replace("PE2", "PE")

sns.violinplot(x='library_id', y="value", data=df,  dodge=True, jitter=True, palette='Set3')
plt.savefig("figures/violin_cluster9_macrophage.png")
plt.show()

import scipy
stat, pvalue = scipy.stats.ttest_ind(df[df.library_id=="PBS"]["value"], df[df.library_id=="PE"]["value"])
stat, pvalue = scipy.stats.ttest_ind(df[df.library_id=="PBS"]["value"], df[df.library_id=="CTL"]["value"])
stat, pvalue = scipy.stats.ttest_ind(df[df.library_id=="PE"]["value"], df[df.library_id=="CTL"]["value"])




df = df[["library_id", "0","2","13","19"]].melt(id_vars=["library_id"])
df.library_id = df.library_id.str.replace("PE1", "PE").replace("PE2", "PE")

sns.violinplot(x='library_id', y="value", data=df, hue="variable", dodge=True, jitter=True, palette='Set3')
plt.savefig("figures/violin_cluster9_macrophage.png")
plt.show()

import scipy
stat, pvalue = scipy.stats.ttest_ind(df[df.library_id=="PBS"]["value"], df[df.library_id=="PE"]["value"])
stat, pvalue = scipy.stats.ttest_ind(df[df.library_id=="PBS"]["value"], df[df.library_id=="CTL"]["value"])
stat, pvalue = scipy.stats.ttest_ind(df[df.library_id=="PE"]["value"], df[df.library_id=="CTL"]["value"])




df = pd.DataFrame(sp_adata.obs)
df = df[df.clusters=="9"].sort_values("library_id")
samples = df["library_id"].tolist()
sample_colors=[]
for item in samples:
    if item=="PBS":
        item_mod = "b"
    elif item=="CTL":
        item_mod = "k"
    else:
        item_mod = "g"
    sample_colors.append(item_mod)

sns.clustermap(df[["0","2","13","19"]], row_colors = sample_colors, row_cluster=True)
plt.savefig("figures/heatmap_cluster9_Endotherial.png")
plt.show()





df = pd.DataFrame(sp_adata[sp_adata.obs["clusters"]=="9"].obsm["map2sc"])
samples = pd.DataFrame(sp_adata[sp_adata.obs["clusters"]=="9"].obs)["library_id"].tolist()
sample_colors=[]
for item in samples:
    if item=="PBS":
        item_mod = "b"
    elif item=="CTL":
        item_mod = "k"
    else:
        item_mod = "g"
    sample_colors.append(item_mod)

celltypes = np.unique(sc_adata.obs.leiden)
cell_color_dict = dict(zip(celltypes, palette[:len(celltypes)]))
celltypes_colors = [cell_color_dict.get(element, element)  for element in sc_adata.obs.leiden]

palette = sns.color_palette("inferno", 20).as_hex() + sns.color_palette("Spectral", 10).as_hex()


sns.clustermap(df, row_colors = sample_colors, col_colors= celltypes_colors, row_cluster=False, vmax=0.01)
plt.savefig("figures/heatmap_cluster9_single_cell.png")
plt.show()






