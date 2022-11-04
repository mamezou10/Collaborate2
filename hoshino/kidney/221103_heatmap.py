
import os
import sys
wd = '/mnt/Donald/hoshino/2_kidney/221103_analysis/'
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

sns.clustermap(df.iloc[:,26:52], row_colors = sample_colors)
plt.savefig("figures/heatmap_cluster9.png")
plt.show()
