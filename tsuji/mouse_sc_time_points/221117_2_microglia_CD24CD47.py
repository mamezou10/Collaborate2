
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import os

data_dir = "/mnt/Donald/tsuji/mouse_sc/221116/"
out_dir = "/mnt/Donald/tsuji/mouse_sc/221117/"
os.makedirs(out_dir)
os.chdir(out_dir)

adata=sc.read_h5ad(f'{data_dir}/adata_total_cellClass1.h5')

## microgliaを含みそうなクラスタに限る scanoramaで揃えてplotする
adata = adata[adata.obs["cellClass1"]=="cellClass1-1"].copy()
adata.X = adata.layers["count"].copy()

adata = adata[(adata.obs.leiden!="17")&(adata.obs.leiden!="21")&(adata.obs.leiden!="12")&(adata.obs.leiden!="16")].copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40, use_rep="Scanorama")
sc.tl.umap(adata)
sc.tl.leiden(adata )



sc.pl.umap(adata, color=["coarse_leiden", "Tmem119", "Cd24a", "Siglecg", "Sirpa", "Cd47"], save="microglia_eat_markers.png")


