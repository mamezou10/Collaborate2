
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import os
import seaborn as sns

data_dir = "/mnt/Donald/tsuji/mouse_sc/221116/"
out_dir = "/mnt/Donald/tsuji/mouse_sc/221117/"
os.makedirs(out_dir)
os.chdir(out_dir)

adata=sc.read_h5ad(f'{data_dir}/adata_total_cellClass1.h5')

## microgliaを含みそうなクラスタに限る scanoramaで揃えてplotする
adata = adata[adata.obs["cellClass1"]=="cellClass1-1"].copy()
adata.X = adata.layers["count"].copy()

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40, use_rep="Scanorama")
sc.tl.umap(adata)
sc.tl.leiden(adata )

adata = adata[(adata.obs.leiden=="11")|(adata.obs.leiden=="13")|(adata.obs.leiden=="14")|(adata.obs.leiden=="15")|(adata.obs.leiden=="19")].copy()

# クラスタごとのsamples割合を積み上げ棒グラフにする
adata.obs.groupby([ "sample"]).size()
# sample
# Day4       127
# Day7       237
# Day10      164
# KO_Day7    194

df = pd.crosstab(adata.obs['leiden'], adata.obs['sample'], normalize='index')
df.plot.bar(stacked=True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
plt.tight_layout()
plt.savefig(f'{out_dir}/stacked_bar.png')


df = adata.obs.groupby(["leiden", "sample"]).size().reset_index()
df.columns = ['leiden', 'sample', "count"]
plt.figure(figsize=(20, 5))
sns.barplot(data=df, x="leiden", y="count", hue="sample")
plt.savefig(f'{out_dir}/lined_bar.png')


# クラスタごとのenrich
# もうやってある

