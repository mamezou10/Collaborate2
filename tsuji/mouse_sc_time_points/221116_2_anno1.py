
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import os

out_dir = "/mnt/Donald/tsuji/mouse_sc/221116/"
os.makedirs(out_dir)
os.chdir(out_dir)

adata=sc.read_h5ad(f'{out_dir}/adata_total_scanorama.h5')

# leiden少し荒くする
sc.tl.leiden(adata, resolution=0.5, key_added="coarse_leiden")
adata.uns['log1p']["base"] = None
sc.tl.rank_genes_groups(adata, 'coarse_leiden', method='t-test')

## get degs
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
res_df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']})
mouse2human = pd.read_csv("/home/hirose/Documents/main/gmts/mouse2human.txt", sep="\t")

cl_geness = []
n_col = len(res_df.columns)//2
for i in range(n_col):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-4 & gene.isin(@adata.var_names)").sort_values("pvals").mouse[:100]
    cl_geness.append(cl_genes.tolist())


## annotate with celltype markers
# CellMarker_gmt = parse_gmt('/home/hirose/Documents/main/gmts/CellMarker_Augmented_2021.txt')
panglao= "/home/hirose/Documents/main/gmts/PanglaoDB_markers_27_Mar_2020.tsv"
col_name = range(1,20,1)
df = pd.read_csv(panglao, names=col_name, sep="\t")
df = df.loc[:,~df.iloc[0,:].isna()]
col_name = df.iloc[0,:]
df = df.iloc[1:,:]
df.columns = col_name

markers = {k: list(df["official gene symbol"][df["cell type"]==k]) for k in set(df["cell type"][df.organ=="Brain"])}
# brain + macrophage marker
macrophages_marker = {"macrophages": list(df["official gene symbol"][df["cell type"]=="Macrophages"])}
markers.update(macrophages_marker)
# markers = {k: v for k, v in CellMarker_gmt.items() if "brain" in k.lower()}
# markers.update({k: v for k, v in CellMarker_gmt.items() if "lung" in k.lower()})

# heatmap brain + mphage
marker_genes = {ct: np.intersect1d(sum(cl_geness,[]), [str.capitalize() for str in markers[ct]]) for ct in markers.keys()}
sc.pl.heatmap(adata, marker_genes, groupby="coarse_leiden", dendrogram=True, save="markers.png", figsize=(30,20))

# heatmap immune
markers = {k: list(df["official gene symbol"][df["cell type"]==k]) for k in set(df["cell type"][df.organ=='Immune system'])}
marker_genes = {ct: np.intersect1d(sum(cl_geness,[]), [str.capitalize() for str in markers[ct]]) for ct in markers.keys()}
sc.pl.heatmap(adata, marker_genes, groupby="coarse_leiden", dendrogram=True, save="markers_immune.png", figsize=(30,20))




## cell annotation class1
obs = adata.obs
obs["cellClass1"]="none"
obs["cellClass1"][(obs.coarse_leiden=="0")|(obs.coarse_leiden=="1")|(obs.coarse_leiden=="2")|(obs.coarse_leiden=="3")|(obs.coarse_leiden=="12")] = "cellClass1-1"
obs["cellClass1"][(obs.coarse_leiden=="19")|(obs.coarse_leiden=="21")|(obs.coarse_leiden=="7")] = "cellClass1-1"
obs["cellClass1"][(obs.coarse_leiden=="4")|(obs.coarse_leiden=="5")|(obs.coarse_leiden=="9")|(obs.coarse_leiden=="13")|(obs.coarse_leiden=="15")|(obs.coarse_leiden=="18")|(obs.coarse_leiden=="20")] = "cellClass1-3"
obs["cellClass1"][(obs.coarse_leiden=="6")|(obs.coarse_leiden=="8")|(obs.coarse_leiden=="10")|(obs.coarse_leiden=="11")|(obs.coarse_leiden=="14")|(obs.coarse_leiden=="16")|(obs.coarse_leiden=="17")] = "cellClass1-1"
adata.obs = obs

sc.pl.umap(adata, color="coarse_leiden", legend_loc="on data", save="coarse_leiden.png")
sc.pl.umap(adata, color="cellClass1", save="cellClass.png")

adata.write_h5ad(f'{out_dir}/adata_total_cellClass1.h5')

