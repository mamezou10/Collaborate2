import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import anndata
import os

out_dir = "/mnt/Donald/tsuji/mouse_sc/221130/"
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)

adata = sc.read_h5ad("adata.h5")

## annotate with celltype markers
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

# CellMarker_gmt = parse_gmt('/home/hirose/Documents/main/gmts/CellMarker_Augmented_2021.txt')
panglao= "/home/hirose/Documents/main/gmts/PanglaoDB_markers_27_Mar_2020.tsv"
col_name = range(1,20,1)
df = pd.read_csv(panglao, names=col_name, sep="\t")
df = df.loc[:,~df.iloc[0,:].isna()]
col_name = df.iloc[0,:]
df = df.iloc[1:,:]
df.columns = col_name
# brain marker
markers = {k: list(df["official gene symbol"][df["cell type"]==k]) for k in set(df["cell type"][df.organ=="Brain"])}
# brain + macrophage marker
macrophages_marker = {"macrophages": list(df["official gene symbol"][df["cell type"]=="Macrophages"])}
markers.update(macrophages_marker)

## bulkから特徴遺伝子
df = pd.read_excel("/mnt/Wake/LAW_RNASeq/20221117_MouseBrainTumor02/F4948_v684_mm10_summary.xlsx")
df_d7_count = df.iloc[1:,-5:]
df_d7_count = df_d7_count.iloc[:,[0,2,4]]
df_d7_count.index=list(df.iloc[1:,0])
df = df_d7_count.copy()
# df_d7_count["log_cancer"] = [ np.log2(n+1) for n in df_d7_count.iloc[:,0] ] 
# df_d7_count["log_MG"] = [ np.log2(n+1) for n in df_d7_count.iloc[:,1:4].mean(axis=1) ] 
# df_d7_count["log_phagoMG"] = [ np.log2(n+1) for n in df_d7_count.iloc[:,4] ] 
df["logFC_cancer"]  = np.array([np.log2(n+1) for n in df_d7_count.iloc[:,0]]) - np.array([np.log2(n+1) for n in df_d7_count.drop(df_d7_count.columns[0], axis=1).mean(axis=1)])
df["logFC_MG"]      = np.array([np.log2(n+1) for n in df_d7_count.iloc[:,1]]) - np.array([np.log2(n+1) for n in df_d7_count.drop(df_d7_count.columns[1], axis=1).mean(axis=1)])
df["logFC_phagoMG"] = np.array([np.log2(n+1) for n in df_d7_count.iloc[:,2]]) - np.array([np.log2(n+1) for n in df_d7_count.drop(df_d7_count.columns[2], axis=1).mean(axis=1)])

cancer_genes =  df.sort_values("logFC_cancer", ascending=False)[:100].index
cancer_genes2 =  df.sort_values("Day7_\nCancer.1", ascending=False)[:50].index
MG_genes =      df.sort_values("logFC_MG", ascending=False)[:100].index
MG_genes2 =      df.sort_values("Day7_\nCortex_\n1G2.1", ascending=False)[:50].index
phagoMG_genes = df.sort_values("logFC_phagoMG", ascending=False)[:100].index
phagoMG_genes2 = df.sort_values("Day7_\nPhagocyteMG.1", ascending=False)[:50].index

# brain + macrophage + bulk marker
bulk_marker = {"bulk_cancer": cancer_genes, "bulk_MG": MG_genes, "bulk_phageMG": phagoMG_genes}
# markers.update(bulk_marker)

# heatmap brain + mphage
marker_genes = {ct: np.intersect1d(sum(cl_geness,[]), [str.capitalize() for str in markers[ct]]) for ct in markers.keys()}
sc.pl.heatmap(adata, marker_genes, groupby="leiden", dendrogram=True, save="markers.png", figsize=(20,15), vmax=5)
marker_genes = {ct: np.intersect1d(sum(cl_geness,[]), [str.capitalize() for str in bulk_marker[ct]]) for ct in bulk_marker.keys()}
marker_genes.pop('bulk_MG')
sc.pl.heatmap(adata, marker_genes, groupby="leiden", dendrogram=True, save="markers_bulk.png", figsize=(20,20))

#
bulk_marker = {"bulk_cancer": list(set(cancer_genes2)-set(phagoMG_genes2)), "bulk_phageMG": list(set(phagoMG_genes2)-set(cancer_genes2))}
marker_genes = {ct: np.intersect1d(sum(cl_geness,[]), [str.capitalize() for str in bulk_marker[ct]]) for ct in bulk_marker.keys()}
sc.pl.heatmap(adata, marker_genes, groupby="leiden", dendrogram=True, save="markers_bulk2.png", figsize=(20,20))

# 
adata.obs["CellClass1"] = "CellClass1-2" 

adata.obs["CellClass1"][adata.obs["leiden"].isin(["0","1","2","3","4","5","7","9","11","13","14","17","18","21","22","28"])] = "CellClass1-1"


sc.pl.umap(adata, color="CellClass1", save="CellClass1.png")

adata.write_h5ad('adata_cellClass1.h5')