import scanpy as sc
import os
import pandas as pd
import numpy as np

out_dir = "/mnt/Donald/hoshino/2_kidney/221107/"
os.mkdir(out_dir)
os.chdir(out_dir)

adata = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/221103/adata_total.h5')

adata.uns['log1p']["base"] = None
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")



## get degs
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
res_df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']})

cl_geness = []
n_col = len(res_df.columns)//2
for i in range(n_col):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-4 & gene.isin(@adata.var_names)").sort_values("pvals").mouse[:100]
    cl_geness.append(cl_genes.tolist())


### part 1
## celltype markers
panglao= "/home/hirose/Documents/main/gmts/PanglaoDB_markers_27_Mar_2020.tsv"
col_name = range(1,20,1)
df = pd.read_csv(panglao, names=col_name, sep="\t")
df = df.loc[:,~df.iloc[0,:].isna()]
col_name = df.iloc[0,:]
df = df.iloc[1:,:]
df.columns = col_name

kidneys = df[(df.organ=="Kidney") | (df.organ=="Epithelium") |(df.organ=="Immune system") | (df.organ=="Vasculature") |(df.organ=="Adrenal glands")]

# heatmap
marker_genes1 = {ct: np.intersect1d(sum(cl_geness,[]), [str.capitalize() for str in kidneys.query("`cell type` == @ct")["official gene symbol"]]) for ct in np.unique(kidneys["cell type"])}
sc.pl.heatmap(adata, marker_genes1, groupby="leiden", dendrogram=True, save="markers.png", figsize=(30,5))


### part 2
## celltype markers
CellMarker_gmt = parse_gmt('/home/hirose/Documents/main/gmts/CellMarker_Augmented_2021.txt')

kidneys = {k: v for k, v in CellMarker_gmt.items() if "kidney" in k.lower()}
kidneys.update({k: v for k, v in CellMarker_gmt.items() if "renal" in k.lower()})

# heatmap
marker_genes = {ct: np.intersect1d(sum(cl_geness,[]), [str.capitalize() for str in kidneys[ct]]) for ct in kidneys.keys()}
sc.pl.heatmap(adata, marker_genes, groupby="leiden", dendrogram=True, save="markers2.png", figsize=(30,5))

# part 3
kidneys = {k: v for k, v in CellMarker_gmt.items() if "renal" in k.lower()}
kidneys.update(marker_genes1)
marker_genes = {ct: np.intersect1d(sum(cl_geness,[]), [str.capitalize() for str in kidneys[ct]]) for ct in kidneys.keys()}
sc.pl.heatmap(adata, marker_genes, groupby="leiden", dendrogram=True, save="merge_markers.png")



obs = pd.DataFrame(adata.obs)
obs["Annotation"]="Immune cells"

obs["Annotation"][(obs.leiden=="0")|(obs.leiden=="2")|(obs.leiden=="13")|(obs.leiden=="19")] = "Endothelial Cells"
obs["Annotation"][(obs.leiden=="1")|(obs.leiden=="25")] = "Podocytes"
obs["Annotation"][(obs.leiden=="3")|(obs.leiden=="4")|(obs.leiden=="20")|(obs.leiden=="21")] = "Distal Tubule Cells"
obs["Annotation"][obs.leiden=="5"] = "Principal Cells"
obs["Annotation"][(obs.leiden=="7")|(obs.leiden=="10")|(obs.leiden=="14")] = "Proximal Tubule Cells"
obs["Annotation"][(obs.leiden=="8")|(obs.leiden=="15")|(obs.leiden=="18")] = "Intercalated Cells"
obs["Annotation"][(obs.leiden=="6")|(obs.leiden=="11")|(obs.leiden=="17")] = "Macrophages"
obs["Annotation"][(obs.leiden=="9")|(obs.leiden=="12")] = "Mesangial Cells"
obs["Annotation"][(obs.leiden=="22")] = "Unknown"

adata.obs = obs
sc.pl.umap(adata, color=["Annotation"], alpha=0.5, save='annotation3.png')
adata.write_h5ad('adata_total_annotated3.h5')
