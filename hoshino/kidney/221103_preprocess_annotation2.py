import scanpy as sc
import os
import pandas as pd
import numpy as np

out_dir = "/mnt/Donald/hoshino/2_kidney/221103/"
os.chdir(out_dir)

# for ingest REF lung
adata_ref = sc.read_h5ad('/mnt/Daisy/Tabula_muris/kidney.h5ad')

# for ingest SC
adata_sample = sc.read_h5ad('adata_total.h5')

# ingest
var_names = adata_ref.var_names.intersection(adata_sample.var_names)
adata_ref = adata_ref[:, var_names]
adata_sample = adata_sample[:, var_names]

kari_adata = sc.tl.ingest(adata_sample, adata_ref, obs='Annotation', inplace=False)
adata_sample.obs["Annotation"] = kari_adata.obs["Annotation"].copy()

sc.pl.umap(adata_sample, color=["leiden"], legend_loc="on data", alpha=0.5, save='ingest_leiden.png')
sc.pl.umap(adata_sample, color=["Annotation"], alpha=0.5, save='ingest_annotation.png')
sc.pl.umap(adata_sample, color=["Annotation"], legend_loc="on data", alpha=0.5, save='ingest_annotation_on_data.png')

sc.pl.umap(adata_sample, color=["Podxl", "Crb2", "Dock5", "Adcy1"], alpha=0.5, save='Podocyte_markers.png')
sc.pl.umap(adata_sample, color=["Slc12a3", "Avpr2", "Slc8a1", "Wnk1"], alpha=0.5, save='Distal_tubule_markers.png')
sc.pl.umap(adata_sample, color=["Rhcg", "Adgrf5", "Jag1", "Insrr"], alpha=0.5, save='Intercalated_markers.png')
sc.pl.umap(adata_sample, color=["Myh11", "Ren1", "Akr1b7", "Acta2"], alpha=0.5, save='Juxtaglomerular_markers.png')
sc.pl.umap(adata_sample, color=["Pla2g7", "Dram1", "Thbs1", "Arid5a"], alpha=0.5, save='Loop_of_Henle_markers.png')
sc.pl.umap(adata_sample, color=["Itga8", "Acta2", "Angpt2", "Actn1"], alpha=0.5, save='Mesangial_markers.png')
sc.pl.umap(adata_sample, color=["Rnf186", "Kcne1", "Tacstd2", "Phactr1"], alpha=0.5, save='Principal_markers.png')
sc.pl.umap(adata_sample, color=["Alpl", "Scin", "Keg1", "G6pc"], alpha=0.5, save='Proximal_tubule_markers.png')



adata_sample.write_h5ad('adata_total_annotated.h5')






# run t-test 
adata =  sc.read_h5ad('adata_total.h5')
adata.uns['log1p']["base"] = None
sc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
# plot as heatmap for cluster5 genes
sc.pl.rank_genes_groups_heatmap(adata, n_genes=3, groupby="leiden", show_gene_labels=True, save=".png")
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
res_df = pd.DataFrame({group + '_' + key[:1]: result[key][group] for group in groups for key in ['names', 'pvals']})
res_df.to_csv('degs.csv')

marker_genes = annotate_geneset(res_df, pre="merker_gene")

marker_genes = parse_gmt('merker_gene_marker.gmt')
impla_gmt = parse_gmt('/home/hirose/Documents/main/gmts/impala.gmt')
msig_gmt = parse_gmt('/home/hirose/Documents/main/gmts/msigdb.v7.4.symbols.gmt')
Panglao_gmt = parse_gmt('/home/hirose/Documents/main/gmts/PanglaoDB_Augmented_2021.txt')
mouse2human = pd.read_csv("/home/hirose/Documents/main/gmts/mouse2human.txt", sep="\t")


for i in range(len(res_df.columns)//2):
    deg_df = res_df.iloc[:, (2*i):(2*i+2)]
    deg_df.columns =["gene", "pvals"]
    deg_df = pd.merge(deg_df, mouse2human, left_on="gene", right_on="mouse")
    total_genes = deg_df.human
    cl_genes = deg_df.query("pvals < 10**-4 ").sort_values("pvals").human[:100]
    pval_df = gene_set_enrichment(cl_genes, total_genes, Panglao_gmt)
    pval_df = pd.DataFrame({"pval":pval_df})
    pval_df[["cluster"]] = "cluster"+str(i)
    pval_df.to_csv('annotation/enrich_impla_cluster_'+ str(i) + '.csv')


obs = pd.DataFrame(adata.obs)
obs["Annotation"]="None"

obs["Annotation"][(obs.leiden=="0")|(obs.leiden=="2")|(obs.leiden=="13")|(obs.leiden=="19")] = "Endothelial Cells"
obs["Annotation"][(obs.leiden=="1")|(obs.leiden=="25")] = "Podocytes"
obs["Annotation"][(obs.leiden=="3")|(obs.leiden=="4")|(obs.leiden=="20")|(obs.leiden=="21")] = "Distal Tubule Cells"
obs["Annotation"][obs.leiden=="5"] = "Principal Cells"
obs["Annotation"][(obs.leiden=="6")|(obs.leiden=="17")] = "Macrophages"
obs["Annotation"][(obs.leiden=="7")|(obs.leiden=="10")|(obs.leiden=="14")] = "Proximal Tubule Cells"
obs["Annotation"][(obs.leiden=="8")|(obs.leiden=="15")|(obs.leiden=="18")] = "Intercalated Cells"
obs["Annotation"][obs.leiden=="9"] = "Pericytes__Juxtaglomerular Cells"
obs["Annotation"][obs.leiden=="11"] = "Myeloid-derived Suppressor Cells__Monocytes"
obs["Annotation"][obs.leiden=="12"] = "Smooth Muscle Cells__Pericytes__Juxtaglomerular Cells__Mesangial Cells"
obs["Annotation"][obs.leiden=="16"] = "T Cells"
obs["Annotation"][obs.leiden=="22"] = "Cholangiocytes"
obs["Annotation"][obs.leiden=="23"] = "B Cells"
obs["Annotation"][obs.leiden=="24"] = "Dendritic Cells"

adata.obs = obs
sc.pl.umap(adata, color=["Annotation"], alpha=0.5, save='ingest_annotation2.png')


# 地道 annotation version
adata.write_h5ad('adata_total_annotated2.h5')