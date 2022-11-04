import scanpy as sc
import os

out_dir = "/mnt/Donald/hoshino/2_kidney/221031/"
os.chdir(out_dir)

# for ingest REF lung
adata_ref = sc.read_h5ad('/mnt/Daisy/Tabula_muris/kidney.h5ad')

# for ingest SC
adata_sample = sc.read_h5ad('/mnt/Donald/hoshino/2_kidney/221024/adata_total.h5')

# ingest
var_names = adata_ref.var_names.intersection(adata_sample.var_names)
adata_ref = adata_ref[:, var_names]
adata_sample = adata_sample[:, var_names]

kari_adata = sc.tl.ingest(adata_sample, adata_ref, obs='Annotation', inplace=False)
adata_sample.obs["Annotation"] = kari_adata.obs["Annotation"].copy()

sc.pl.umap(adata_sample, color=["leiden","Annotation"], alpha=0.5, save='ingest_umap.png')
adata_sample.write_h5ad('/mnt/Donald/hoshino/2_kidney/221024/adata_total_annotated.h5')