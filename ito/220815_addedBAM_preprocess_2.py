import vicdyf
import scvelo as scv
import scanpy as sc
from matplotlib import pyplot as plt
import os
import pandas as pd
import sys
from importlib import reload
from sklearn.preprocessing import LabelEncoder
sys.path.append("/mnt/244hirose/Scripts/")

import vicdyf_mod.src.vicdyf
reload(vicdyf_mod.src.vicdyf)
from vicdyf_mod.src.vicdyf import workflow as wf

wd = "/mnt/Donald/ito/220822_2"
os.makedirs(wd, exist_ok=True)
os.chdir(wd)


adata_1 = scv.read("/mnt/Donald/ito/added_bam/ConDay3/re_sorted_sample_alignments_ConDay3_possorted_genome_bam_filterd_G8E4H.loom", validate=False)
adata_2 = scv.read("/mnt/Donald/ito/added_bam/Day24(Right)/re_sorted_sample_alignments_Day24(Right)_possorted_genome_bam_filterd_VT3DE.loom", validate=False)
adata_3 = scv.read("/mnt/Donald/ito/added_bam/ReDay3(Left)/re_sorted_sample_alignments_ReDay3(Left)_possorted_genome_bam_filterd_0Q3W4.loom", validate=False)

adata_1.var_names_make_unique()
adata_2.var_names_make_unique()
adata_3.var_names_make_unique()

adata = adata_1.concatenate([adata_2, adata_3], 
                            uns_merge="same", index_unique=None, batch_key="sample", 
                            batch_categories=["ConDay3", "Day24", "ReDay3"])
raw_adata = adata.copy()

#adata = adata_1.copy()

# adata.layers["count"] =  adata.layers["matrix"].copy()
# adata.X =  adata.layers["matrix"].copy()
sc.pp.filter_cells(adata, min_genes=10)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, )
# adata = adata[:, adata.var.highly_variable]

# sc.pp.scale(adata)
sc.pp.pca(adata)
sc.pp.neighbors(adata)
sc.tl.umap(adata)
sc.tl.tsne(adata)
sc.tl.louvain(adata)
sc.tl.leiden(adata)

# sc.tl.leiden(adata, resolution=0.5, key_added="coarse_leiden")

sc.pl.umap(adata, color=['leiden'])
sc.pl.umap(adata, color=['leiden', 'sample'], save="ref.pdf")



adata.write_h5ad("/mnt/Donald/ito/added_bam/preprocessed_adata.h5ad")


