library(Seurat)
# library(SeuratData)
library(SeuratDisk)
load("/mnt/Daisy/sc_BRAIN_GSE153424/all_types_all_brains.rds")
setwd("/mnt/Daisy/sc_BRAIN_GSE153424")

obj <- merged
SaveH5Seurat(obj, filename = "brain.h5Seurat")
Convert("brain.h5Seurat", dest = "h5ad")


# sc = import("scanpy")
# anndata = import("anndata")
# sc_scanpy = anndata$AnnData(t(as.matrix((obj[["RNA"]]@data))), 
#                             obs=obj@meta.data,
#                             var=data.frame(Symbol=rownames(obj)))
# rownames(sc_scanpy$var) = rownames(obj)
# sc_scanpy$obsm = list(pca=obj@reductions$pca@cell.embeddings[,1:2],
#                       umap=obj@reductions$umap@cell.embeddings[,1:2])
# sc_scanpy$var["highly_variable"] = sapply(sc_scanpy$var_names$tolist(), function(x) x %in% obj[["RNA"]]@var.features)
# sc_scanpy
# sc_scanpy.write("kari.h5sd")
