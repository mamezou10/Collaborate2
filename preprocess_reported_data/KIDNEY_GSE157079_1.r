
## Rオブジェクトを取り込んでh5adにした
# SeuratDiskのインストール周りに注意
# 環境は(seurat4)

library(Seurat)
if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes")
}
remotes::install_github("mojaveazure/seurat-disk")
library(SeuratDisk) ## condaで入れてはいけない！！！

#library(SeuratObject)
# count 
counts <- readRDS("/mnt/Daisy/sc_KIDNEY_GSE157079/GSE157079_P0_adult_counts.rds") 
object <- CreateSeuratObject(counts, project = "SeuratProject", assay = "RNA")


SaveH5Seurat(object, filename = "/mnt/Daisy/sc_KIDNEY_GSE157079/kidney.h5Seurat")
Convert("/mnt/Daisy/sc_KIDNEY_GSE157079/kidney.h5Seurat", dest = "h5ad")










######## 未使用

DimPlot(object, group.by="色分けをする種類", split.by="別々に散布図を表示するとき", label=TRUE)+ ggtitle("自動でのCelltyping")
DimPlot(object,  label=TRUE)+ ggtitle("自動でのCelltyping")

object[["percent.mt"]] <- PercentageFeatureSet(object, pattern = "^mt-")

# Visualize QC metrics as a violin plot
VlnPlot(object, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)

# FeatureScatter is typically used to visualize feature-feature relationships, but can be used
# for anything calculated by the object, i.e. columns in object metadata, PC scores etc.
plot1 <- FeatureScatter(object, feature1 = "nCount_RNA", feature2 = "percent.mt")
plot2 <- FeatureScatter(object, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")
plot1 + plot2


object <- subset(object, subset = nFeature_RNA > 200 & nFeature_RNA < 2500 & percent.mt < 5)
object <- NormalizeData(object, normalization.method = "LogNormalize", scale.factor = 10000)

object <- FindVariableFeatures(object, selection.method = "vst", nfeatures = 2000)

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(object), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(object)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2

object <- FindVariableFeatures(object, selection.method = "vst", nfeatures = 2000)

# Identify the 10 most highly variable genes
top10 <- head(VariableFeatures(object), 10)

# plot variable features with and without labels
plot1 <- VariableFeaturePlot(object)
plot2 <- LabelPoints(plot = plot1, points = top10, repel = TRUE)
plot1 + plot2

pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))

# Examine and visualize PCA results a few different ways
print(pbmc[["pca"]], dims = 1:5, nfeatures = 5)

VizDimLoadings(pbmc, dims = 1:2, reduction = "pca")
DimPlot(pbmc, reduction = "pca")
DimHeatmap(pbmc, dims = 1:15, cells = 500, balanced = TRUE)

# NOTE: This process can take a long time for big datasets, comment out for expediency. More
# approximate techniques such as those implemented in ElbowPlot() can be used to reduce
# computation time
pbmc <- JackStraw(pbmc, num.replicate = 100)
pbmc <- ScoreJackStraw(pbmc, dims = 1:20)

JackStrawPlot(pbmc, dims = 1:15)
ElbowPlot(pbmc)







