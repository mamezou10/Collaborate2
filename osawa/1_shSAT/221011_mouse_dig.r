library("DESeq2")
library(data.table)
library(tidyverse)
library(immunedeconv)
library(pheatmap)

directory <- ""

sampleFiles <- list.files("/mnt/ssd8t/Projects/osawa/SAT1/", recursive=TRUE, full.names=TRUE, pattern="_mouse_count.txt")

condition <- c(rep("SC", 3), rep("shSAT1", 3))
sampleTable <- data.frame(sampleName = basename(sampleFiles),
                      fileName = sampleFiles,
                      condition = condition)
ddsHTSeq <- DESeqDataSetFromHTSeqCount(sampleTable = sampleTable,
                                   directory = directory,
                                   design= ~ condition)
dds <- DESeq(ddsHTSeq)
res <- results(dds, contrast = c("condition", "shSAT1", "SC"))
fwrite(data.frame(res), "shSAT1_vs_SC.tsv", sep="\t", row.names = TRUE)
fwrite(as.data.frame(ddsHTSeq@assays@data$counts), "shSAT1_SC_counts.tsv", sep="\t", row.names = TRUE)

mouse2human <- fread("/home/hirose/Documents/Collaborate/gmts/mouse2human.txt")

df <- rownames_to_column(as.data.frame(ddsHTSeq@assays@data$counts), "mouse") 
df <- df %>% inner_join(mouse2human, by="mouse")
df <- df %>% group_by(human) %>% select(-c("mouse")) %>% summarise_all(sum)

dec_df <- data.frame(df, row.names=df$human) %>% select(-"human")



res <- immunedeconv::deconvolute(dec_df, "quantiseq")
hm_df <- res[,2:7]
#hm_df[ hm_df > 0.05 ] <- 0.05
rownames(hm_df) <- res$cell_type
pdf("mouse_quantiseq.pdf")
pheatmap(hm_df, fontsize_row = 10, cluster_col=F, cluster_row=T); dev.off()

res_xcell <- immunedeconv::deconvolute(dec_df, "xcell")
hm_df <- res_xcell[,2:7]
#hm_df[ hm_df > 0.008 ] <- 0.008
rownames(hm_df) <- res_xcell$cell_type
pdf("mouse_xcell.pdf")
pheatmap(hm_df, fontsize_row = 10); dev.off()

res <- immunedeconv::deconvolute(dec_df, "epic")
hm_df <- res[,2:7]
hm_df[ hm_df > 0.001 ] <- 0.001
rownames(hm_df) <- res$cell_type
pdf("mouse_epic.pdf")
pheatmap(hm_df, fontsize_row = 10); dev.off()



res <- results(dds, contrast = c("condition", "shSAT1", "SC"))
upgenes <- as.data.frame(res) %>% filter(log2FoldChange > log2(1.5) & pvalue < 0.05)
downgenes <- as.data.frame(res) %>% filter(log2FoldChange < log2(1/1.5) & pvalue < 0.05)

upgenes <- rownames_to_column(upgenes, "mouse") %>% inner_join(mouse2human, by="mouse")
downgenes <- rownames_to_column(downgenes, "mouse") %>% inner_join(mouse2human, by="mouse")

genesets <- vector("list", 2)
genesets[1] <- list(upgenes$human)
genesets[2] <- list(downgenes$human)
names(genesets) <- c("shSAT1_up_genes", "shSAT1_down_genes")
results <- cbind(names(genesets), sapply(genesets, length), sapply(genesets, paste, collapse="\t"))
fwrite(results, "genesets_mouse.gmt", row.names = FALSE, col.names = FALSE, sep="\t", quote = FALSE)

source("/mnt/244hirose/Scripts/gmts/my_fisher.R")

obj <- my_fisher("genesets_mouse.gmt", "/mnt/244hirose/Scripts/gmts/impala.gmt")
pval_thre <- 2
pval <- t(obj$p.value)
gene <- t(obj$genes)
id <- which(pval >= pval_thre, arr.ind=TRUE)
df <- data.frame("gene_ontology"=rownames(pval)[id[,1]],
"geneset"=colnames(pval)[id[,2]],gene=gene[pval>=pval_thre],minus_log10_pvalue=pval[pval>=pval_thre])
df <- df[sort.list(df[,4],decreasing=TRUE),]
write.table(df,"enrichment_analysis_impala_mouse.txt",
            row.names=FALSE,col.names=TRUE,sep="\t",quote=FALSE)

obj <- my_fisher("genesets_mouse.gmt", "/mnt/244hirose/Scripts/gmts/msigdb.v7.4.symbols.gmt")
pval_thre <- 2
pval <- t(obj$p.value)
gene <- t(obj$genes)
id <- which(pval >= pval_thre, arr.ind=TRUE)
df <- data.frame("gene_ontology"=rownames(pval)[id[,1]],
                 "geneset"=colnames(pval)[id[,2]],
                 gene=gene[pval>=pval_thre],
                 minus_log10_pvalue=pval[pval>=pval_thre])
df <- df[sort.list(df[,4],decreasing=TRUE),]
write.table(df,"enrichment_analysis_msigdb_mouse.txt",
            row.names=FALSE,col.names=TRUE,sep="\t",quote=FALSE)

