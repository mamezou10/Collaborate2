# https://rockefelleruniversity.github.io/RU_ATAC_Workshop.html

dir.create("/mnt/Donald/morim_atac_5")
setwd("/mnt/Donald/morim_atac_5")

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager"=="3.14")

BiocManager::install("GenomeInfoDb")

library(Rsubread)
library(Rsamtools)
library(ggplot2)
library(magrittr)

library(dplyr)
library(soGGi)
library(TxDb.Hsapiens.UCSC.hg38.knownGene)
BiocManager::install("TxDb.Hsapiens.UCSC.hg19.knownGene")
BiocManager::install("TxDb.Hsapiens.UCSC.hg18.knownGene")
BiocManager::install("TxDb.Mmusculus.UCSC.mm10.knownGene")
BiocManager::install("TxDb.Mmusculus.UCSC.mm9.knownGene")
BiocManager::install("TxDb.Rnorvegicus.UCSC.rn4.ensGene")
BiocManager::install("TxDb.Celegans.UCSC.ce6.ensGene")
BiocManager::install("TxDb.Dmelanogaster.UCSC.dm3.ensGene")
library(TxDb.Hsapiens.UCSC.hg19.knownGene)
library(GenomicAlignments)
library(rtracklayer)

library(biomaRt)
db <- useMart("ensembl")
hd <- useDataset("hsapiens_gene_ensembl", mart = db)

library(ChIPQC)
library(rtracklayer)
library(DT)
library(dplyr)
library(tidyr)


# 
peaks <- dir("/mnt/Daisy/morim_atac/macs_ed/", pattern = "*.narrowPeak", 
    full.names = TRUE)
myPeaks <- lapply(peaks, ChIPQC:::GetGRanges, simple = TRUE)

names(myPeaks) <- c("BR_NEG1", "BR_NEG2", "BR_NEG3", "BR_POS1", "BR_POS2", "BR_POS3", "DOX_NEG1", "DOX_NEG2", "DOX_NEG3", "DOX_POS1", "DOX_POS2", "DOX_POS3")
#Group <- factor(c("BR_NEG", "BR_POS", "DOX_NEG", "DOX_POS"))
myGRangesList <- GRangesList(myPeaks)

blklist <- import.bed("/mnt/Daisy/hg38_atac_blacklist/ENCFF356LFX.bed.gz")

myGRangesList<-GRangesList(myPeaks)   
reduced <- reduce(unlist(myGRangesList))
consensusIDs <- paste0("consensus_", seq(1, length(reduced)))
mcols(reduced) <- do.call(cbind, lapply(myGRangesList, function(x) (reduced %over% x) + 0))
reducedConsensus <- reduced
mcols(reducedConsensus) <- cbind(as.data.frame(mcols(reducedConsensus)), consensusIDs)
consensusIDs <- paste0("consensus_", seq(1, length(reducedConsensus)))
reducedConsensus
# https://www.biostars.org/p/392637/

consensusToCount <- reducedConsensus[!reducedConsensus %over% blklist & !seqnames(reducedConsensus) %in% "chrM"]


library(DESeq2)
load("/mnt/Donald/morim_atac_3/countsFromATAC.RData")
#metaData <- data.frame(Group=factor(c(rep("BR_NEG", 3), rep("BR_POS", 3), rep("DOX_NEG", 3), rep("DOX_POS", 3))), row.names = colnames(myCounts))
metaData <- data.frame(Group=factor(c("BR_neg", rep("BR_NEG", 2), "BR_pos", rep("BR_POS", 2), "DOX_neg", rep("DOX_NEG", 2), "DOX_pos", rep("DOX_POS", 2))), row.names = colnames(myCounts))
atacDDS <- DESeqDataSetFromMatrix(myCounts, metaData, ~Group, rowRanges = consensusToCount)
atacDDS <- DESeq(atacDDS)
atac_Rlog <- rlog(atacDDS)

library(DESeq2)
library(BSgenome.Hsapiens.UCSC.hg38)
library(tracktables)
library(clusterProfiler)
library(ChIPseeker)
library(data.table)
toOverLap <- promoters(TxDb.Hsapiens.UCSC.hg38.knownGene, upstream = 2000, downstream = 500)

## BR_NEG_minus_WT_NEG 
BR_NEG_minus_WT_NEG <- results(atacDDS, c("Group", "BR_NEG", "DOX_NEG"))#, format = "GRanges")
BR_NEG_minus_WT_NEG <- BR_NEG_minus_WT_NEG[order(BR_NEG_minus_WT_NEG$pvalue)]
BR_NEG_minus_WT_NEG

BR_NEG_minus_WT_NEG <- BR_NEG_minus_WT_NEG[(!is.na(BR_NEG_minus_WT_NEG$padj) & BR_NEG_minus_WT_NEG$padj < 0.05) & BR_NEG_minus_WT_NEG %over% toOverLap, ]
makebedtable(BR_NEG_minus_WT_NEG, "BR_NEG_minus_WT_NEG.html", ".")
anno_BR_NEG_minus_WT_NEG <- annotatePeak(BR_NEG_minus_WT_NEG, TxDb = TxDb.Hsapiens.UCSC.hg38.knownGene)

anno_BR_NEG_minus_WT_NEG_df <- as.data.frame(anno_BR_NEG_minus_WT_NEG)
res <- getBM(attributes = c("hgnc_symbol", "entrezgene_id"), 
            filters = "entrezgene_id", values = anno_BR_NEG_minus_WT_NEG_df$geneId, 
            mart = hd, useCache = FALSE) 
res$entrezgene_id <- as.character(res$entrezgene_id)
anno_BR_NEG_minus_WT_NEG_df <- anno_BR_NEG_minus_WT_NEG_df %>% full_join(res, by=c("geneId"="entrezgene_id"))


## BR_NEG_minus_BR_POS
BR_NEG_minus_BR_POS <- results(atacDDS, c("Group", "BR_NEG", "BR_POS"), format = "GRanges")
BR_NEG_minus_BR_POS <- BR_NEG_minus_BR_POS[order(BR_NEG_minus_BR_POS$pvalue)]
BR_NEG_minus_BR_POS

BR_NEG_minus_BR_POS <- BR_NEG_minus_BR_POS[(!is.na(BR_NEG_minus_BR_POS$padj) & BR_NEG_minus_BR_POS$padj < 0.05) & BR_NEG_minus_BR_POS %over% toOverLap, ]
makebedtable(BR_NEG_minus_BR_POS, "BR_NEG_minus_BR_POS.html", ".")
anno_BR_NEG_minus_BR_POS <- annotatePeak(BR_NEG_minus_BR_POS, TxDb = TxDb.Hsapiens.UCSC.hg38.knownGene)

anno_BR_NEG_minus_BR_POS_df <- as.data.frame(anno_BR_NEG_minus_BR_POS)
res <- getBM(attributes = c("hgnc_symbol", "entrezgene_id"), 
            filters = "entrezgene_id", values = anno_BR_NEG_minus_BR_POS_df$geneId, 
            mart = hd, useCache = FALSE) 
res$entrezgene_id <- as.character(res$entrezgene_id)
anno_BR_NEG_minus_BR_POS_df <- anno_BR_NEG_minus_BR_POS_df %>% full_join(res, by=c("geneId"="entrezgene_id"))


p_thre  <- 1
FC_thre <- 1.2
## atac??????????????????
atac_BR_neg_up      <- anno_BR_NEG_minus_BR_POS_df %>% filter(log2FoldChange > log2(FC_thre)   & padj < p_thre)
atac_BR_neg_down    <- anno_BR_NEG_minus_BR_POS_df %>% filter(log2FoldChange < log2(1/FC_thre) & padj < p_thre)
atac_BR_neg_WT_up   <- anno_BR_NEG_minus_WT_NEG_df %>% filter(log2FoldChange > log2(FC_thre)   & padj < p_thre)
atac_BR_neg_WT_down <- anno_BR_NEG_minus_WT_NEG_df %>% filter(log2FoldChange < log2(1/FC_thre) & padj < p_thre)

## Seq??????????????????
BR_WT <- fread("/mnt/Donald/morim_atac/RNAseq_copy/LS411N_BR_DOX_minus_vs_DOX_minus.tsv")
BRPOS_BRNEG <- fread("/mnt/Donald/morim_atac/RNAseq_copy/LS411N_BR_DOX_plus_vs_BR_DOX_minus.tsv")

BR_WT_up    <- BR_WT       %>% filter(log2FoldChange > log2(FC_thre)   & pvalue < p_thre)
BR_BRPOS_up <- BRPOS_BRNEG %>% filter(log2FoldChange < log2(1/FC_thre) & pvalue < p_thre)

## atac???Seq?????????????????????
# BR_NEG/BR_POS ???UP
inter_genes <- intersect(as.character(BR_BRPOS_up$V1), as.character(atac_BR_neg_up$hgnc_symbol))
targ_res <- getBM(attributes = c("entrezgene_id","hgnc_symbol"), 
            filters = "hgnc_symbol", values = inter_genes, 
            mart = hd, useCache = FALSE) 
ego_result <- enrichGO(gene    = targ_res$entrezgene_id,
                #universe      = res$entrezgene_id,
                OrgDb         = org.Hs.eg.db,
                ont           = "ALL", #"BP","CC","MF","ALL"????????????
                pAdjustMethod = "BH",
                pvalueCutoff  = 0.5,
                # qvalueCutoff  = 0.05, 
                readable      = TRUE) #Gene ID????????????????????????

kegg_enrichment_result <- enrichKEGG(
                            gene = targ_res$entrezgene_id,
                            organism = "hsa",
                            keyType = "kegg",
                            pvalueCutoff = 0.5,
                            pAdjustMethod = "BH",
                            minGSSize = 10,
                            maxGSSize = 500,
                            use_internal_data = FALSE)

df <- as.data.frame(ego_result) %>% arrange(qvalue)
fwrite(df, "enrichGO_common.tsv", sep="\t")
fwrite(as.data.frame(targ_res), "genes_common.tsv", sep="\t")

## atac???Seq?????????????????????
# BR_NEG/WT_NEG ???UP????????? BR_NEG/BR_POS ???UP
seq_targ_genes <- intersect(BR_WT_up$V1, BR_BRPOS_up$V1)
atac_targ_genes <- intersect(atac_BR_neg_WT_up$hgnc_symbol, atac_BR_neg_up$hgnc_symbol)

inter_genes <- intersect(as.character(seq_targ_genes), as.character(atac_targ_genes))
targ_res <- getBM(attributes = c("entrezgene_id","hgnc_symbol"), 
            filters = "hgnc_symbol", values = inter_genes, 
            mart = hd, useCache = FALSE) 

ego_result <- enrichGO(gene    = targ_res$entrezgene_id,
                #universe      = res$entrezgene_id,
                OrgDb         = org.Hs.eg.db,
                ont           = "ALL", #"BP","CC","MF","ALL"????????????
                pAdjustMethod = "BH",
                pvalueCutoff  = 0.5,
                # qvalueCutoff  = 0.05, 
                readable      = TRUE) #Gene ID????????????????????????


kegg_enrichment_result <- enrichKEGG(
                            gene = targ_res$entrezgene_id,
                            organism = "hsa",
                            keyType = "kegg",
                            pvalueCutoff = 0.5,
                            pAdjustMethod = "BH",
                            minGSSize = 10,
                            maxGSSize = 500,
                            use_internal_data = FALSE)

# df <- as.data.frame(ego_result) %>% arrange(qvalue)
# fwrite(df, "enrichGO_common.tsv", sep="\t")
fwrite(as.data.frame(targ_res), "genes_common_wide.tsv", sep="\t")


## atac???Seq?????????????????????
# BR_NEG/WT_NEG ???UP
inter_genes <- intersect(as.character(BR_WT_up$V1), as.character(atac_BR_neg_WT_up$hgnc_symbol))
targ_res <- getBM(attributes = c("entrezgene_id","hgnc_symbol"), 
            filters = "hgnc_symbol", values = inter_genes, 
            mart = hd, useCache = FALSE) 
ego_result <- enrichGO(gene    = targ_res$entrezgene_id,
                #universe      = res$entrezgene_id,
                OrgDb         = org.Hs.eg.db,
                ont           = "ALL", #"BP","CC","MF","ALL"????????????
                pAdjustMethod = "BH",
                pvalueCutoff  = 0.5,
                # qvalueCutoff  = 0.05, 
                readable      = TRUE) #Gene ID????????????????????????

kegg_enrichment_result <- enrichKEGG(
                            gene = targ_res$entrezgene_id,
                            organism = "hsa",
                            keyType = "kegg",
                            pvalueCutoff = 0.5,
                            pAdjustMethod = "BH",
                            minGSSize = 10,
                            maxGSSize = 500,
                            use_internal_data = FALSE)

df <- as.data.frame(ego_result) %>% arrange(qvalue)
fwrite(df, "enrichGO_common_WT.tsv", sep="\t")
fwrite(as.data.frame(targ_res), "genes_common_WT.tsv", sep="\t")




## kegg_link('hsa', 'pathway')
ttp_kegg_link <- function(target_db, source_db) {
    url <- paste0("https://rest.kegg.jp/link/", target_db, "/", source_db, collapse="")
    local_mydownload <- function (url, method, quiet = TRUE, ...) 
    {
        if (capabilities("libcurl")) {
            dl <- tryCatch(utils::download.file(url, quiet = quiet, 
                method = "libcurl", ...), error = function(e) NULL)
        }
        else {
            dl <- tryCatch(downloader::download(url, quiet = TRUE, 
                method = method, ...), error = function(e) NULL)
        }
        return(dl)
    }
    local_kegg_rest <- function (rest_url) 
    {
        message("Reading KEGG annotation online:\n")
        f <- tempfile()
        dl <- local_mydownload(rest_url, destfile = f)
        if (is.null(dl)) {
            message("fail to download KEGG data...")
            return(NULL)
        }
        content <- readLines(f)
        content %<>% strsplit(., "\t") %>% do.call("rbind", .)
        res <- data.frame(from = content[, 1], to = content[, 2])
        return(res)
    }
    local_kegg_rest(url)
}

ttp_kegg_list <- function(db) {
    url <- paste0("https://rest.kegg.jp/list/", db, collapse="")
    local_mydownload <- function (url, method, quiet = TRUE, ...) 
    {
        if (capabilities("libcurl")) {
            dl <- tryCatch(utils::download.file(url, quiet = quiet, 
                method = "libcurl", ...), error = function(e) NULL)
        }
        else {
            dl <- tryCatch(downloader::download(url, quiet = TRUE, 
                method = method, ...), error = function(e) NULL)
        }
        return(dl)
    }
    local_kegg_rest <- function (rest_url) 
    {
        message("Reading KEGG annotation online:\n")
        f <- tempfile()
        dl <- local_mydownload(rest_url, destfile = f)
        if (is.null(dl)) {
            message("fail to download KEGG data...")
            return(NULL)
        }
        content <- readLines(f)
        content %<>% strsplit(., "\t") %>% do.call("rbind", .)
        res <- data.frame(from = content[, 1], to = content[, 2])
        return(res)
    }
    local_kegg_rest(url)
}

rlang::env_unlock(env = asNamespace('clusterProfiler'))
rlang::env_binding_unlock(env = asNamespace('clusterProfiler'))
assign('kegg_link', ttp_kegg_link, envir = asNamespace('clusterProfiler'))
assign('kegg_list', ttp_kegg_list, envir = asNamespace('clusterProfiler'))
rlang::env_binding_lock(env = asNamespace('clusterProfiler'))
rlang::env_lock(asNamespace('clusterProfiler'))
#########