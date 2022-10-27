# https://rockefelleruniversity.github.io/RU_ATAC_Workshop.html

dir.create("/mnt/Donald/morim_atac_3")
setwd("/mnt/Donald/morim_atac_3")

library(Rsubread)
library(Rsamtools)
library(ggplot2)
library(magrittr)

library(dplyr)
library(soGGi)
library(TxDb.Hsapiens.UCSC.hg38.knownGene)
library(GenomicAlignments)
library(rtracklayer)


bams <- list.files("/mnt/Daisy/morim_atac/", full.name=T, pattern=".bam.sort.bam$")

TSSs <- resize(genes(TxDb.Hsapiens.UCSC.hg38.knownGene), fix = "start", 1)
TSSs

for(sortedBAM in bams[5]){
    # sortedBAM <- "/mnt/Daisy/morim_atac/20220520_BR-NEG_1.bam.sort.bam"
    sortedBAM <- bams[11]
    sample_name <- gsub("\\.bam\\.sort\\.bam", "", basename(sortedBAM))

    # pmapped <- propmapped(sortedBAM)
    # # pmapped
    # g <- idxstatsBam(sortedBAM) %>% 
    #     ggplot(aes(seqnames, mapped, fill = seqnames)) + 
    #     geom_bar(stat = "identity") + 
    #     coord_flip()
    # ggsave("mapping_chr.pdf")

    atacReads <- readGAlignmentPairs(sortedBAM, 
                                    param = ScanBamParam(mapqFilter = 1, 
                                    flag = scanBamFlag(isPaired = TRUE, isProperPair = TRUE), 
                                    what = c("qname", "mapq", "isize"),))
                                    #which = GRanges("chr20", IRanges(1, 63025520)))
    # length(atacReads)
    # atacReads

    atacReads_read1 <- GenomicAlignments::first(atacReads)
    insertSizes <- abs(elementMetadata(atacReads_read1)$isize)
    # head(insertSizes)

    # fragLenPlot <- table(insertSizes) %>% 
    #                 data.frame %>% 
    #                 rename(InsertSize = insertSizes, Count = Freq) %>% 
    #                 mutate(InsertSize = as.numeric(as.vector(InsertSize)), Count = as.numeric(as.vector(Count))) %>% 
    #                 ggplot(aes(x = InsertSize, y = Count)) + 
    #                 geom_line()
    # g <- fragLenPlot + geom_vline(xintercept = c(180, 247), colour = "red") + 
    #             geom_vline(xintercept = c(315, 437), colour = "darkblue") + 
    #             geom_vline(xintercept = c(100), colour = "darkgreen") + 
    #             theme_bw() #ヌクレオソームフリー（<100bp）、モノヌクレオソーム（180bp-247bp）、およびジヌクレオソーム（315-437
    # ggsave(paste0(sample_name, "frag_len.pdf"))

    # # Nucleosome free
    # nucFree <- regionPlot(bamFile = sortedBAM, testRanges = TSSs, style = "point", 
    #                         format = "bam", paired = TRUE, minFragmentLength = 0, 
    #                         maxFragmentLength = 100, forceFragment = 50)
    # # Mononucleosome
    # monoNuc <- regionPlot(bamFile = sortedBAM, testRanges = TSSs, style = "point", 
    #                         format = "bam", paired = TRUE, minFragmentLength = 180, 
    #                         maxFragmentLength = 240, forceFragment = 80)
    # # Dinucleosome
    # diNuc <- regionPlot(bamFile = sortedBAM, testRanges = TSSs, style = "point", 
    #                     format = "bam", paired = TRUE, minFragmentLength = 315, 
    #                     maxFragmentLength = 437, forceFragment = 160)

    # nucFree_gL <- nucFree
    # monoNuc_gL <- monoNuc 
    # diNuc_gL <- diNuc
    # save(monoNuc_gL,nucFree_gL,diNuc_gL,file = paste0(sample_name, '_soGGiResults.RData'))

    # #load(file = "soGGiResults.RData")

    # g <- plotRegion(nucFree_gL, outliers = 0.01)
    # ggsave(paste0(sample_name, "_nucFree_region.pdf"))
    # g <- plotRegion(monoNuc_gL, outliers = 0.01)
    # ggsave(paste0(sample_name, "_monoNuc_region.pdf"))
    # g <- plotRegion(diNuc_gL, outliers = 0.01)
    # ggsave(paste0(sample_name, "_diNuc_region.pdf"))

    atacReads_Open <- atacReads[insertSizes < 100, ]
    atacReads_MonoNuc <- atacReads[insertSizes > 180 & insertSizes < 240, ]
    atacReads_diNuc <- atacReads[insertSizes > 315 & insertSizes < 437, ]

    openRegionBam <- gsub("\\.bam", "_openRegions\\.bam", sortedBAM)
    monoNucBam <- gsub("\\.bam", "_monoNuc\\.bam", sortedBAM)
    diNucBam <- gsub("\\.bam", "_diNuc\\.bam", sortedBAM)

    export(atacReads_Open, openRegionBam, format = "bam")
    export(atacReads_MonoNuc, monoNucBam, format = "bam")

    openRegionBigWig <- gsub("\\.bam", "_openRegions\\.bw", sortedBAM)
    openRegionRPMBigWig <- gsub("\\.bam", "_openRegionsRPM\\.bw", sortedBAM)
    atacFragments_Open <- granges(atacReads_Open)
    export.bw(coverage(atacFragments_Open), openRegionBigWig)
}


## 
> print(object.size(atacReads), units = "auto")
25.8 Gb
> rm(atacReads)
> gc()



## macs
bams <- list.files("/mnt/Daisy/morim_atac/", full.name=T, pattern=".bam.sort.bam$")
for (bam in bams){
    sample_name <- gsub("\\.bam\\.sort\\.bam", "", basename(bam))
    path_out <- "/mnt/Daisy/morim_atac/macs_ed/"
    system(paste0("macs2 callpeak -t ", bam, " -f BAMPE --outdir ", path_out, " --name ", sample_name, " -g hs"))
}


library(ChIPQC)
library(rtracklayer)
library(DT)
library(dplyr)
library(tidyr)
blkList <- import.bed("/mnt/Daisy/hg38_atac_blacklist/ENCFF356LFX.bed.gz")

bams <- list.files("/mnt/Daisy/morim_atac/", full.name=T, pattern=".bam.sort.bam$")
i<- 1
bam <- bams[i]
sample_name <- gsub("\\.bam\\.sort\\.bam", "", basename(bam))

openRegionPeaks <- paste0("/mnt/Daisy/morim_atac/macs_ed/", sample_name, "_peaks.narrowPeak")

qcRes <- ChIPQCsample(gsub("\\.bam\\.sort\\.bam", "_openRegions\\.bam\\.sort_openRegions\\.bam", bam), 
                      peaks = openRegionPeaks, annotation = "hg38", 
                      blacklist = blkList, verboseT = FALSE) # chromosomes = "chr20", 
QCmetrics(qcRes) %>% t %>% 
                    data.frame %>% 
                    dplyr:::select(Reads, starts_with(c("Filt")), starts_with(c("RiP")), starts_with(c("RiBL"))) %>% 
                    datatable(rownames = NULL)

flagtagcounts(qcRes) %>% t %>% 
                    data.frame %>% 
                    mutate(Dup_Percent = (DuplicateByChIPQC/Mapped) * 100) %>% 
                    dplyr:::select(Mapped, Dup_Percent) %>% 
                    datatable(rownames = NULL)

MacsCalls_chr20 <- granges(qcRes[seqnames(qcRes) %in% "chr20"])

data.frame(Blacklisted = sum(MacsCalls_chr20 %over% blkList), 
           Not_Blacklisted = sum(!MacsCalls_chr20 %over% blkList))
MacsCalls_chr20_filtered <- MacsCalls_chr20[!MacsCalls_chr20 %over% blkList]

library(ChIPseeker)
MacsCalls_chr20_filteredAnno <- annotatePeak(MacsCalls_chr20_filtered, TxDb = TxDb.Hsapiens.UCSC.hg38.knownGene)
MacsCalls_chr20_filteredAnno
plotAnnoPie(MacsCalls_chr20_filteredAnno)
upsetplot(MacsCalls_chr20_filteredAnno)

MacsGranges_Anno <- as.GRanges(MacsCalls_chr20_filteredAnno)
TSS_MacsGranges_Anno <- MacsGranges_Anno[abs(MacsGranges_Anno$distanceToTSS) < 500]
TSS_MacsGranges_Anno

## downstream
library(rGREAT)
seqlevelsStyle(MacsCalls_chr20_filtered) <- "UCSC"
great_Job <- submitGreatJob(MacsCalls_chr20_filtered, species = "hg38")
availableCategories(great_Job)

great_ResultTable = getEnrichmentTables(great_Job, category = "GO")
names(great_ResultTable)
great_ResultTable[["GO Biological Process"]][1:4, ]

save(great_ResultTable, file = "ATAC_Data/ATAC_RData/Great_Results.RData")




## 
peaks <- dir("/mnt/Daisy/morim_atac/macs_ed/", pattern = "*.narrowPeak", 
    full.names = TRUE)
myPeaks <- lapply(peaks, ChIPQC:::GetGRanges, simple = TRUE)

names(myPeaks) <- c("BR_NEG1", "BR_NEG2", "BR_NEG3", "BR_POS1", "BR_POS2", "BR_POS3", "DOX_NEG1", "DOX_NEG2", "DOX_NEG3", "DOX_POS1", "DOX_POS2", "DOX_POS3")
Group <- factor(c("BR_NEG", "BR_POS", "DOX_NEG", "DOX_POS"))
myGRangesList <- GRangesList(myPeaks)
#consensusToCount <- soGGi:::runConsensusRegions(myGRangesList)

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

consensusToCount




library(limma)

g <- as.data.frame(elementMetadata(consensusToCount)) %>% dplyr::select(starts_with("BR_NEG")) %>% 
    vennDiagram(main = "Overlap for BR_NEG open regions") 
ggsave("overlap_BR_NEG.pdf")
g <- as.data.frame(elementMetadata(consensusToCount)) %>% dplyr::select(starts_with("BR_POS")) %>% 
    vennDiagram(main = "Overlap for BR_POS open regions") 
ggsave("overlap_BR_POS.pdf")
g <- as.data.frame(elementMetadata(consensusToCount)) %>% dplyr::select(starts_with("DOX_NEG")) %>% 
    vennDiagram(main = "Overlap for DOX_NEG open regions") 
ggsave("overlap_DOX_NEG.pdf")
g <- as.data.frame(elementMetadata(consensusToCount)) %>% dplyr::select(starts_with("DOX_POS")) %>% 
    vennDiagram(main = "Overlap for DOX_POS open regions") 
ggsave("overlap_DOX_POS.pdf")

library(tidyr)

myPlot <- as.data.frame(elementMetadata(consensusToCount)) %>% 
            dplyr::select(-consensusIDs) %>% 
            as.matrix %>% t %>% prcomp %>% .$x %>% 
            data.frame %>% 
            mutate(Samples = rownames(.)) %>% 
            mutate(Group = gsub("\\d$", "", Samples)) %>% 
            ggplot(aes(x = PC1, y = PC2, colour = Group,label=Samples)) + geom_point(size = 5) +geom_text(hjust=0, vjust=0)
ggsave("PCA.pdf")
myPlot

library(Rsubread)
occurrences <- elementMetadata(consensusToCount) %>% as.data.frame %>% dplyr::select(-consensusIDs) %>% rowSums

table(occurrences) %>% rev %>% cumsum

consensusToCount <- consensusToCount[occurrences >= 2, ]
consensusToCount

bamsToCount <- dir("/mnt/Daisy/morim_atac/", full.names = TRUE, pattern = "*\\.sort\\.bam$")
# indexBam(bamsToCount)
regionsToCount <- data.frame(GeneID = paste("ID", seqnames(consensusToCount), 
    start(consensusToCount), end(consensusToCount), sep = "_"), Chr = seqnames(consensusToCount), 
    Start = start(consensusToCount), End = end(consensusToCount), Strand = strand(consensusToCount))
fcResults <- featureCounts(bamsToCount, annot.ext = regionsToCount, isPairedEnd = TRUE, 
    countMultiMappingReads = FALSE, maxFragLength = 100)
myCounts <- fcResults$counts
colnames(myCounts) <-  c("BR_NEG1", "BR_NEG2", "BR_NEG3", "BR_POS1", "BR_POS2", "BR_POS3", "DOX_NEG1", "DOX_NEG2", "DOX_NEG3", "DOX_POS1", "DOX_POS2", "DOX_POS3")

save(myCounts, file = "countsFromATAC.RData")

library(DESeq2)
load("countsFromATAC.RData")
metaData <- data.frame(Group=factor(c(rep("BR_NEG", 3), rep("BR_POS", 3), rep("DOX_NEG", 3), rep("DOX_POS", 3))), row.names = colnames(myCounts))
atacDDS <- DESeqDataSetFromMatrix(myCounts, metaData, ~Group, rowRanges = consensusToCount)
atacDDS <- DESeq(atacDDS)
atac_Rlog <- rlog(atacDDS)
g <- plotPCA(atac_Rlog, intgroup = "Group", ntop = nrow(atac_Rlog))
ggsave("PCA_narrow.pdf")

library(DESeq2)
library(BSgenome.Hsapiens.UCSC.hg38)
library(tracktables)
library(TxDb.Hsapiens.UCSC.hg38.knownGene)
library(clusterProfiler)
library(ChIPseeker)
toOverLap <- promoters(TxDb.Hsapiens.UCSC.hg38.knownGene, upstream = 2000, downstream = 500)

## BR_NEG_minus_WT_NEG 
BR_NEG_minus_WT_NEG <- results(atacDDS, c("Group", "BR_NEG", "DOX_NEG"), format = "GRanges")
BR_NEG_minus_WT_NEG <- BR_NEG_minus_WT_NEG[order(BR_NEG_minus_WT_NEG$pvalue)]
BR_NEG_minus_WT_NEG

BR_NEG_minus_WT_NEG <- BR_NEG_minus_WT_NEG[(!is.na(BR_NEG_minus_WT_NEG$padj) & BR_NEG_minus_WT_NEG$padj < 0.05) & BR_NEG_minus_WT_NEG %over% toOverLap, ]
makebedtable(BR_NEG_minus_WT_NEG, "BR_NEG_minus_WT_NEG.html", ".")
anno_BR_NEG_minus_WT_NEG <- annotatePeak(BR_NEG_minus_WT_NEG, TxDb = TxDb.Hsapiens.UCSC.hg38.knownGene)

go1 <- enrichGO(as.data.frame(as.GRanges(anno_BR_NEG_minus_WT_NEG)[as.GRanges(anno_BR_NEG_minus_WT_NEG)$log2FoldChange > 
    0])$geneId, OrgDb = "org.Hs.eg.db", ont = "ALL", maxGSSize = 5000)
go2 <- enrichGO(as.data.frame(as.GRanges(anno_BR_NEG_minus_WT_NEG)[as.GRanges(anno_BR_NEG_minus_WT_NEG)$log2FoldChange < 
    0])$geneId, OrgDb = "org.Hs.eg.db", ont = "ALL", maxGSSize = 5000)
fwrite(as.data.frame(go1), "BR_NEG_minus_WT_NEG_GO1.tsv", sep="\t")
fwrite(as.data.frame(go2), "BR_NEG_minus_WT_NEG_GO2.tsv", sep="\t")

head(go1, 10) %>% dplyr::select(ID, Description, pvalue, p.adjust) %>% datatable(elementId = "goEle1")
head(go2, 10) %>% dplyr::select(ID, Description, pvalue, p.adjust) %>% datatable(elementId = "goEle2")

anno_BR_NEG_minus_WT_NEG_GRanges <- as.GRanges(anno_BR_NEG_minus_WT_NEG)
#anno_BR_NEG_minus_WT_NEG_GRanges_Up <- anno_BR_NEG_minus_WT_NEG[elementMetadata(anno_BR_NEG_minus_WT_NEG)$log2FoldChange > 0]
anno_BR_NEG_minus_WT_NEG_GRanges_Up <- anno_BR_NEG_minus_WT_NEG_GRanges[anno_BR_NEG_minus_WT_NEG@anno$log2FoldChange > 0]
# anno_BR_NEG_minus_WT_NEG_GRanges_Down <- anno_BR_NEG_minus_WT_NEG[elementMetadata(anno_BR_NEG_minus_WT_NEG)$log2FoldChange < 0]
anno_BR_NEG_minus_WT_NEG_GRanges_Down <- anno_BR_NEG_minus_WT_NEG_GRanges[anno_BR_NEG_minus_WT_NEG@anno$log2FoldChange < 0]
export.bed(anno_BR_NEG_minus_WT_NEG_GRanges_Up, "BR_NEG_minus_WT_NEG_Up.bed")
export.bed(anno_BR_NEG_minus_WT_NEG_GRanges_Down, "BR_NEG_minus_WT_NEG_Down.bed")

anno_BR_NEG_minus_WT_NEG_df <- as.data.frame(anno_BR_NEG_minus_WT_NEG)
write.table(anno_BR_NEG_minus_WT_NEG_df, "BR_NEG_minus_WT_NEG.csv", quote = FALSE, row.names = FALSE, sep = ",")

## BR_NEG_minus_BR_POS
BR_NEG_minus_BR_POS <- results(atacDDS, c("Group", "BR_NEG", "BR_POS"), format = "GRanges")
BR_NEG_minus_BR_POS <- BR_NEG_minus_BR_POS[order(BR_NEG_minus_BR_POS$pvalue)]
BR_NEG_minus_BR_POS

BR_NEG_minus_BR_POS <- BR_NEG_minus_BR_POS[(!is.na(BR_NEG_minus_BR_POS$padj) & BR_NEG_minus_BR_POS$padj < 0.05) & BR_NEG_minus_BR_POS %over% toOverLap, ]
makebedtable(BR_NEG_minus_BR_POS, "BR_NEG_minus_BR_POS.html", ".")
anno_BR_NEG_minus_BR_POS <- annotatePeak(BR_NEG_minus_BR_POS, TxDb = TxDb.Hsapiens.UCSC.hg38.knownGene)

go1 <- enrichGO(as.data.frame(as.GRanges(anno_BR_NEG_minus_BR_POS)[as.GRanges(anno_BR_NEG_minus_BR_POS)$log2FoldChange > 
    0])$geneId, OrgDb = "org.Hs.eg.db", ont = "ALL", maxGSSize = 5000)
go2 <- enrichGO(as.data.frame(as.GRanges(anno_BR_NEG_minus_BR_POS)[as.GRanges(anno_BR_NEG_minus_BR_POS)$log2FoldChange < 
    0])$geneId, OrgDb = "org.Hs.eg.db", ont = "ALL", maxGSSize = 5000)

head(go1, 10) %>% dplyr::select(ID, Description, pvalue, p.adjust) %>% datatable(elementId = "goEle1")
head(go2, 10) %>% dplyr::select(ID, Description, pvalue, p.adjust) %>% datatable(elementId = "goEle2")
fwrite(as.data.frame(go1), "BR_NEG_minus_BR_POS_GO1.tsv", sep="\t")
fwrite(as.data.frame(go2), "BR_NEG_minus_BR_POS_GO2.tsv", sep="\t")

anno_BR_NEG_minus_BR_POS_GRanges <- as.GRanges(anno_BR_NEG_minus_BR_POS)
#anno_BR_NEG_minus_BR_POS_GRanges_Up <- anno_BR_NEG_minus_BR_POS[elementMetadata(anno_BR_NEG_minus_BR_POS)$log2FoldChange > 0]
anno_BR_NEG_minus_BR_POS_GRanges_Up <- anno_BR_NEG_minus_BR_POS_GRanges[anno_BR_NEG_minus_BR_POS@anno$log2FoldChange > 0]
# anno_BR_NEG_minus_BR_POS_GRanges_Down <- anno_BR_NEG_minus_BR_POS[elementMetadata(anno_BR_NEG_minus_BR_POS)$log2FoldChange < 0]
anno_BR_NEG_minus_BR_POS_GRanges_Down <- anno_BR_NEG_minus_BR_POS_GRanges[anno_BR_NEG_minus_BR_POS@anno$log2FoldChange < 0]
export.bed(anno_BR_NEG_minus_BR_POS_GRanges_Up, "BR_NEG_minus_BR_POS_Up.bed")
export.bed(anno_BR_NEG_minus_BR_POS_GRanges_Down, "BR_NEG_minus_BR_POS_Down.bed")

anno_BR_NEG_minus_BR_POS_df <- as.data.frame(anno_BR_NEG_minus_BR_POS)
write.table(anno_BR_NEG_minus_BR_POS_df, "BR_NEG_minus_BR_POS.csv", quote = FALSE, row.names = FALSE, sep = ",")










library(MotifDb)
library(Biostrings)
library(BSgenome.Hsapiens.UCSC.hg19)

CTCF <- query(MotifDb, c("CTCF"))
CTCF <- as.list(CTCF)
myRes <- matchPWM(CTCF[[1]], BSgenome.Hsapiens.UCSC.hg19[["chr20"]])
toCompare <- GRanges("chr20", ranges(myRes))

read1 <- first(atacReads_Open)
read2 <- second(atacReads_Open)
Firsts <- resize(granges(read1), fix = "start", 1)
First_Pos_toCut <- shift(granges(Firsts[strand(read1) == "+"]), 4)
First_Neg_toCut <- shift(granges(Firsts[strand(read1) == "-"]), -5)


Seconds <- resize(granges(read2), fix = "start", 1)
Second_Pos_toCut <- shift(granges(Seconds[strand(read2) == "+"]), 4)
Second_Neg_toCut <- shift(granges(Seconds[strand(read2) == "-"]), -5)

test_toCut <- c(First_Pos_toCut, First_Neg_toCut, Second_Pos_toCut, Second_Neg_toCut)
cutsCoverage <- coverage(test_toCut)
cutsCoverage20 <- cutsCoverage["chr20"]
CTCF_Cuts_open <- regionPlot(cutsCoverage20, testRanges = toCompare, style = "point", 
    format = "rlelist", distanceAround = 500)
plotRegion(CTCF_Cuts_open, outliers = 0.001) + ggtitle("NucFree Cuts Centred on CTCF")