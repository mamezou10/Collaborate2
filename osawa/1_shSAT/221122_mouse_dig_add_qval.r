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
# fwrite(data.frame(res), "shSAT1_vs_SC.tsv", sep="\t", row.names = TRUE)
# fwrite(as.data.frame(ddsHTSeq@assays@data$counts), "shSAT1_SC_counts.tsv", sep="\t", row.names = TRUE)

mouse2human <- fread("/home/hirose/Documents/Collaborate/gmts/mouse2human.txt")

df <- rownames_to_column(as.data.frame(ddsHTSeq@assays@data$counts), "mouse") 
df <- df %>% inner_join(mouse2human, by="mouse")
df <- df %>% group_by(human) %>% select(-c("mouse")) %>% summarise_all(sum)

dec_df <- data.frame(df, row.names=df$human) %>% select(-"human")



res <- immunedeconv::deconvolute(dec_df, "quantiseq")
hm_df <- res[,2:7]
hm_df[ hm_df > 0.3 ] <- 0.3
rownames(hm_df) <- res$cell_type
pdf("/mnt/ssd8t/Projects/osawa/SAT1/mouse_quantiseq_0.3.pdf")
pheatmap(hm_df, fontsize_row = 10, cluster_col=F, cluster_row=T); dev.off()


hm_df[ hm_df > 0.2 ] <- 0.2
rownames(hm_df) <- res$cell_type
pdf("/mnt/ssd8t/Projects/osawa/SAT1/mouse_quantiseq_0.2.pdf")
pheatmap(hm_df, fontsize_row = 10, cluster_col=F, cluster_row=T); dev.off()


library(ggsignif)
res <- immunedeconv::deconvolute(dec_df, "quantiseq") %>% pivot_longer(-"cell_type") %>% mutate(name = str_sub(name, 1,2))

g <- ggplot(res, aes(x=name, y=value, fill=name)) + 
    geom_boxplot() + 
    facet_wrap(cell_type ~ ., scales = "free") +
    #geom_jitter() +
    geom_signif(comparisons = list(c("SC", "sh")), test.args=list(alternative = "two.sided", var.equal = TRUE, paired=FALSE), test = "t.test", textsize=2)#, family="serif", y_position = 17000, vjust = 0.2) +#,) map_signif_level=TRUE, 

ggsave("/mnt/ssd8t/Projects/osawa/SAT1/box.pdf")


# > t.test(c(0.014,0.18,0.183), c(0,0.102,0))

#         Welch Two Sample t-test

# data:  c(0.014, 0.18, 0.183) and c(0, 0.102, 0)
# t = 1.4021, df = 3.3038, p-value = 0.2475
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#  -0.1059770  0.2893104
# sample estimates:
# mean of x mean of y 
# 0.1256667 0.0340000 





source("/mnt/244hirose/Scripts/gmts/my_fisher.R")

obj <- my_fisher("/mnt/ssd8t/Projects/osawa/SAT1/genesets_mouse.gmt", "/mnt/244hirose/Scripts/gmts/impala.gmt")

## qvalue
i=1
pval <- t(obj$p.value)
gene <- t(obj$genes)
for (i in 1:ncol(pval)){
    pvalue <- pval[,i]
    qvalue <- p.adjust(10^-(pvalue), method="BH")
    genes <- gene[,i]
    # qvalue <- -log10(qvalue)
    p_df = data.frame(minus_log10_p = pvalue, pathway=names(pvalue))
    q_df = data.frame(qval_BH=qvalue, pathway=names(pvalue))
    q_df["minus_log10_q"] = -log10(q_df$qval_BH)
    genes_df = data.frame(genes=genes, pathway=names(pvalue))
    df = p_df %>% inner_join(q_df, by="pathway") %>% 
                inner_join(genes_df, by="pathway") %>% 
                select(pathway, minus_log10_p, everything()) %>% 
                arrange(qval_BH)
    out_file <- paste0("/mnt/ssd8t/Projects/osawa/SAT1/impala_", colnames(pval)[i], ".txt")
    fwrite(df, out_file, sep="\t")
}


obj <- my_fisher("/mnt/ssd8t/Projects/osawa/SAT1/genesets_mouse.gmt", "/mnt/244hirose/Scripts/gmts/msigdb.v7.4.symbols.gmt")
i=1
pval <- t(obj$p.value)
gene <- t(obj$genes)
for (i in 1:ncol(pval)){
    pvalue <- pval[,i]
    qvalue <- p.adjust(10^-(pvalue), method="BH")
    genes <- gene[,i]
    # qvalue <- -log10(qvalue)
    p_df = data.frame(minus_log10_p = pvalue, pathway=names(pvalue))
    q_df = data.frame(qval_BH=qvalue, pathway=names(pvalue))
    q_df["minus_log10_q"] = -log10(q_df$qval_BH)
    genes_df = data.frame(genes=genes, pathway=names(pvalue))
    df = p_df %>% inner_join(q_df, by="pathway") %>% 
                inner_join(genes_df, by="pathway") %>% 
                select(pathway, minus_log10_p, everything()) %>% 
                arrange(qval_BH)
    out_file <- paste0("/mnt/ssd8t/Projects/osawa/SAT1/msig", colnames(pval)[i], ".txt")
    fwrite(df, out_file, sep="\t")
}
