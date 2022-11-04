library(GEOquery)
library(dplyr)
library(pheatmap)
library(survival)
library(survminer)
# TCGA
library(RTCGA)
library(RTCGA.clinical)
library(RTCGA.mRNA)
library(RTCGA.mutations)
library(tidyverse)
library(data.table)
library(TCGAbiolinks)
library(janitor)
library(ggplot2)
library(ggpubr)

setwd("/mnt/Donald/morim_BRAF")

# 変異があるかのデータ

library(maftools)

query <- GDCquery(
    project = "TCGA-COAD", 
    data.category = "Simple Nucleotide Variation", 
    access = "open", 
    #legacy = FALSE, 
    # sample.type = "Primary Tumor",
    # data.type = "Annotated Somatic Mutation", 
    #workflow.type = "Aliquot Ensemble Somatic Variant Merging and Masking"
)

query.2 = query
tmp     = query.2$results[[1]]
tmp     = tmp[which(!duplicated(tmp$cases)),]
query.2$results[[1]] = tmp
GDCdownload(query.2)
maf <- GDCprepare(query.2)

# GDCdownload(query)
# maf <- GDCprepare(query)

maf$submitter_id=str_sub(maf$Tumor_Sample_Barcode, start=1,end=12)
# maf <-maf %>% filter(HGVSp_Short=="p.V600E")
# maf <-maf %>% filter(HGVSp_Short=="p.V640E")

maf <-maf %>% filter(Hugo_Symbol=="BRAF")

# 生存日数などのデータ
clin <- GDCquery_clinic("TCGA-COAD", "clinical")

# 変異のあるpatientのまとめ
coad.merge <- inner_join(maf, clin, by="submitter_id") %>% 
                as.data.frame() %>% 
                #filter(HGVSp_Short=="p.V600E") %>% 
                select_if(~sum(!is.na(.)) > 0)
## https://www.biostars.org/p/215175/ ファイル名との紐付け
meta <- fread("File_metadata.txt") %>% 
        select("cases.0.submitter_id", "file_name" )
coad_merge <- inner_join(meta, coad.merge, by=c("cases.0.submitter_id"="submitter_id")) #%>% 
    # select(-c(Entrez_Gene_Id,dbSNP_Val_Status,Tumor_Validation_Allele1,Tumor_Validation_Allele2,Validation_Method, Sequencer, Tumor_Sample_UUID,Matched_Norm_Sample_UUID, all_effects, Gene,  Feature, Feature_type, DOMAINS,tumor_bam_uuid, normal_bam_uuid, GDC_FILTER, COSMIC, MC3_Overlap,
    # GDC_Validation_Status, updated_datetime, diagnosis_id, treatments_pharmaceutical_treatment_id, treatments_radiation_treatment_id,exposure_id, demographic_id, src_vcf_id, ))

# 変異のあるpatientのファイル番号
# braf_pati <- coad_merge$file_name




# Expression
#  "/Volumes/Bambi/Projects/morim_braf/analysis/220216"から
exp <- fread("final.tsv")
exp <- exp %>% mutate(MYC_state=case_when(
    MYC >= mean(exp$MYC) ~ "high",
    TRUE ~ "low",)) 

thre <- quantile(exp$MYC, c(0,0.25,0.5,0.75,1.0))
exp$MYC_q <- cut(exp$MYC, thre,
                labels = c("0-25%","25-50%","50-75%","75-100%"),
                include.lowest = TRUE)

df <- coad.merge %>% filter(HGVSp_Short=="p.V640E") %>% inner_join(exp, by="submitter_id")
TCGAanalyze_survival(df, "MYC_q",
        main = "TCGA Set\n MYC_q\n BRAF mutatnt",height = 10, width=10, filename = "survival_MYC_q.png")


exp$NOTCH_q <- cut(exp$NOTCH1, thre,
                labels = c("0-25%","25-50%","50-75%","75-100%"),
                include.lowest = TRUE)

df <- coad.merge %>% filter(HGVSp_Short=="p.V640E") %>% inner_join(exp, by="submitter_id")
TCGAanalyze_survival(df, "NOTCH_q",
        main = "TCGA Set\n NOTCH_q\n BRAF mutatnt",height = 10, width=10, filename = "survival_NOTCH_q.png")

exp <- exp %>% mutate(NOTCH1_state=case_when(
    NOTCH1 >= mean(exp$NOTCH1) ~ "high",
    TRUE ~ "low",)) 
df <- coad.merge %>% filter(HGVSp_Short=="p.V640E") %>% inner_join(exp, by="submitter_id")
TCGAanalyze_survival(df, "NOTCH1_state",
        main = "TCGA Set\n NOTCH_q\n BRAF mutatnt",height = 10, width=10, filename = "survival_NOTCH_50.png")









## id convert
id2symbol <- fread("/Volumes/Bambi/Projects/morim_braf/custom.txt") %>%
    select("Ensembl gene ID", "Approved symbol")
exp <- inner_join(id2symbol, kari, by=c("Ensembl gene ID"="gene"))

exp_braf <- exp %>% select(`Approved symbol`, braf_pati) %>% 
            t() %>% janitor::row_to_names(1) %>% as.data.frame()
exp_braf <- data.frame(lapply(exp_braf, as.numeric))

ggplot(exp_braf, aes(MUC1, y=MICA)) + geom_point()


## targetの遺伝子たちでplot
genes <- c("MUC1", "CCNA2",
"CCNB1",
"CCNB2",
"CCND1",
"MYC",
"SMARCA4",
"ARID1A",
"PBRM1",
"NOTCH1",
"NANOG",
"SOX9",
"E2F1",
"CDKN1A")

for(i in genes){
    print(i %in% colnames(exp_braf))
}

# BRAF patientのみ
exp_braf_targ_genes <- exp_braf %>% 
            select(genes) %>% 
            pivot_longer(cols=-MUC1)

p <- ggplot(exp_braf_targ_genes, aes(MUC1, y=value)) 
p <- p + geom_point(alpha=0.5) + facet_wrap(~ name, scales = "free") 
p <- p + stat_cor(label.y.npc="top", label.x.npc = "left", method = "pearson", size=2.5)
p <- p + geom_smooth(method='lm', formula= y~x)
p

ggsave(file = "scatter.pdf", plot = p)



## 全patient ver
exp_all_patie <- exp %>% select(-`Ensembl gene ID`) %>% 
            t() %>% janitor::row_to_names(1) %>% as.data.frame()
exp_all_patie <- data.frame(lapply(exp_all_patie, as.numeric))

exp_all_patie_targ_genes <- exp_all_patie %>% 
            select(genes) %>% 
            pivot_longer(cols=-MUC1)
p <- ggplot(exp_all_patie_targ_genes, aes(MUC1, y=value)) 
p <- p + geom_point(alpha=0.5) + facet_wrap(~ name, scales = "free") 
p <- p + stat_cor(label.y.npc="top", label.x.npc = "left", method = "pearson", size=2.5)
p <- p + geom_smooth(method='lm', formula= y~x)
p
ggsave(file = "scatter_all_patient.pdf", plot = p)
