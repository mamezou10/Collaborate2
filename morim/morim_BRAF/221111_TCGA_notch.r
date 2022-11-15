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
library(jsonlite)

install.packages("RSJSONIO") 
vignette("RSJSONIO")

setwd("/mnt/Donald/morim_BRAF")

# 変異があるかのデータ

library(maftools)

query <- GDCquery(
    project = "TCGA-COAD", 
    data.category = "Simple Nucleotide Variation", 
    access = "open", 
    # legacy = TRUE, #/mnt/Donald/morim_BRAF/kari.png
    # sample.type = "Primary Tumor",
    # data.type = "Masked Somatic Mutation", 
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



# GDCquery_Maf("COAD", pipelines = "muse")


patients_wt  <- maf %>% filter(Hugo_Symbol!="BRAF") %>% mutate(HGVSp_Short= "WT") %>% distinct(submitter_id, HGVSp_Short)
patients_braf <- maf %>% filter(Hugo_Symbol=="BRAF") %>% mutate(HGVSp_Short= HGVSp_Short)%>% distinct(submitter_id, HGVSp_Short)

mt_df <- rbind(patients_wt, patients_braf)




# setdiff(patients_all, patients_braf)

# maf <- maf %>% mutate(brafmt = case_when(
#     submitter_id %in% patients_braf$submitter_id ~ "braf_mt",
#     TRUE ~ "braf_wt"))

# mt_df <- maf %>% mutate(HGVSp_Short = case_when(
#     brafmt == "braf_wt" ~ "WT",
#     TRUE ~ HGVSp_Short))%>% distinct(submitter_id, brafmt, HGVSp_Short )



# 生存日数などのデータ
clin <- GDCquery_clinic("TCGA-COAD", "clinical")

# 変異のあるpatientのまとめ
# coad.merge <- inner_join(maf, clin, by="submitter_id") %>% 
#                 as.data.frame() %>% 
#                 #filter(HGVSp_Short=="p.V600E") %>% 
#                 select_if(~sum(!is.na(.)) > 0)
coad.merge <- inner_join(mt_df, clin, by="submitter_id") %>% 
                as.data.frame() %>% 
                #filter(HGVSp_Short=="p.V600E") %>% 
                select_if(~sum(!is.na(.)) > 0)

## https://www.biostars.org/p/215175/ ファイル名との紐付け
meta <- fread("File_metadata.txt") %>% 
        select("cases.0.submitter_id", "file_name" )
# coad_merge <- inner_join(meta, coad.merge, by=c("cases.0.submitter_id"="submitter_id")) #%>% 
    # select(-c(Entrez_Gene_Id,dbSNP_Val_Status,Tumor_Validation_Allele1,Tumor_Validation_Allele2,Validation_Method, Sequencer, Tumor_Sample_UUID,Matched_Norm_Sample_UUID, all_effects, Gene,  Feature, Feature_type, DOMAINS,tumor_bam_uuid, normal_bam_uuid, GDC_FILTER, COSMIC, MC3_Overlap,
    # GDC_Validation_Status, updated_datetime, diagnosis_id, treatments_pharmaceutical_treatment_id, treatments_radiation_treatment_id,exposure_id, demographic_id, src_vcf_id, ))


coad.merge <- inner_join(coad.merge, meta, by=c("submitter_id"="cases.0.submitter_id")) 


library(ggsignif)

# Expression
exp <- fread("final.tsv")
# exp$BRAF_mut

df <- coad.merge %>% full_join(exp, by="submitter_id")
df <- df  %>% mutate(BRAF_mut=case_when(
                submitter_id %in% patients_braf$submitter_id ~ BRAF_mut,
                submitter_id %in% patients_wt$submitter_id ~ "WT",
                TRUE ~ "NA"))

df <- df %>% distinct(submitter_id, .keep_all=TRUE)
# GDCportalからcountデータをDL（biolinkだと進まない）
# files <- list.files("/mnt/Bambi/Projects/morim_braf/analysis/220206",recursive=T, pattern=".FPKM.txt",full.names=T)

# res_df <- data.frame(gene=c("kari"))
# for(i in 1:length(files)){
#     df <- fread(files[i])
#     colnames(df) <- c("gene", basename(files[i]))
#     res_df <- full_join(res_df, df, by="gene")
# }
# kari <- res_df %>% separate(gene, c("gene", "suff"), sep="\\.") %>% select(-suff)
# # # expression(TPM)
# # res_df <- res_df[-1,]
# # rs <- rowSums(res_df[,-1])
# # res_tpm <- res_df %>% mutate_if(is.numeric, funs(. / rs * 1000000)) 
# # kari <- res_tpm %>% separate(gene, c("gene", "suff"), sep="\\.") %>% select(-suff)

# ## id convert
# id2symbol <- fread("/mnt/Bambi/Projects/morim_braf/custom.txt") %>%
#     select("Ensembl gene ID", "Approved symbol")
# exp <- inner_join(id2symbol, kari, by=c("Ensembl gene ID"="gene")) %>% 
#         select(-`Ensembl gene ID`) %>% 
#         column_to_rownames(., var = "Approved symbol") %>% 
#         t() %>% as.data.frame() 

# ## 最終のデータフレーム
# df <- exp %>% 
#     rownames_to_column(var="file_name") %>% 
#     full_join(coad.merge, by="file_name" ) %>%
#     select(file_name, submitter_id, everything()) 
    


#df <- coad.merge %>% inner_join(exp, by="submitter_id")
g <-  ggplot(df[df$BRAF_mut!="NA"&df$BRAF_mut!="none", ], aes(x = BRAF_mut, y = MUC1)) + 
        geom_boxplot() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        geom_signif(comparisons = list(c("p.V600E", "WT")), map_signif_level=TRUE) +#, test = "t.test") 
        geom_signif(comparisons = list(c("p.D594N", "WT")), y_position = 18000) 

ggsave("kari.png") 

t.test(df[df$HGVSp_Short=="WT"]$MUC1, df[df$HGVSp_Short=="p.V640E"]$MUC1)



exp <- df

thre <- quantile(exp$NOTCH1, c(0,0.25,0.5,0.75,1.0))
thre_half <- quantile(exp$NOTCH1, c(0,0.5,1.0))

exp$NOTCH_q <- cut(exp$NOTCH1, thre,
                labels = c("0-25%","25-50%","50-75%","75-100%"),
                include.lowest = TRUE)

exp$NOTCH_half <- cut(exp$NOTCH1, thre_half,
                labels = c("low","high"),
                include.lowest = TRUE)

df_surv <- exp

TCGAanalyze_survival(df_surv, "NOTCH_q",
        main = "TCGA Set\n NOTCH_q\n BRAF mutatnt",height = 10, width=10, filename = "survival_NOTCH_q_2.png")

TCGAanalyze_survival(df_surv[df_surv$BRAF_mut=="WT",], "NOTCH_q",
        main = "TCGA Set\n NOTCH_q\n BRAF mutatnt",height = 10, width=10, filename = "survival_NOTCH_q_wt.png")

TCGAanalyze_survival(df_surv[df_surv$BRAF_mut=="p.V600E",], "NOTCH_q",
        main = "TCGA Set\n NOTCH_q\n BRAF mutatnt",height = 10, width=10, filename = "survival_NOTCH_q_mt.png")


df_surv_censor <- df_surv %>% 
                mutate(vital_status = ifelse(days_to_death > 1000, "Alive", vital_status)) %>%
                mutate(days_to_death = ifelse(days_to_death > 1000, 1000, days_to_death))  %>%
                mutate(days_to_last_follow_up = ifelse(days_to_last_follow_up > 1000, 1000, days_to_last_follow_up)) %>%  
                mutate(days = ifelse(days > 1000, 1000, days))  


TCGAanalyze_survival(df_surv_censor[df_surv_censor$BRAF_mut=="WT",], "NOTCH_half", risk.table=FALSE, 
        main = "TCGA Set\n NOTCH_q\n BRAF mutatnt",height = 10, width=10, filename = "survival_NOTCH_half_wt.png")

TCGAanalyze_survival(df_surv_censor[df_surv_censor$BRAF_mut=="p.V600E",], "NOTCH_half", 
        main = "TCGA Set\n NOTCH_q\n BRAF mutatnt",height = 10, width=10, filename = "survival_NOTCH_half_mt.png")


df_surv_censor_braf <- df_surv_censor[df_surv_censor$BRAF_mut=="p.V600E", ] %>%
                        mutate(NOTCH_half_braf=case_when(
                                NOTCH1 >= median(NOTCH1) ~ "high",
                                TRUE ~ "low",)) 

TCGAanalyze_survival(df_surv_censor_braf, "NOTCH_half_braf", 
        main = "TCGA Set\n NOTCH_q\n BRAF mutatnt",height = 10, width=10, filename = "survival_NOTCH_half_mt_braf.png")















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
