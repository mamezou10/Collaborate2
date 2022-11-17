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

library(survival)
library(survminer)

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


patients_wt  <- maf %>% filter(Hugo_Symbol!="BRAF") %>% mutate(HGVSp_Short= "WT") %>% distinct(submitter_id, HGVSp_Short)
patients_braf <- maf %>% filter(Hugo_Symbol=="BRAF") %>% mutate(HGVSp_Short= HGVSp_Short)%>% distinct(submitter_id, HGVSp_Short)

mt_df <- rbind(patients_wt, patients_braf)

# 生存日数などのデータ
clin <- GDCquery_clinic("TCGA-COAD", "clinical")

# 変異のあるpatientのまとめ

coad.merge <- inner_join(mt_df, clin, by="submitter_id") %>% 
                as.data.frame() %>% 
                #filter(HGVSp_Short=="p.V600E") %>% 
                select_if(~sum(!is.na(.)) > 0)

## https://www.biostars.org/p/215175/ ファイル名との紐付け
meta <- fread("File_metadata.txt") %>% 
        select("cases.0.submitter_id", "file_name" )

coad.merge <- inner_join(coad.merge, meta, by=c("submitter_id"="cases.0.submitter_id")) 


library(ggsignif)

# Expression
exp <- fread("final.tsv") ## ＊もともとfull_joinしてるのでWTも含む
# exp$BRAF_mut

df <- coad.merge %>% full_join(exp, by="submitter_id")
df <- df  %>% mutate(BRAF_mut=case_when(
                submitter_id %in% patients_braf$submitter_id ~ BRAF_mut,
                submitter_id %in% patients_wt$submitter_id ~ "WT",
                TRUE ~ "NA"))

df <- df %>% distinct(submitter_id, .keep_all=TRUE)


# KMplot
exp <- df
# thre <- quantile(exp$NOTCH1, c(0,0.25,0.5,0.75,1.0))
thre_half <- quantile(exp$NOTCH1, c(0,0.5,1.0))

exp$NOTCH_half <- cut(exp$NOTCH1, thre_half,
                labels = c("low","high"),
                include.lowest = TRUE)


df_surv <- exp %>% mutate(days=case_when(
    event == 0  ~ days_to_last_follow_up,
    TRUE ~ days_to_death)) 

# 3年で打ち切り
df_surv <- df_surv %>% mutate(event = ifelse(days>1100, 0, event)) 
df_surv <- df_surv %>% mutate(days = ifelse(days>1100, 1100, days)) 


#logr <- survdiff(Surv(days, event) ~ NOTCH_half, data=df_surv[df_surv$BRAF_mut=="WT",])


# KM WT
sfit <- survfit(Surv(days, event) ~ NOTCH_half, data=df_surv[df_surv$BRAF_mut=="WT",])
summary(sfit)
g <- ggsurvplot(sfit, conf.int=FALSE, pval=TRUE, risk.table=TRUE, xscale = "d_m" ,break.time.by=365.25*0.5, xlim=c(0,1100),
           legend.labs=c("low", "high"), legend.title="NOTCH1",  xlab="month",
           palette=c("dodgerblue2", "orchid2"), 
           title="Kaplan-Meier Curve for COAD", 
           risk.table.height=.3)

png("survival_NOTCH1_wt_final.png")
print(g, newpage = FALSE)
dev.off()


# KM mut
sfit <- survfit(Surv(days, event) ~ NOTCH_half, data=df_surv[df_surv$BRAF_mut=="p.V600E",])
summary(sfit)

g <- ggsurvplot(sfit, conf.int=FALSE, pval=TRUE, risk.table=TRUE, xscale = "d_m" ,break.time.by=365.25*0.5, xlim=c(0,1100),
           legend.labs=c("low", "high"), legend.title="NOTCH1",  xlab="month",
           palette=c("dodgerblue2", "orchid2"), 
           title="Kaplan-Meier Curve for COAD with mut", 
           risk.table.height=.3)

png("survival_NOTCH1_mt_final.png")
print(g, newpage = FALSE)
dev.off()


# 4yrs version
df_surv <- exp %>% mutate(days=case_when(
    event == 0  ~ days_to_last_follow_up,
    TRUE ~ days_to_death)) 

# 3年で打ち切り
df_surv <- df_surv %>% mutate(event = ifelse(days>1500, 0, event)) 
df_surv <- df_surv %>% mutate(days = ifelse(days>1500, 1500, days)) 

# KM WT
sfit <- survfit(Surv(days, event) ~ NOTCH_half, data=df_surv[df_surv$BRAF_mut=="WT",])
summary(sfit)
g <- ggsurvplot(sfit, conf.int=FALSE, pval=TRUE, risk.table=TRUE, xscale = "d_m" ,break.time.by=365.25*0.5, xlim=c(0,1500),
           legend.labs=c("low", "high"), legend.title="NOTCH1",  xlab="month",
           palette=c("dodgerblue2", "orchid2"), 
           title="Kaplan-Meier Curve for COAD", 
           risk.table.height=.3)

png("survival_NOTCH1_wt_4yrs_final.png")
print(g, newpage = FALSE)
dev.off()


# KM mut
sfit <- survfit(Surv(days, event) ~ NOTCH_half, data=df_surv[df_surv$BRAF_mut=="p.V600E",])
summary(sfit)

g <- ggsurvplot(sfit, conf.int=FALSE, pval=TRUE, risk.table=TRUE, xscale = "d_m" ,break.time.by=365.25*0.5, xlim=c(0,1500),
           legend.labs=c("low", "high"), legend.title="NOTCH1",  xlab="month",
           palette=c("dodgerblue2", "orchid2"), 
           title="Kaplan-Meier Curve for COAD with mut", 
           risk.table.height=.3)

png("survival_NOTCH1_mt_4yrs_final.png")
print(g, newpage = FALSE)
dev.off()
