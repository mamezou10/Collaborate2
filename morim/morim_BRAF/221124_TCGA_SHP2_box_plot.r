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

dir.create("/mnt/Donald/morim_BRAF/221124")
setwd("/mnt/Donald/morim_BRAF/221124")


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






# 生存日数などのデータ
clin <- GDCquery_clinic("TCGA-COAD", "clinical")

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
exp <- fread("../final.tsv") ## ＊もともとfull_joinしてるのでWTも含む
# exp$BRAF_mut

df <- coad.merge %>% full_join(exp, by="submitter_id")
df <- df  %>% mutate(BRAF_mut=case_when(
                submitter_id %in% patients_braf$submitter_id ~ BRAF_mut,
                submitter_id %in% patients_wt$submitter_id ~ "WT",
                TRUE ~ "NA"))

df <- df %>% distinct(submitter_id, .keep_all=TRUE)



#df <- coad.merge %>% inner_join(exp, by="submitter_id")
g <-  ggplot(df[df$BRAF_mut!="NA"&df$BRAF_mut!="none", ], aes(x = BRAF_mut, y = PTPN11, fill=BRAF_mut)) + 
        geom_boxplot() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
        geom_signif(comparisons = list(c("p.V600E", "WT")), y_position = 17000, vjust = 0.2) #+#, test = "t.test") , map_signif_level=TRUE, textsize=9, family="serif"

ggsave("box_colored_mod.png") 

# > t.test(df[df$BRAF_mut=="WT",]$MUC1, df[df$BRAF_mut=="p.V600E",]$MUC1)

#         Welch Two Sample t-test

# data:  df[df$BRAF_mut == "WT", ]$MUC1 and df[df$BRAF_mut == "p.V600E", ]$MUC1
# t = -2.5383, df = 45.871, p-value = 0.0146
# alternative hypothesis: true difference in means is not equal to 0
# 95 percent confidence interval:
#  -2342.5193  -270.3351
# sample estimates:
# mean of x mean of y 
#  1740.013  3046.441 