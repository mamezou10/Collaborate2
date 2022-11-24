
cd /mnt/Daisy/tsuji_sc_mouse/


cellranger count --id=tsuji_human_day4 \
   --fastqs=/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq \
   --sample=CMT167_1G2_D4_5DE \
   --transcriptome=/mnt/Daisy/ref_cellranger/refdata-gex-GRCh38-2020-A

cellranger count --id=tsuji_human_day7 \
   --fastqs=/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq \
   --sample=CMT167_1G2_D7_5DE \
   --transcriptome=/mnt/Daisy/ref_cellranger/refdata-gex-GRCh38-2020-A

cellranger count --id=tsuji_human_day10 \
   --fastqs=/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq \
   --sample=CMT167_1G2_D10_5DE \
   --transcriptome=/mnt/Daisy/ref_cellranger/refdata-gex-GRCh38-2020-A

cellranger count --id=tsuji_human_KO_day7 \
   --fastqs=/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq \
   --sample=CMT167_DK1_D7_5DE \
   --transcriptome=/mnt/Daisy/ref_cellranger/refdata-gex-GRCh38-2020-A












