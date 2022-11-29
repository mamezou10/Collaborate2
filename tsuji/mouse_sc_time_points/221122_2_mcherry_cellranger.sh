
import pandas as pd

# mCherryにマッピング
# get fasta
#https://www.ncbi.nlm.nih.gov/nuccore/AY678264 -> mCherry.txt

# cp mCherry.txt mCherry.fa
# echo -e ">test\nAAAGGGGGCCATTGGG" >> mCherry.fa


# make GTF

pd.DataFrame({
    "seqname": ["test", "lcl|AY678264.1_cds_AAV52164.1_1", "lcl|AY678264.1_cds_AAV52164.1_1", "test"],
    "source": [".", ".", ".", "."],
    "feature": ["exon", "exon", "exon", "exon"],
    "start": ["1", "1", "1", "1"],
    "end": ["11", "711", "711", "11"],
    "score": [".", ".", ".", "."],
    "strand": ["+", "+", "-", "-"],
    "frame": [".", ".", ".", "."],
    "attribute": ['gene_id "test1";transcript_id "test1"', 'gene_id "mCherry";transcript_id "mCherry_f"', 'gene_id "mCherry";transcript_id "mCherry_r"', 'gene_id "test1";transcript_id "test2"']
}).to_csv("mCherry.gtf", sep="\t", header=None, index=None)


cellranger mkref \
  --genome=mCherry_genome \
  --fasta=mCherry.fa \
  --genes=mCherry.gtf


cellranger count --id=tsuji_day7_mCherry --chemistry SC5P-R2 \
    --fastqs=/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq \
    --sample=CMT167_1G2_D7_5DE \
    --transcriptome=mCherry_genome

cellranger count --id=tsuji_day4_mCherry --chemistry SC5P-R2 \
    --fastqs=/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq \
    --sample=CMT167_1G2_D4_5DE \
    --transcriptome=mCherry_genome
    
cellranger count --id=tsuji_day10_mCherry --chemistry SC5P-R2 \
    --fastqs=/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq \
    --sample=CMT167_1G2_D10_5DE \
    --transcriptome=mCherry_genome
    
cellranger count --id=tsuji_KO_day7_mCherry --chemistry SC5P-R2 \
    --fastqs=/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq \
    --sample=CMT167_DK1_D7_5DE \
    --transcriptome=mCherry_genome
