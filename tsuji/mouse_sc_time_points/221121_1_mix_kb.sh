
# index

# kb ref -i mixed_index.idx -g mixed_t2g.txt -f1 mixed_cdna.fa -f2 mixed_intron.fa \
# --workflow lamanno -c1 spliced_t2g.txt -c2 unspliced_t2g.txt --overwrite \
# Mus_musculus.GRCm38.dna.primary_assembly.fa.gz,Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz \
# Mus_musculus.GRCm38.98.gtf.gz,Homo_sapiens.GRCh38.98.gtf.gz


## mix kallisto
# %%time
kb count -i mixed_index.idx -g mixed_t2g.txt -x 10XV2 -o tsuji_day10 --h5ad --gene-names --verbose \
# --filter bustools -c1 spliced_t2g.txt -c2 unspliced_t2g.txt --workflow lamanno \
/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq/CMT167_1G2_D10_5DE_S3_L001_R1_001.fastq.gz \
/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq/CMT167_1G2_D10_5DE_S3_L001_R2_001.fastq.gz

kb count -i mixed_index.idx -g mixed_t2g.txt -x 10XV2 -o tsuji_day4 --h5ad -t 2 --gene-names --verbose \
/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq/CMT167_1G2_D4_5DE_S4_L001_R1_001.fastq.gz \
/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq/CMT167_1G2_D4_5DE_S4_L001_R2_001.fastq.gz

kb count -i mixed_index.idx -g mixed_t2g.txt -x 10XV2 -o tsuji_day7 --h5ad -t 2 --gene-names --verbose \
/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq/CMT167_1G2_D7_5DE_S2_L001_R1_001.fastq.gz \
/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq/CMT167_1G2_D7_5DE_S2_L001_R2_001.fastq.gz

kb count -i mixed_index.idx -g mixed_t2g.txt -x 10XV2 -o tsuji_KO_day7 --h5ad -t 2 --gene-names --verbose \
/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq/CMT167_DK1_D7_5DE_S1_L001_R1_001.fastq.gz \
/mnt/Daisy/tsuji_sc_mouse/F4710_220927_122647_fastq/CMT167_DK1_D7_5DE_S1_L001_R2_001.fastq.gz




