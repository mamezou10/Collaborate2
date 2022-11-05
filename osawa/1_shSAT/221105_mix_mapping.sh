(kb-tools)
import os
os.chdir("/mnt/Daisy/kb_human_mouse")

%%time
wget ftp://ftp.ensembl.org/pub/release-98/fasta/mus_musculus/dna/Mus_musculus.GRCm38.dna.primary_assembly.fa.gz
wget ftp://ftp.ensembl.org/pub/release-98/gtf/mus_musculus/Mus_musculus.GRCm38.98.gtf.gz
wget ftp://ftp.ensembl.org/pub/release-98/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz
wget ftp://ftp.ensembl.org/pub/release-98/gtf/homo_sapiens/Homo_sapiens.GRCh38.98.gtf.gz

kb ref -i mixed_index.idx -g mixed_t2g.txt -f1 mixed_cdna.fa \
Mus_musculus.GRCm38.dna.primary_assembly.fa.gz,Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz Mus_musculus.GRCm38.98.gtf.gz,Homo_sapiens.GRCh38.98.gtf.gz


%%time
kb ref -i mixed_index.idx -g mixed_t2g.txt -f1 mixed_cdna.fa \
Mus_musculus.GRCm38.dna.primary_assembly.fa.gz,Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz \
Mus_musculus.GRCm38.98.gtf.gz,Homo_sapiens.GRCh38.98.gtf.gz


# %%time
# kb count -i mixed_index.idx -g mixed_t2g.txt -x SMARTSEQ2 --parity paired -o output --h5ad -t 2 --gene-names --report --verbose \
# /mnt/ssd8t/Projects/osawa/SAT1/SC_1/trim/SC_1_1_val_1.fq.gz /mnt/ssd8t/Projects/osawa/SAT1/SC_1/trim/SC_1_2_val_2.fq.gz

for i in SC_1 SC_2 SC_3 shSAT1_2_1 shSAT1_2_2 shSAT1_2_3
do
    /home/hirose/.local/lib/python3.8/site-packages/kb_python/bins/linux/kallisto/kallisto quant -i mixed_index.idx -o /mnt/ssd8t/Projects/osawa/SAT1/${i}/mix \
    /mnt/ssd8t/Projects/osawa/SAT1/${i}/trim/${i}_1_val_1.fq.gz /mnt/ssd8t/Projects/osawa/SAT1/${i}/trim/${i}_2_val_2.fq.gz
done
