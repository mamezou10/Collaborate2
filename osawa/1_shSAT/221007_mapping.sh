wd="/mnt/Chip/Projects/osawa/SAT1/"

## まずヒトにmapping
mkdir /mnt/Bambi/hg38_from_shimam_sensei/star_index
STAR --runMode genomeGenerate \
     --genomeDir /mnt/Bambi/hg38_from_shimam_sensei/star_index \
     --genomeFastaFiles /mnt/Bambi/hg38_from_shimam_sensei/hg38.fa 
     # --limitGenomeGenerateRAM 3400000000

## 次にマウスにマッピング
wget http://igenomes.illumina.com.s3-website-us-east-1.amazonaws.com/Mus_musculus/UCSC/mm10/Mus_musculus_UCSC_mm10.tar.gz

mkdir /mnt/Bambi/mm10/star_index
STAR --runMode genomeGenerate \
     --genomeDir /mnt/Bambi/mm10/star_index \
     --genomeFastaFiles /mnt/Bambi/mm10/Mus_musculus/UCSC/mm10/Sequence/WholeGenomeFasta/genome.fa


for i in SC_1 SC_2 SC_3 shSAT1_2_1 shSAT1_2_2 shSAT1_2_3
do
     mkdir -p /mnt/ssd8t/Projects/osawa/SAT1/${i}/trim
     cd /mnt/ssd8t/Projects/osawa/SAT1/${i}/trim
     trim_galore --paired /mnt/196public/from_osawa_2022_07_27/RH22043699/${i}_1.fq.gz /mnt/196public/from_osawa_2022_07_27/RH22043699/${i}_2.fq.gz

     mkdir -p /mnt/ssd8t/Projects/osawa/SAT1/${i}/human
     cd /mnt/ssd8t/Projects/osawa/SAT1/${i}/human
     STAR --runThreadN 4 \
          --genomeDir /mnt/Bambi/hg38_from_shimam_sensei/star_index \
          --readFilesIn ../trim/${i}_1_val_1.fq.gz ../trim/${i}_2_val_2.fq.gz \
          --readFilesCommand gunzip -c \
          --genomeLoad NoSharedMemory \
          --outFilterMultimapNmax 1 \
          --outReadsUnmapped Fastx
          #--outSAMtype BAM SortedByCoordinate \

     mkdir -p /mnt/ssd8t/Projects/osawa/SAT1/${i}/mouse
     cd /mnt/ssd8t/Projects/osawa/SAT1/${i}/mouse
     STAR --runThreadN 4 \
          --genomeDir /mnt/Bambi/mm10/star_index \
          --readFilesIn ../human/Unmapped.out.mate1 ../human/Unmapped.out.mate2 \
          --genomeLoad NoSharedMemory \
          --outFilterMultimapNmax 1 \
          --outReadsUnmapped Fastx
          #--outSAMtype BAM SortedByCoordinate \
done

# BAMに変換、上で一緒にすればよかった
for i in SC_1 SC_2 SC_3 shSAT1_2_1 shSAT1_2_2 shSAT1_2_3
do
     cd /mnt/ssd8t/Projects/osawa/SAT1/${i}/human
     samtools sort -@ 8 -O bam -o ${i}_human.sort.bam Aligned.out.sam
     samtools index ${i}_human.sort.bam && rm Aligned.out.sam
     cd /mnt/ssd8t/Projects/osawa/SAT1/${i}/mouse
     samtools sort -@ 8 -O bam -o ${i}_mouse.sort.bam Aligned.out.sam
     samtools index ${i}_mouse.sort.bam && rm Aligned.out.sam
done


# mouseでカウント
for i in SC_1 SC_2 SC_3 shSAT1_2_1 shSAT1_2_2 shSAT1_2_3
do
     cd /mnt/ssd8t/Projects/osawa/SAT1/${i}/mouse
     htseq-count -f bam -r pos ${i}_mouse.sort.bam /mnt/Bambi/mm10/Mus_musculus/UCSC/mm10/Annotation/Archives/archive-2015-07-17-14-33-26/Genes/genes.gtf > ${i}_mouse_count.txt
done

# humanでカウント
for i in SC_1 SC_2 SC_3 shSAT1_2_1 shSAT1_2_2 shSAT1_2_3
do
     cd /mnt/ssd8t/Projects/osawa/SAT1/${i}/human
     htseq-count -f bam -r pos ${i}_human.sort.bam /mnt/Bambi/Projects/Homo_sapiens/UCSC/hg38/Annotation/Archives/archive-2015-08-14-08-18-15/Genes/genes.gtf > ${i}_human_count.txt
done



# mkdir /mnt/Chip/Projects/osawa/SAT1/SC_1/mouse_mapping
# STAR --runThreadN 4 \
#      --genomeDir /mnt/Bambi/mm10/star_index \
#      --readFilesIn /mnt/Chip/Projects/osawa/SAT1/SC_1/Unmapped.out.mate1 /mnt/Chip/Projects/osawa/SAT1/SC_1/Unmapped.out.mate2 \
#      --genomeLoad NoSharedMemory \
#      --outFilterMultimapNmax 1 \
#      --outReadsUnmapped Fastx
# mv Log.final.out Log.out Log.progress.out SJ.out.tab Unmapped.out.mate2 Unmapped.out.mate1 Aligned.out.sam /mnt/Chip/Projects/osawa/SAT1/SC_1/mouse_mapping/.

# mkdir /mnt/Chip/Projects/osawa/SAT1/SC_2/mouse_mapping
# STAR --runThreadN 4 \
#      --genomeDir /mnt/Bambi/mm10/star_index \
#      --readFilesIn /mnt/Chip/Projects/osawa/SAT1/SC_2/Unmapped.out.mate1 /mnt/Chip/Projects/osawa/SAT1/SC_2/Unmapped.out.mate2 \
#      --genomeLoad NoSharedMemory \
#      --outFilterMultimapNmax 1 \
#      --outReadsUnmapped Fastx
# mv Log.final.out Log.out Log.progress.out SJ.out.tab Unmapped.out.mate2 Unmapped.out.mate1 Aligned.out.sam /mnt/Chip/Projects/osawa/SAT1/SC_2/mouse_mapping/.

# mkdir /mnt/Chip/Projects/osawa/SAT1/SC_3/mouse_mapping
# STAR --runThreadN 4 \
#      --genomeDir /mnt/Bambi/mm10/star_index \
#      --readFilesIn /mnt/Chip/Projects/osawa/SAT1/SC_3/Unmapped.out.mate1 /mnt/Chip/Projects/osawa/SAT1/SC_3/Unmapped.out.mate2 \
#      --genomeLoad NoSharedMemory \
#      --outFilterMultimapNmax 1 \
#      --outReadsUnmapped Fastx
# mv Log.final.out Log.out Log.progress.out SJ.out.tab Unmapped.out.mate2 Unmapped.out.mate1 Aligned.out.sam /mnt/Chip/Projects/osawa/SAT1/SC_3/mouse_mapping/.

# mkdir /mnt/Chip/Projects/osawa/SAT1/shSAT1_2_1/mouse_mapping
# STAR --runThreadN 4 \
#      --genomeDir /mnt/Bambi/mm10/star_index \
#      --readFilesIn /mnt/Chip/Projects/osawa/SAT1/shSAT1_2_1/Unmapped.out.mate1 /mnt/Chip/Projects/osawa/SAT1/shSAT1_2_1/Unmapped.out.mate2 \
#      --genomeLoad NoSharedMemory \
#      --outFilterMultimapNmax 1 \
#      --outReadsUnmapped Fastx
# mv Log.final.out Log.out Log.progress.out SJ.out.tab Unmapped.out.mate2 Unmapped.out.mate1 Aligned.out.sam /mnt/Chip/Projects/osawa/SAT1/shSAT1_2_1/mouse_mapping/.

# mkdir /mnt/Chip/Projects/osawa/SAT1/shSAT1_2_2/mouse_mapping
# STAR --runThreadN 4 \
#      --genomeDir /mnt/Bambi/mm10/star_index \
#      --readFilesIn /mnt/Chip/Projects/osawa/SAT1/shSAT1_2_2/Unmapped.out.mate1 /mnt/Chip/Projects/osawa/SAT1/shSAT1_2_2/Unmapped.out.mate2 \
#      --genomeLoad NoSharedMemory \
#      --outFilterMultimapNmax 1 \
#      --outReadsUnmapped Fastx
# mv Log.final.out Log.out Log.progress.out SJ.out.tab Unmapped.out.mate2 Unmapped.out.mate1 Aligned.out.sam /mnt/Chip/Projects/osawa/SAT1/shSAT1_2_2/mouse_mapping/.

# mkdir /mnt/Chip/Projects/osawa/SAT1/shSAT1_2_3/mouse_mapping
# STAR --runThreadN 4 \
#      --genomeDir /mnt/Bambi/mm10/star_index \
#      --readFilesIn /mnt/Chip/Projects/osawa/SAT1/shSAT1_2_3/Unmapped.out.mate1 /mnt/Chip/Projects/osawa/SAT1/shSAT1_2_3/Unmapped.out.mate2 \
#      --genomeLoad NoSharedMemory \
#      --outFilterMultimapNmax 1 \
#      --outReadsUnmapped Fastx
# mv Log.final.out Log.out Log.progress.out SJ.out.tab Unmapped.out.mate2 Unmapped.out.mate1 Aligned.out.sam /mnt/Chip/Projects/osawa/SAT1/shSAT1_2_3/mouse_mapping/.








