cd /mnt/Donald/ito
source ~/miniconda3/bin/activate velocyto
cd added_bam

# https://www.10xgenomics.com/resources/analysis-guides/tutorial-navigating-10x-barcoded-bam-files

samtools view /mnt/244share/22年08月05日_九州大学・伊藤先生からのデータ/sample_alignments_ConDay3.bam \
    | LC_ALL=C grep "xf:i:25" > body_filtered_sam
samtools view -H /mnt/244share/22年08月05日_九州大学・伊藤先生からのデータ/sample_alignments_ConDay3.bam \
    > header_filted_sam 
cat header_filted_sam body_filtered_sam \
    > sample_alignments_ConDay3_possorted_genome_bam_filterd.sam
samtools view -b sample_alignments_ConDay3_possorted_genome_bam_filterd.sam \
    > sample_alignments_ConDay3_possorted_genome_bam_filterd.bam
rm sample_alignments_ConDay3_possorted_genome_bam_filterd.sam
samtools view sample_alignments_ConDay3_possorted_genome_bam_filterd.bam \
    | cut -f 12- | tr "\t" "\n" | grep "CB:Z:" | uniq \
    > cell_barcodes.txt
sed -i -e 's/CB:Z://g' cell_barcodes.txt 

# samtools sort -@ 8 -m 5G -t CB \
#     -O BAM \
#     -o cell_sorted_sample_alignments_ConDay3_possorted_genome_bam_filterd.bam \
#     sample_alignments_ConDay3_possorted_genome_bam_filterd.bam  

velocyto run -b cell_barcodes.txt \
    -o ConDay3 \
    sample_alignments_ConDay3_possorted_genome_bam_filterd.bam \
    /mnt/Daisy/ref_cellranger/refdata-gex-mm10-2020-A/genes/genes.gtf





bam_list=("Day24(Right)" "ReDay3(Left)" "OxtMG")
lane_n=1
for i in ${bam_list[@]}
do
    samtools view /mnt/244share/22年08月05日_九州大学・伊藤先生からのデータ/sample_alignments_${i}.bam \
        | LC_ALL=C grep "xf:i:25" > body_filtered_sam
    samtools view -H /mnt/244share/22年08月05日_九州大学・伊藤先生からのデータ/sample_alignments_${i}.bam \
        > header_filted_sam 
    cat header_filted_sam body_filtered_sam \
        > sample_alignments_${i}_possorted_genome_bam_filterd.sam
    samtools view -b sample_alignments_${i}_possorted_genome_bam_filterd.sam \
        > sample_alignments_${i}_possorted_genome_bam_filterd.bam
    rm sample_alignments_${i}_possorted_genome_bam_filterd.sam
    samtools view sample_alignments_${i}_possorted_genome_bam_filterd.bam \
        | cut -f 12- | tr "\t" "\n" | grep "CB:Z:" | uniq \
        > cell_barcodes_${i}.txt
    sed -i -e 's/CB:Z://g' cell_barcodes_${i}.txt

    # samtools sort -@ 8 -m 5G -t CB \
    #     -O BAM \
    #     -o cell_sorted_sample_alignments_${i}_possorted_genome_bam_filterd.bam \
    #     sample_alignments_${i}_possorted_genome_bam_filterd.bam  

    velocyto run -b cell_barcodes_${i}.txt \
        -o ${i} \
        sample_alignments_${i}_possorted_genome_bam_filterd.bam \
        /mnt/Daisy/ref_cellranger/refdata-gex-mm10-2020-A/genes/genes.gtf
done


bam_list=("ConDay3" "Day24\(Right\)" "ReDay3\(Left\)" "OxtMG")
for i in ${bam_list[@]}
do
    velocyto run -b cell_barcodes_${i}.txt \
        -o ${i} \
        cellsorted_sample_alignments_${i}_possorted_genome_bam_filterd.bam \
        /mnt/Daisy/ref_cellranger/refdata-gex-mm10-2020-A/genes/genes.gtf
done


bam_list=("ConDay3" "Day24\(Right\)" "ReDay3\(Left\)" "OxtMG")
for i in ${bam_list[@]}
do
    samtools sort -@ 8 -m 5G \
        -O BAM \
        -o re_sorted_sample_alignments_${i}_possorted_genome_bam_filterd.bam \
        sample_alignments_${i}_possorted_genome_bam_filterd.bam  

    velocyto run -b cell_barcodes_${i}.txt \
        -o ${i} \
        re_sorted_sample_alignments_${i}_possorted_genome_bam_filterd.bam  \
        /mnt/Daisy/ref_cellranger/refdata-gex-mm10-2020-A/genes/genes.gtf
done

