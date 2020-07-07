#!/bin/sh

#cmd sh mapping.sh path_ref path_workingdir prefixfile : sh mapping.sh /scratch/test_EC/Drosophila_melanogaster.BDGP6.22.fa /scratch/test_EC/10M hiseq_inv_BDGP6_10Mreads 
###### Execution
path_ref=$1
path_wdir=$2
prefix_file=$3

#indexing
cmd_Index="bwa index $path_ref" 
echo "Indexing step : $cmd_Index";
$cmd_Index;
#Mapping
rgstring='@RG\tID:foo\tSM:bar'
cmd_mem="bwa mem -R "$rgstring" $path_ref $path_wdir/${prefix_file}_R1.fastq $path_wdir/${prefix_file}_R2.fastq"
sub_var0="$path_wdir/${prefix_file}.sam"
echo "Mapping step : $cmd_mem > $sub_var0";
$cmd_mem > $sub_var0;
#SamToBam
cmd_view="samtools view -b $path_wdir/${prefix_file}.sam -o $path_wdir/${prefix_file}.bam"
echo "SamtoBam step : $cmd_view";
$cmd_view;
#BamOrder
cmd_order="samtools sort -l 0 -o $path_wdir/${prefix_file}.sorted.bam $path_wdir/${prefix_file}.bam"
echo "Sorting bam file : $cmd_order";
$cmd_order;
#bai
cmd_bai="samtools index -b $path_wdir/${prefix_file}.sorted.bam"
echo "Indexing bam file & creating bai file : $cmd_bai";
$cmd_bai;
#BamToBed
cmd_bamTobed="bamToBed -cigar -i $path_wdir/${prefix_file}.sorted.bam"
sub_var="$path_wdir/${prefix_file}.sorted.bed"
echo "BamToBed & extractiong cigar steps : $cmd_bamTobed > $sub_var";
$cmd_bamTobed > $sub_var;
#Samtools bedcov: bed Covrage
cmd_bedcov="samtools bedcov $path_wdir/${prefix_file}.sorted.bed $path_wdir/${prefix_file}.sorted.bam"
sub_var1="$path_wdir/${prefix_file}.sorted.bed.cov"
echo "Estimating coverage per region step : $cmd_bedcov > $path_wdir/${prefix_file}.sorted.bed.cov";
$cmd_bedcov > $sub_var1;
# Extracting tags from bam file: NM MD MC AS
cmd_extract="samtools view $path_wdir/${prefix_file}.sorted.bam" 
cmd_extract1="grep -v '\*'"
cmd_extract2="cut -f 12,13,14,15"
sub_var2="$path_wdir/${prefix_file}.sorted.tag.bed"
echo "Extracting NM MD MC AS tags  : $cmd_extract | $cmd_extract1 | $cmd_extract2 > $sub_var2";
$cmd_extract | grep -v '\*' | $cmd_extract2 > $sub_var2;
# Fusion of bed files
cmd_fusion="paste  $path_wdir/${prefix_file}.sorted.bed.cov $path_wdir/${prefix_file}.sorted.tag.bed"
sub_var3="$path_wdir/${prefix_file}.sorted.bed.cov.tag"
echo "Creating prediction input matrix : $cmd_fusion > $sub_var3";
$cmd_fusion > $sub_var3;
#Cleaning temp files
cmd_clean="rm $path_wdir/${prefix_file}.sorted.bed $path_wdir/${prefix_file}.sorted.bed.cov $path_wdir/${prefix_file}.sorted.tag.bed"
echo "Cleaning temp files  : $cmd_clean";
$cmd_clean;
