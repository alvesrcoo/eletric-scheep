#!/bin/sh
#iss cmd line InSilicoSeq iss generate --cpus 4 --draft your-ref-path --model hiseq --gc_bias --n_reads numberofread (int) --output output_path
#exp: iss generate --cpus 4 --draft /scratch/test_EC/Drosophila_melanogaster.BDGP6.22_inv_rearranged.fasta --model hiseq --gc_bias --n_reads 10M --output /scratch/test_EC
###### Execution 
cmd1="module load system/python/3.6.5"
echo "loading python3 module : $cmd1";
$cmd1;
# you need to adapt this part if you want to use it
path_ref="/scratch/test_EC/Drosophila_melanogaster.BDGP6.22_inv_rearranged.fasta"
path_ouput="/scratch/test_EC"
prefix_file="hiseq_ref_BDGP6_10M"
nreads="10M"
cmd="/home/cherif/.local/bin/iss generate --cpus 4 --draft $path_ref --model hiseq --gc_bias --n_reads $nreads --output $path_ouput/${prefix_file}reads" 
echo "Generating simulated reads : $cmd";
$cmd;
