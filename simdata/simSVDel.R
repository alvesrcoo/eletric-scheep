## try http:// if https:// URLs are not supported
source("https://bioconductor.org/biocLite.R")
biocLite("RSVSim")
library(RSVSim)
setwd("your_working_directory_path") #exp /home/toto/SVsim
#######################################################################
# Etapes de simulation des SVs: Deletions                             #
#######################################################################
#estimateSVSizes : Dessine des tailles de variation structurelles aleatoires.
#creation du vecteur svSizes
delSizes=read.table("DelSizes.txt", header = FALSE) 
sizeDels=estimateSVSizes(n = 8962, svSizes = delSizes[[1]], hist = TRUE)
#Simulation des délétions
sim_del = simulateSV(output="drososim/", 
                     genome="Drosophila_melanogaster.BDGP6.22.fa", 
                     dels=8962, sizeDels=sizeDels, repeatBias = TRUE, 
                     bpSeqSize=7, verbose=TRUE)
sim_del
metadata(sim_del)
#######################################################################
