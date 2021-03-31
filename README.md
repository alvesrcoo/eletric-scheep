# RF4SV: A Random Forest approach for accurate deletion detection

<p align="center">
  <img width="250" height="250" src="https://github.com/alvesrcoo/eletric-scheep/blob/master/images/RF4SV_logo.png?raw=true">
</p>

This is a machine learning model based for accurate deletion detection for <a href="https://osf.io/6kf92/">RF4SV OSF project</a>. 

# About RF4SV

Efficiently detecting genomic structural variants (SVs) is a key step to grasp the "missing heritability" underlying complex traits involved in major evolutionary processes such as speciation, phenotypic plasticity, and adaptive responses. We present a random forest ensemble method for accurate deletion identification. We called this approach RF4SV.

## Getting Started 

Requirements: 
```
- python3
- numpy
- pandas
- scikit-learn
- keras
- tensorflow
```

# Download Drosophila melanogaster genome
```
 wget ftp://ftp.ensemblgenomes.org/pub/metazoa/release-43/fasta/drosophila_melanogaster/dna/Drosophila_melanogaster.BDGP6.22.dna.chromosome.*
```

# Simulate deletions (RSVSim)
```
simdata/simSVDel.R
```

# Read simulation process

Read simulation
```
simdata/iss_ReadSim.sh
```
-- Remember to adapt the paths and outputs.

# Pre-process
Create input prediction matrix
```
MappExtract.sh
```

## Handling imbalanced Data
Resampling data to find "good" balance
```
python balance_dataset.py
```

## Benchmarking 1

To benchmark classique (LR: Logistic Regression; LDA: Linear Discriminant Analysis; KNN: k-Nearest Neighbors; NB: Naive Bayes; 
CART: Classification and Regression Tree algorithm) and ensemble learning methods (RF: Random Forest; ADA: AdaBoost; GBM: Gradient Boosting Machines):

To run the benchmark: 
```
python benchml.py
```
-- Remeber to change input file in the data folder

## Benchmarking 2

To benchmark using Neural Network (Deep Learning)
```
python ml4sv_cnn.py
```

-- Remember to change input file in the data folder

## Building the RF model

To build and save a RF model 

```
python rf_model.py
```

## Do prediction

To load a saved RF model (uploaded in <a href="https://osf.io/6kf92/">OSF project</a> --> Files --> Model --> 'RF_model.sav') and predict new data

```
python predict.py
```

## Post-process

For friendly user prediction results

```
python path/PredictPostProcess052020.py path/predictionfile.csv
```

-- In this case, data must be with 1 or 0 in column "Variant"

## Docker RF4SV
Train and predict models using Random Forest, Benchmarking Algorithms and Neural Network

To run RF4SV project using <a href="https://hub.docker.com/repository/docker/robertoxavier/rf4sv">Docker</a>:
```
docker run --rm -v /your/data/dir:/data robertoxavier/rf4sv:v1.0 bash
```
