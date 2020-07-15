# RF4SV: A Random Forest approach for accurate deletion detection

<p align="center">
  <img width="250" height="250" src="https://github.com/alvesrcoo/eletric-scheep/blob/master/images/RF4SV_logo.png?raw=true">
</p>

Towards a machine learning based model for variant calling. 

# About RF4SV

Efficiently detecting genomic structural variants (SVs) is a key step to grasp the "missing heritability" underlying complex traits involved in major evolutionary processes such as speciation, phenotypic plasticity, and adaptive responses. We present a random forest ensemble method for accurate deletion identification. We called this approach RF4SV.

## Getting Started 

To create/import conda environment for machine learning [RF4SV]: 
```
conda env create -f env.yaml
```

To activate the environment:  
```
source activate ml4sv
```

To deactivate an active environment: 
```
source deactivate
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

To benchmark using a new simple CNN (Deep Learning) on CPU or GPU
```
python ml4sv_cnn.py
```

-- Remember to change input file in the data folder
-- Run with [Deepo](https://hub.docker.com/r/ufoym/deepo/) - The GPU version

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
python predict052020.py
```
