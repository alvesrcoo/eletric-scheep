![](images/RF4SV_logo.png)
==========================

# Detecting variants with machine learning.

Towards a machine learning based model for variant calling. 

## Getting Started 

To create/import conda environment for machine learning [ml4sv]: 
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

## Benchmarking 1

To benchmark classique (LR: Logistic Regression; LDA: Linear Discriminant Analysis; KNN: k-Nearest Neighbors; NB: Naive Bayes; 
CART: Classification and Regression Tree algorithm) and ensemble learning methods (RF: Random Forest; ADA: AdaBoost; GBM: Gradient Boosting Machines):

To run the benchmark: 
```
python benchml.py
```
-- Remeber to change input file in the data folder

## Benchmarking 2
To benchmark using simple CNN (deep learning) on CPU
```
python cnn.py
```

## Benchmarking 2.1
To benchmark using simple CNN (deep learning) on GPU
```
python cnn_gpu.py
```

## Benchmarking 2.2
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
To load a saved RF model (uploaded on Drive 'RF_model.sav' --- Bigger than 25MB to upload here) and predict new data
```
python predict.py
```

## Handling imbalanced Data
Resampling data to find "good" balance
```
python balance_dataset.py
```
