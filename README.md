# ML4SV: Detecting variants with machine learning.

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

## Balance Dataset
Resample dataset to balance classes
```
python balance_dataset.py
```
