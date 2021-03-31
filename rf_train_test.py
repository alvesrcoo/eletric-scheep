#########################################################
#
# by Ronnie Alves
#
#
# Compare several classique and ensemble learning methods
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time

start_time = time.time()

#load dataset
#dataframe = pd.read_csv('data/test_del_ajout.csv', sep='\t')
dataframe = pd.read_csv('data/data.csv', sep=',', low_memory=False)

##########################
# get dummies data frames
#
# variables: CIGAR, Variant
#
dataframe['CIGAR'] = dataframe['CIGAR'].str.replace('[^a-zA-Z]', '')
dummies_CIGAR = pd.get_dummies(dataframe['CIGAR'])
#print(dummies_CIGAR.head(2))

dummies_Variant = pd.get_dummies(dataframe['Variant'])
#print(dummies_Variant.head(2))

dataframeok = pd.concat([dataframe[['Start','End','CovPerReadRegion']],
		dummies_CIGAR,
		dummies_Variant['Deletion']], axis=1, sort=False)
#print(dataframeok.head(9))

#################################
# prepare data for model learning
#
df = dataframeok.shape[1]
array = dataframeok.values
X = array[:,0:df-2]
#print(X)
Y = array[:,df-1]
Y=Y.astype('int') # transform to int for predict
#print(Y)

seed = 777

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators = 1000)
rf.fit(X_train, y_train)
predictions = rf.predict(X_test)

outFile = open("rf_output.txt", "w")
# Accuracy
#print("Accuracy: %f" % (metrics.accuracy_score(y_test, predictions)))
outFile.write("Accuracy: %f\n" % (metrics.accuracy_score(y_test, predictions)))
# F1
#print("F1 Score: %f" % (metrics.f1_score(y_test, predictions)))
outFile.write("F1 Score: %f\n" % (metrics.f1_score(y_test, predictions)))
# Precision
#print("Precision Score: %f" % (metrics.precision_score(y_test, predictions)))
outFile.write("Precision Score: %f\n" % (metrics.precision_score(y_test, predictions)))
# Recall
#print("Recall Score: %f" % (metrics.recall_score(y_test, predictions)))
outFile.write("Recall Score: %f\n" % (metrics.recall_score(y_test, predictions)))
# AUC
#print("AUC: %f" % (metrics.roc_auc_score(y_test, predictions)))
outFile.write("AUC: %f\n" % (metrics.roc_auc_score(y_test, predictions)))
# TIME
#print("---runtime %s seconds ---" % (time.time() - start_time))
outFile.write("---runtime %s seconds ---\n" % (time.time() - start_time))
outFile.close()
