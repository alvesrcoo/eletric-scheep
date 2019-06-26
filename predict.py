import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot
from sklearn import metrics
import pickle
import time

start_time = time.time()

# Load dataset
dataframe = pd.read_csv('hiseq_rearg_1M_BDGP6.sorted.bed.cov.tag.csv', sep='\t')

##########################
# get dummies data frames
#
# variables: CIGAR, Variant
#
#------ ENCODE CIGAR to numeric
dataframe['CIGAR'] = dataframe['CIGAR'].str.replace('[^a-zA-Z]', '')
v_CIGAR = dataframe['CIGAR'].values
#print(v_CIGAR)

size_CIGAR = 50
encoded_CIGAR = [one_hot(d, size_CIGAR) for d in v_CIGAR]
#print(encoded_CIGAR)

df_encoded_CIGAR = pd.DataFrame(data=encoded_CIGAR, columns=['CIGAR'])
dataframe['CIGAR'] = df_encoded_CIGAR

#dummies_Variant = pd.get_dummies(dataframe['Variant'])
#print(dummies_Variant.head(2))

dataframeok = pd.concat([dataframe[['Start','End','CIGAR','CovPerReadRegion']]], axis=1, sort=False)
#print(dataframeok.head(9))

#################################
# prepare data for model learning
#
array = dataframeok.values
X = array[:,0:4]
#print(X)
#Y = array[:,4]
#y_test=Y.astype('int') # transform to int for predict
#print(Y)

# Load the model
rf = pickle.load(open('RF_model.sav', 'rb'))
predictions = rf.predict(X)

df = pd.DataFrame(predictions)
df.to_csv('rf_1M.csv', index=False)
'''
#outFile = open("rf_output.txt", "w")
# Accuracy
print("Accuracy: %f" % (metrics.accuracy_score(y_test, predictions)))
#outFile.write("Accuracy: %f\n" % (metrics.accuracy_score(y_test, predictions)))
# F1
print("F1 Score: %f" % (metrics.f1_score(y_test, predictions)))
#outFile.write("F1 Score: %f\n" % (metrics.f1_score(y_test, predictions)))
# Precision
print("Precision Score: %f" % (metrics.precision_score(y_test, predictions)))
#outFile.write("Precision Score: %f\n" % (metrics.precision_score(y_test, predictions)))
# Recall
print("Recall Score: %f" % (metrics.recall_score(y_test, predictions)))
#outFile.write("Recall Score: %f\n" % (metrics.recall_score(y_test, predictions)))
# AUC
print("AUC: %f" % (metrics.roc_auc_score(y_test, predictions)))
#outFile.write("AUC: %f\n" % (metrics.roc_auc_score(y_test, predictions)))
# TIME
print("---runtime %s seconds ---" % (time.time() - start_time))
#outFile.write("---runtime %s seconds ---\n" % (time.time() - start_time))
#outFile.close()
'''