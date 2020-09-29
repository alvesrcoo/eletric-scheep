import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.preprocessing.text import one_hot
import pickle
import time

start_time = time.time()

# Load dataset
dataframe = pd.read_csv('data/data.csv', sep=',', low_memory=False)

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

dummies_Variant = pd.get_dummies(dataframe['Variant'])
#print(dummies_Variant.head(2))

dataframeok = pd.concat([dataframe[['Start','End','CIGAR','CovPerReadRegion']],
		dummies_Variant['Deletion']], axis=1, sort=False)
#print(dataframeok.head(9))

#################################
# prepare data for model learning
#
array = dataframeok.values

X = array[:,0:4]
#print(X)
Y = array[:,4]
Y=Y.astype('int') # transform to int for predict
#print(Y)

seed = 42

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators = 1000, n_jobs = -1)
rf.fit(X_train, y_train)

# Save the model
filename = 'RF_model.sav'
pickle.dump(rf, open(filename, 'wb'))
print("Time: %s seconds" % (time.time() - start_time))
