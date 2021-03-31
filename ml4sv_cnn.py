# CNN
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import keras
from keras import regularizers
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, learning_curve, cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from keras.utils import plot_model
from keras.layers import Dense, Dropout, Flatten, Activation, Embedding, Input
from keras.preprocessing.text import one_hot,Tokenizer
from sklearn.model_selection import train_test_split
from keras.utils import multi_gpu_model

# Metrics
from keras import backend as K

def check_units(y_true, y_pred):
    if y_pred.shape[1] != 1:
      y_pred = y_pred[:,1:2]
      y_true = y_true[:,1:2]
    return y_true, y_pred

def precision(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    y_true, y_pred = check_units(y_true, y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    y_true, y_pred = check_units(y_true, y_pred)
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

############ GPU ########################################
# running with GPU
# GPU=0,1,2,3,4,5,6,7 nvidia-docker run -it -v /bio/roberto_academico/ml4sv_experiments:/data -v /host/config:/config ufoym/deepo bash
#import tensorflow as tf
#from keras.backend import tensorflow_backend
#config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
#session = tf.Session(config=config)
#tensorflow_backend.set_session(session)

start_time = time.time()

# Load dataset
dataframe = pd.read_csv('data/data.csv', sep=',', low_memory=False)

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
sc = MinMaxScaler()
X = sc.fit_transform(X)
#print(X)

seed = 42

scoring = ['precision','recall','f1']

batch_size = 32
epochs = 10
nsplits = 10

#------ MODEL building -> basic CNN model
def create_baseline():
	model = Sequential()
	model.add(Dense(128, input_dim=4, activation='relu'))
	model.add(Dense(1, activation='softmax'))
	#model = keras.utils.multi_gpu_model(model, gpus=8)
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[precision,recall,f1])
	return model

estimator = KerasClassifier(build_fn=create_baseline, epochs=epochs, batch_size=batch_size, verbose=2)
kfold = StratifiedKFold(n_splits=nsplits, shuffle=True, random_state=seed)
results = cross_validate(estimator, X, Y, cv=kfold, scoring=scoring)
perf = {'tr_p' : results['train_precision'],
	'tt_p' : results['test_precision'],
	'tr_r' : results['train_recall'],
	'tt_r' : results['test_recall'],
	'tr_f' : results['train_f1'],
	'tt_f' : results['test_f1']
	}
csv = pd.DataFrame(perf, columns = ['tr_p','tt_p','tr_r','tt_r','tr_f','tt_f'])
csv.to_csv('ml4sv.csv')
outFile = open("ml4sv.txt", "w")
outFile.write("Time: %s seconds" % (time.time() - start_time))
#print("Time: %s seconds" % (time.time() - start_time))
outFile.close()
