####################
##
## Ronnie , 17/06/2019
##
## Basic CNN
##
##
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, learning_curve, cross_validate
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from keras.utils import plot_model
from keras.layers import Dense, Dropout, Flatten, Activation, Embedding, Input
from keras.preprocessing.text import one_hot,Tokenizer

############ CODE for ++ metrics ###################
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

#you can use it as following
#model.compile(loss='binary_crossentropy',
start_time = time.time()

# running with NN cores / UV:testes (control by TF backend)
import tensorflow as tf
from keras import backend as K
K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=71, inter_op_parallelism_threads=71)))

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset
dataframe = pd.read_csv('data/data.csv', sep='\t')

#------ ENCODE CIGAR to numeric
dataframe['CIGAR'] = dataframe['CIGAR'].str.replace('[^a-zA-Z]', '')
e_CIGAR = Embedding(200,50, input_length=1)
v_CIGAR = dataframe['CIGAR'].values
#print(v_CIGAR)
size_CIGAR = 50
encoded_CIGAR = [one_hot(d, size_CIGAR) for d in v_CIGAR]

df_encoded_CIGAR = pd.DataFrame(data=encoded_CIGAR, columns=['CIGAR'])
#print("df_encoded_CIGAR", df_encoded_CIGAR.head(2))

#------ ENCODE Variant
labels = dataframe["Variant"].copy()
#print("Class", variantLabels)

df_encoded_Variant = pd.get_dummies(labels)
#print("df_encoded_Variant", df_encoded_Variant.head(2))

#------ DROP not useful cols // ASSIGN 
dataframe = dataframe.drop("Variant", axis=1) # drop target
dataframe = dataframe.drop("Read_Id", axis=1) # drop target
dataframe = dataframe.drop("chr", axis=1) # drop target
dataframe = dataframe.drop("Sens", axis=1) # drop target
dataframe = dataframe.drop("CIGAR", axis=1) # drop target
dataframe = dataframe.drop("NM_dist2ref", axis=1) # drop target
dataframe = dataframe.drop("MD_missmatchposi", axis=1) # drop target
dataframe = dataframe.drop("MC_CIGARmate", axis=1) # drop target
dataframe = dataframe.drop("AlignScore", axis=1) # drop target
dataframe['Align'] = dataframe['End']-dataframe['Start'] 

#print(dataframe.head(2))

#------ SPLIT into input (X) and output (Y) variables
dataset = dataframe.values
X = dataset[:,0:5].astype(float)
Y = df_encoded_Variant['Deletion']

#print(X)
#print(Y)

scoring = ['precision','recall','f1']

#------ MODEL building -> basic CNN model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(5, input_dim=5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(3, input_dim=5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc']) #one metric
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[precision,recall,f1]) #++ metrics
	# plot_model(model,to_file='basicCNN.png',show_shapes=True)
	return model
	
estimator = KerasClassifier(build_fn=create_baseline, epochs=10000, batch_size=5, verbose=2)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
# results = cross_val_score(estimator, X, Y, cv=kfold) #one metric
results = cross_validate(estimator, X, Y, cv=kfold, scoring=scoring) #scoring iterable sklearn
perf = {'tr_p' : results['train_precision'],
	'tt_p' : results['test_precision'],
	'tr_r' : results['train_recall'],
	'tt_r' : results['test_recall'],
	'tr_f' : results['train_f1'],
	'tt_f' : results['test_f1']
	}
csv = pd.DataFrame(perf, columns = ['tr_p','tt_p','tr_r','tt_r','tr_f','tt_f'])
csv.to_csv('perf.csv')
#print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100)) #one metric
print("---runtime %s seconds ---" % (time.time() - start_time))
