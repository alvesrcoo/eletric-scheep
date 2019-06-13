#########################################################
#
# by Ronnie Alves 
# CNN simple arch
#
# Compare several classique and ensemble learning methods
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import keras
from keras import metrics
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, Embedding, Input
from keras.optimizers import Adam, RMSprop
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences


# load dataset
#names = ['chr', 'Start', 'End', 'Read_Id', 'Bed_Score', 'Sens', 'CIGAR', 'age', 'CovPerReadRegion', 
#			'NM_dist2ref', 'MD_missmatchposi', 'MC_CIGARmate', 'AlignScore', 'Variant']

dataframe = pd.read_csv('data/test_del_ajout.csv', sep='\t')

print(dataframe.head(3))

variantLabels = dataframe["Variant"].copy()
print("Class", variantLabels)

# get only chars from CIGAR
dataframe['CIGAR'] = dataframe['CIGAR'].str.replace('[^a-zA-Z]', '')

dataframe = dataframe.drop("Variant", axis=1) # drop target
print(dataframe.head(3))

dataframe_num = dataframe.select_dtypes(include=[np.number])
print("Num.Cols", dataframe_num.head(3))

print("CIGAR", dataframe['CIGAR'].unique())

print("Variant", variantLabels.unique())

########################
##
## Preprocessing
##
##
'''
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
#       ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
        ('std_scaler', StandardScaler()),
    ])

try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer # Scikit-Learn < 0.20

try:
    from sklearn.preprocessing import OrdinalEncoder # just to raise an ImportError if Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20
    
#########################

dataframe_num = dataframe.select_dtypes(include=[np.number])
print(dataframe_num.head(3))

num_attribs = list(dataframe_num)
print("NumCols: ", num_attribs)

cat_attribs = ["CIGAR"]
print(dataframe['CIGAR'].unique())

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

dataframe_prepared = full_pipeline.fit_transform(dataframe)

print(dataframe_prepared.shape)

print(dataframe_prepared)

'''
###########################
##
## Embeddings
##

dummies_Variant = pd.get_dummies(variantLabels)
print("dummies", dummies_Variant.head(2))

e_CIGAR = Embedding(200,17, input_length=1)
v_CIGAR = dataframe['CIGAR'].values
print(v_CIGAR)
size_CIGAR = 50
encoded_CIGAR = [one_hot(d, size_CIGAR) for d in v_CIGAR]
#print("Encoded CIGAR", encoded_CIGAR)
#print(np.unique(encoded_CIGAR))

df_CIGAR = pd.DataFrame(data=encoded_CIGAR, columns=['CIGAR'])

# define the model
model = Sequential()
model.add(Embedding(size_CIGAR, 3, input_length=1))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# summarize the model
print(model.summary())

# fit the model
model.fit(df_CIGAR, dummies_Variant['Deletion'], epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(df_CIGAR, dummies_Variant['Deletion'], verbose=0)
print('Accuracy: %f' % (accuracy*100))

'''
#variant_input = Input(shape(1,),dtype='int32', name='variantLabels')
#variant_input = Input(shape(1,))
variant_emb = Embedding(3, 3, input_length=20)#(variant_input)
variant_out = Flatten()(variant_emb)
variant_output = Dense(1,activation='relu',name='variant_model_out')

#cigar_input = Input(shape(1,),dtype='int32', name='cigarLabels')
cigar_emb = Embedding(output_dim=9, input_dim=9, input_length=1)(cigar_input)
cigar_out = Flatten()(cigar_emb)
cigar_output = Dense(1,activation='relu',name='cigar_model_out')

###########################
##
## CNN
##

main_input = Input(shape=(32,))
lyr = keras.layers.concatenate([main_input,variant_out])
lyr = Dense(100,activation="relu")(lyr)
lyr = Dense(50,activation="relu")(lyr)
main_output = Dense(1, name="main_output")(lyr)

var_model = Model(
	inputs=[main_input, cigar_input, variant_input],
	outputs=[main_output,cigar_output, variant_output]
	) 

var_model.compile(
	loss="mean_squared_error",
	optimizer=Adam(lr=0.001),
	metrics=[metrics.mae],
	loss_weights=[1.0,0.5] 
	)
	
epochs = 500
batch = 128

history = var_model.fit(
	dataframe_num,
	variant_output,
	batch_size=batch,
	epochs=epochs,
	shuffle=True,
	verbose=1,
	callbacks=keras_callbacks,
	validation_split=0.30
	)
'''