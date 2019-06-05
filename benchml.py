#########################################################
#
# by Ronnie Alves 
#
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

# load dataset
#names = ['chr', 'Start', 'End', 'Read_Id', 'Bed_Score', 'Sens', 'CIGAR', 'age', 'CovPerReadRegion', 
#			'NM_dist2ref', 'MD_missmatchposi', 'MC_CIGARmate', 'AlignScore', 'Variant']

dataframe = pd.read_csv('data/test_del_ajout.csv', sep='\t')

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

#dataframe2 = dataframe[['Start','End','CovPerReadRegion']]
#print(dataframe2.head(2))

dataframeok = pd.concat([dataframe[['Start','End','CovPerReadRegion']],
		dummies_CIGAR,
		dummies_Variant['Deletion']], axis=1, sort=False)
print(dataframeok.head(2))

#df2 = dataframe[['Start','End','CovPerReadRegion']]
#print(df2.head(2))
#print(dataframe.loc[:, dataframe.columns.str.startswith('CIGAR_')].head(2))
#print(dataframe.loc[:, dataframe.columns.str.startswith('Variant_')].head(2))
#print(dataframe.head(2))

#################################
# prepare data for model learning
#
array = dataframeok.values
X = array[:,0:9]
print(X)
Y = array[:,10]
Y=Y.astype('int') # transform to int for predict
print(Y)

seed = 777

################
# set models
#
models = []
models.append(('LR', LogisticRegression(solver='lbfgs')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='scale')))
models.append(('AdaB', AdaBoostClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('GBM', GradientBoostingClassifier()))

#############################
# model evaluation
# f1_score :: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
#
results = []
names = []
scoring = 'f1'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring, error_score=np.nan)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
	
##############################
# boxplot ml comparison
fig = plt.figure()
fig.suptitle('Benchmark 10Fold-CV classsique ML : F1_score')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()