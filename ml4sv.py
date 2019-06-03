# Compare Algorithms
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

# load dataset
#names = ['chr', 'Start', 'End', 'Read_Id', 'Bed_Score', 'Sens', 'CIGAR', 'age', 'CovPerReadRegion', 
#			'NM_dist2ref', 'MD_missmatchposi', 'MC_CIGARmate', 'AlignScore', 'Variant']

dataframe = pd.read_csv('/Users/3i521388/Documents/Github/ml4sv/data/test_del_ajout.csv', sep='\t')

##########################
# get dummies data frames
#
# CIGAR, Variant
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

array = dataframeok.values
#X = array[:,1:3]
X = array[:,0:9]
print(X)
Y = array[:,10]
Y=Y.astype('int') # transform to int for predict
print(Y)

#X = array[:,0:8]
#Y = array[:,8]
# prepare configuration for cross validation test harness
seed = 777

################
# prepare models
#
models = []
#models.append(('LR', LogisticRegression()))
models.append(('LR', LogisticRegression(solver='lbfgs')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))
models.append(('SVM', SVC(gamma='scale')))

#############################
# evaluate each model in turn
#
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring, error_score=np.nan)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Benchmark classsique ML')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()