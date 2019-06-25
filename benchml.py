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
import time

start_time = time.time()
#dataframe = pd.read_csv('data/test_del_ajout.csv', sep='\t')
dataframe = pd.read_csv('data.csv', sep=',', low_memory=False)

##########################
# get dummies data frames
#
# variables: CIGAR, Variant
#
dataframe['CIGAR'] = dataframe['CIGAR'].str.replace('[^a-zA-Z]', '')
#print(dataframe.CIGAR.head(3))
dummies_CIGAR = pd.get_dummies(dataframe['CIGAR'])
#print(dummies_CIGAR.head(2))

dummies_Variant = pd.get_dummies(dataframe['Variant'])
#print(dummies_Variant.head(2))

dataframeok = pd.concat([dataframe[['Start','End','CovPerReadRegion']],
		dummies_CIGAR,
		dummies_Variant['Deletion']], axis=1, sort=False)
#print(dataframeok.head(10))

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

################
# set models
#
models = []
models.append(('LR', LogisticRegression(solver='lbfgs', n_jobs = -1)))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier(n_jobs = -1)))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='scale')))
models.append(('AdaB', AdaBoostClassifier()))
models.append(('RF', RandomForestClassifier(n_estimators = 1000, n_jobs = -1)))
models.append(('GBM', GradientBoostingClassifier()))

outFile = open("output.txt", "w")

#############################
# model evaluation
results = []
names = []
scoring = 'f1'
for name, model in models:
	kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring, error_score=np.nan, n_jobs=-1)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f) " % (name, cv_results.mean(), cv_results.std())
	outFile.write(msg)
	print(msg)
	outFile.write("- Time: %s in seconds" % (time.time() - start_time))
	print("- Time: %s in seconds" % (time.time() - start_time))
	outFile.write("\n")

##############################
# boxplot ml comparison
fig = plt.figure()
fig.suptitle('Benchmark 10Fold-CV Stratified classsique ML : F1_score')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
#plt.savefig('save.png')

#print("---runtime %s seconds ---" % (time.time() - start_time))
#outFile.write("---runtime %s seconds ---" % (time.time() - start_time))
outFile.close()
