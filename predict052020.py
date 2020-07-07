
import os
import sys
import pathlib
import argparse
import re
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import one_hot
from sklearn import metrics
import pickle
import time
######################################################
start_time = time.time()


# Load dataset
# =============================================================================
#Verifying Steps
# =============================================================================

#cmdline = python path/predict052020.py path/inputfile.csv path/RF_model.sav --output file.csv (this option is not mandatory)

#verifying Argument
parser=argparse.ArgumentParser()
parser.add_argument("csv" ,help="Take a csv file as argument/ file path needed if it's outside your working directory")
parser.add_argument("rfmodel" ,help="Take the model file (.sav) as argument/ file path needed if it's outside your working directory")
parser.add_argument("-out","--output", type=str, default="rf4sv_PredictionFile.csv", help="String to name the outputfile")
args=parser.parse_args()

# Captures path and verifies if it's an actual file 
from pathlib import Path
filename=Path(args.csv)


# Verifies if the file exists
if not filename.exists():
    print("Oops, file doesn't exist! \n")
    exit()
else:
   print(os.path.realpath(args.csv))

if not Path.is_file(filename):
# debug   print(filename.name) 
    print("Oops, this is not a file! \n")
    exit()
else:
    print("File successfully loaded")
    
modelfile=Path(args.rfmodel)

modefilepath=os.path.realpath(args.rfmodel)


# Verifies if the file exists
if not modelfile.exists():
    print("Oops, file doesn't exist! \n")
    exit()
else:
    # print(modelfile.name, '\n') 
    print(modefilepath)

if not Path.is_file(modelfile):
# debug   print(filename.name) 
    print("Oops, this is not a file! \n")
    exit()
else:
    print("File successfully loaded")
# ============================================================================= 
#Read input file

dataframe = pd.read_csv(filename, sep='\t')

##########################
# get dummies data frames
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
# print(dataframeok.head(9))
# exit

# #################################
# prepare data for model learning
#
array = dataframeok.values
X = array[:,0:4]
# print(X)
# exit
#Y = array[:,4]
#y_test=Y.astype('int') # transform to int for predict
#print(Y)

# Load the model
rf = pickle.load(open(modefilepath, 'rb'))
predictions = rf.predict(X)

df = pd.DataFrame(predictions)
df.columns=["Variant"]

# df1= pd.concat([dataframe['chr'],dataframeok,df],axis=1, sort=False)
df1= pd.concat([dataframe,df],axis=1, sort=False)
# print(df1.head(9))
# exit
df1.to_csv(args.output, index=False)
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

'''
PredictPostprocessing incompatibilties python2vs3
df=pd.read_csv("rf_1Mcig.csv", dtype={"chr":str,"Start":int,"End":int,"Variant": str},sep=',')
# print(df.head(9))
# exit
print("for in range\n")
print("chr"+'\t'+"Start"+'\t'+"End"+'\t'+"Deletion_Size",file=open("subMpitetroupas.csv","a"))
for i in range(len(df.index)):
    if df.loc[i,'Variant']=='1' and df.loc[i-1,'Variant']=='0' and df.loc[i+1,'Variant']=='0':
        print(df.loc[i, 'chr'],'\t',df.loc[i, 'Start'],'\t',df.loc[i, 'End'], '\t',(df.loc[i, 'End']-df.loc[i, 'Start']), 
              file=open("subMpitetroupas.csv","a"))
    elif df.loc[i,'Variant']=='1' and df.loc[i-1,'Variant']=='0' and df.loc[i+1,'Variant']=='1':
        
        toto=df.loc[i, 'chr']+'\t'+str(df.loc[i, 'End'])
        enddf=toto.split("\t")
        # print(enddf[1])
        
        # print(df.loc[i, 'chr'],'\t',df.loc[i, 'End'], 
              # file=open("/home/emira.cherif/Documents/ml4sv/eletric-scheep-master/testouput/subMpitetroupas.csv","a"))
    elif df.loc[i,'Variant']=='1' and df.loc[i-1,'Variant']=='1' and df.loc[i+1,'Variant']=='0':
        if df.loc[i,'Start']< int(enddf[1]):
            size=abs(df.loc[i,'Start']-int(enddf[1]))
            print(enddf[0],'\t',df.loc[i, 'Start'],'\t',enddf[1],'\t',size,          
                file=open("subMpitetroupas.csv","a"))
        else:
            size=df.loc[i,'Start']-int(enddf[1])
            print(toto,'\t',df.loc[i, 'Start'], '\t',size,          
                file=open("subMpitetroupas.csv","a"))
'''
