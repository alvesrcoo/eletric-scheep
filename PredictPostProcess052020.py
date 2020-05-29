#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# =============================================================================
import os, sys, pathlib, argparse, re
import pandas as pd
# =============================================================================
#Verifying Steps
# =============================================================================

#cmdline = python path/PredictPostProcess052020.py path/predictionfile.csv (predict script output)

#verifying Argument
parser=argparse.ArgumentParser()
parser.add_argument("predict" ,help="Take the prediction csv file as argument/ file path needed if it's outside your working directory")
args=parser.parse_args()

# Captures path and verifies if it's an actual file 
from pathlib import Path
file=Path(args.predict)
# print(modelfile)
filepath=os.path.realpath(args.predict)
# print(filepath)

# Verifies if the file exists
if not file.exists():
    print("Oops, file doesn't exist! \n")
    exit()
else:
    print("Analysing ",file.name, '\n') 

if not Path.is_file(file):
# debug   print(filename.name) 
    print("Oops, this is not a file! \n")
    exit()
else:
    print(file.name," successfully loaded\n")

#======================================    

# print("Parsing File step\n")
# # =============================================================================
# #Reading file, extracting results and estimating deletion size
# # =============================================================================
        
df=pd.read_csv(filepath, dtype={"chr":str,"Start":int,"End":int,"Variant": str},sep=',')
# print(df.head(9))

print("Building reslut file\n")
print("chr"+'\t'+"Start"+'\t'+"End"+'\t'+"Deletion_Size",file=open("ml4sv_results.csv","a"))
for i in range(len(df.index)):
    if df.loc[i,'Variant']=='1' and df.loc[i-1,'Variant']=='0' and df.loc[i+1,'Variant']=='0':
        print(df.loc[i, 'chr'],'\t',df.loc[i, 'Start'],'\t',df.loc[i, 'End'], '\t',(df.loc[i, 'End']-df.loc[i, 'Start']), 
              file=open("ml4sv_results.csv","a"))
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
                file=open("ml4sv_results.csv","a"))
        else:
            size=df.loc[i,'Start']-int(enddf[1])
            print(toto,'\t',df.loc[i, 'Start'], '\t',size,          
                file=open("ml4sv_results.csv","a"))
        
print("ml4sv_results.csv is ready to use! \n")

# # =============================================================================
