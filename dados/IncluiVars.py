import pandas as pd
import numpy as np


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--RefFile", default='Cabine16384.pandas', type=str, help='maximum depth')
parser.add_argument("--CSVFile", default='transform-fp-head_cellcenter.csv', type=str, help='maximum depth')
parser.add_argument("--Output", default='transform-fp-head_cellcenter.pandas', type=str, help='maximum depth')

RefFile  = parser.parse_args().RefFile
CSVFile  = parser.parse_args().CSVFile
file  = parser.parse_args().Output

Vars = ['Vel','Tinsu','Qinsu']

dfref = pd.read_pickle(RefFile)
lista = dfref[['CaseId']+Vars].drop_duplicates()
lista['CaseId'] = lista['CaseId'].apply(lambda x: x+1)

dfdata = pd.read_csv(CSVFile)
dfdata.rename(columns={'caso':'CaseId'},inplace=True)

dfdata.insert(4,Vars[0],0)
dfdata.insert(5,Vars[1],0)
dfdata.insert(6,Vars[2],0)

for j in Vars: dfdata[j] = dfdata['CaseId'].apply(lambda i: lista[lista['CaseId']==i][j].to_numpy()[0])

dfdata.to_pickle(file)
