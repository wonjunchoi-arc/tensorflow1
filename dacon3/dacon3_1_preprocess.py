import pandas as pd
import numpy as np

train = pd.read_csv('../data/dacon3/train.csv',index_col=None,header=0)
test = pd.read_csv('../data/dacon3/test.csv',index_col=None,header=0)
dev = pd.read_csv('../data/dacon3/dev.csv',index_col=None,header=0)


train= pd.concat([train,dev])

x = train['SMILES']
y1 = train['S1_energy(eV)']
y2 = train['T1_energy(eV)']
y = y1 - y2
print(y)

############데이터 가져오기########################
