import pandas as pd

trn=pd.read_csv('train.csv')
nei=pd.get_dummies(trn.MSZoning)
print(nei)