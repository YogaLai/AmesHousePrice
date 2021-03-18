import numpy as np 
import pandas as pd 
import pickle

def predict_price(feature):
    pipe = pickle.load(open('pipe.sav', 'rb'))
    # test=pd.read_csv('test.csv')
    # preds=pipe.predict(test.iloc[0:1])
    trn=pd.read_csv('train.csv')
    trn['SalePrice'] = np.log1p(trn.SalePrice)
    y=trn['SalePrice']
    del trn['SalePrice']

    #轉換feature
    trn=trn.iloc[0:1]
    # trn['LotArea']=feature['LotArea']
    trn['YearBuilt']=feature['YearBuilt']
    trn['Neighborhood']=feature['Neighborhood']
    # trn['LandContour']=feature['LandContour']
    trn['1stFlrSF']=trn['2ndFlrSF']=trn['TotalBsmtSF']=feature['sm']/3
    trn['LotArea']=feature['sm']/3/0.6
    trn['FullBath']=feature['bath']
    trn['BsmtFullBath']=trn['BsmtHalfBath']=trn['HalfBath']=0
    # pipe.fit(trn,y.iloc[0:1])

    preds=pipe.predict(trn)
    output=np.asscalar(np.expm1(preds))
    output='${:.4f}'.format(output)
    return output

# if __name__ == "__main__":
#     pipe = pickle.load(open('pipe.sav', 'rb'))
#     trn=pd.read_csv('train.csv')
#     trn['SalePrice'] = np.log1p(trn.SalePrice)
#     y=trn['SalePrice']
#     del trn['SalePrice']

#     preds=pipe.predict(trn)
#     print(f'Train set RMSE: {round(np.sqrt(mean_squared_error(y, preds)), 4)}')
#     print(f'Train set MAE: {round(mean_absolute_error(np.expm1(y), np.expm1(preds)), 2)}')
