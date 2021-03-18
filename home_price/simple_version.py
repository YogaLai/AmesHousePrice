import numpy as np 
import pandas as pd 

# import seaborn as sns
# %matplotlib inline

from sklearn.model_selection import StratifiedShuffleSplit, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.inspection import plot_partial_dependence

from sklearn.linear_model import Lasso, Ridge, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
# import xgboost as xgb
# import lightgbm as lgb

import df_pipeline as dfp  # all the custom pipeline parts
import explore_data as exp  # all function to quickly explore the data

import warnings
import pickle

class tr_numeric(BaseEstimator, TransformerMixin):
    def __init__(self, SF_room=True, bedroom=True, bath=True, lot=True, service=True):
        self.columns = []  # useful to well behave with FeatureUnion
        self.SF_room = SF_room
        self.bedroom = bedroom
        self.bath = bath
        self.lot = lot
        self.service = service
     

    def fit(self, X, y=None):
        return self
    

    def remove_skew(self, X, column):
        X[column] = np.log1p(X[column])
        return X


    def SF_per_room(self, X):
        if self.SF_room:
            X['sf_per_room'] = X['GrLivArea'] / X['TotRmsAbvGrd']
        return X


    def bedroom_prop(self, X):
        if self.bedroom:
            X['bedroom_prop'] = X['BedroomAbvGr'] / X['TotRmsAbvGrd']
            del X['BedroomAbvGr'] # the new feature makes it redundant and it is not important
        return X


    def total_bath(self, X):
        if self.bath:
            X['total_bath'] = (X[[col for col in X.columns if 'FullBath' in col]].sum(axis=1) +
                             0.5 * X[[col for col in X.columns if 'HalfBath' in col]].sum(axis=1))
            del X['FullBath']  # redundant 

        del X['HalfBath']  # not useful anyway
        del X['BsmtHalfBath']
        del X['BsmtFullBath']
        return X


    def lot_prop(self, X):
        if self.lot:
            X['lot_prop'] = X['LotArea'] / X['GrLivArea']
        return X 


    def service_area(self, X):
        if self.service:
            X['service_area'] = X['TotalBsmtSF'] + X['GarageArea']
            del X['TotalBsmtSF']
            del X['GarageArea']
        return X
    

    def transform(self, X, y=None):
        for col in ['GrLivArea', '1stFlrSF', 'LotArea']:
            X = self.remove_skew(X, col)

        X = self.SF_per_room(X)
        X = self.bedroom_prop(X)
        X = self.total_bath(X)
        X = self.lot_prop(X)
        X = self.service_area(X)

        self.columns = X.columns
        return X
    

    def get_features_name(self):
        return self.columns
    
    
class drop_columns(BaseEstimator, TransformerMixin):
    '''
    Drops columns that are not useful for the model
    The decisions come from several iterations
    '''
    def __init__(self, lasso=False, ridge=False, forest=False, xgb=False, lgb=False):
        self.columns = []
        self.lasso = lasso
        self.ridge = ridge
        self.forest = forest
        self.xgb = xgb
        self.lgb = lgb
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        to_drop = [col for col in X.columns if 'NoGrg' in col]  # dropping dummies that are redundant
        to_drop += [col for col in X.columns if 'NoBsmt' in col]

        if self.lasso:
            to_drop += [col for col in X.columns if 'BsmtExposure' in col]
            to_drop += [col for col in X.columns if 'BsmtCond' in col]
            to_drop += [col for col in X.columns if 'ExterCond' in col]
            to_drop += [col for col in X.columns if 'HouseStyle' in col] 
            to_drop += [col for col in X.columns if 'LotShape' in col] 
            to_drop += [col for col in X.columns if 'LotFrontage' in col]
            to_drop += [col for col in X.columns if 'GarageYrBlt' in col] 
            to_drop += [col for col in X.columns if 'GarageType' in col] 
            to_drop += ['OpenPorchSF', '3SsnPorch'] 
        if self.ridge: 
            to_drop += [col for col in X.columns if 'BsmtExposure' in col]
            to_drop += [col for col in X.columns if 'BsmtCond' in col]
            to_drop += [col for col in X.columns if 'ExterCond' in col] 
            to_drop += [col for col in X.columns if 'LotFrontage' in col]
            to_drop += [col for col in X.columns if 'LotShape' in col] 
            to_drop += [col for col in X.columns if 'HouseStyle' in col] 
            to_drop += [col for col in X.columns if 'GarageYrBlt' in col]
            to_drop += [col for col in X.columns if 'GarageCars' in col] 
            to_drop += [col for col in X.columns if 'BldgType' in col] 
            to_drop += ['OpenPorchSF', '3SsnPorch']
        if self.forest: 
            to_drop += [col for col in X.columns if 'BsmtExposure' in col]
            to_drop += [col for col in X.columns if 'BsmtCond' in col]
            to_drop += [col for col in X.columns if 'ExterCond' in col] 
            to_drop += ['OpenPorchSF', '3SsnPorch'] 
        if self.xgb:
            to_drop += [col for col in X.columns if 'BsmtExposure' in col]
            to_drop += [col for col in X.columns if 'BsmtCond' in col]
            to_drop += [col for col in X.columns if 'ExterCond' in col]
        if self.lgb: 
            to_drop += [col for col in X.columns if 'LotFrontage' in col] 
            to_drop += [col for col in X.columns if 'HouseStyle' in col]
            to_drop += ['MisBsm'] 
            
        
        for col in to_drop:
            try:
                del X[col]
            except KeyError:
                pass
            
        self.columns = X.columns
        return X
    
    def get_feature_names(self):
        return list(self.columns)
    
    
class make_ordinal(BaseEstimator, TransformerMixin):
    '''
    Transforms ordinal features in order to have them as numeric (preserving the order)
    If unsure about converting or not a feature (maybe making dummies is better), make use of
    extra_cols and unsure_conversion
    '''
    def __init__(self, cols, extra_cols=None, include_extra='include'):
        self.cols = cols
        self.extra_cols = extra_cols
        self.mapping = {'Po':1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
        self.include_extra = include_extra  # either include, dummies, or drop (any other option)
    

    def fit(self, X, y=None):
        return self
    

    def transform(self, X, y=None):
        if self.extra_cols:
            if self.include_extra == 'include':
                self.cols += self.extra_cols
            elif self.include_extra == 'dummies':
                pass
            else:
                for col in self.extra_cols:
                    del X[col]
        
        for col in self.cols:
            X.loc[:, col] = X[col].map(self.mapping).fillna(0)
        return X
    
    
class recode_cat(BaseEstimator, TransformerMixin):        
    '''
    Recodes some categorical variables according to the insights gained from the
    data exploration phase.
    '''
    def __init__(self, mean_weight=10, te_neig=True, te_mssc=True):
        self.mean_tot = 0
        self.mean_weight = mean_weight
        self.smooth_neig = {}
        self.smooth_mssc = {}
        self.te_neig = te_neig
        self.te_mssc = te_mssc
    
    
    def smooth_te(self, data, target, col):
        tmp_data = data.copy()
        tmp_data['target'] = target
        mean_tot = tmp_data['target'].mean()
        means = tmp_data.groupby(col)['target'].mean()
        counts = tmp_data.groupby(col)['target'].count()

        smooth = ((counts * means + self.mean_weight * mean_tot) / 
                       (counts + self.mean_weight))
        return mean_tot, smooth
    
    def fit(self, X, y):
        if self.te_neig:
            self.mean_tot, self.smooth_neig = self.smooth_te(data=X, target=y, col='Neighborhood')

        if self.te_mssc:
            self.mean_tot, self.smooth_mssc = self.smooth_te(X, y, 'MSSubClass')
            
        return self
    
    
    def tr_GrgType(self, data):
        data['GarageType'] = data['GarageType'].map({'Basment': 'Attchd',
                                                     'CarPort': 'Detchd',
                                                     '2Types': 'Attchd' }).fillna(data['GarageType'])
        return data
    
    
    def tr_LotShape(self, data):
        fil = (data.LotShape != 'Reg')
        data['LotShape'] = 1
        data.loc[fil, 'LotShape'] = 0
        return data
    
    
    def tr_LandCont(self, data):
        fil = (data.LandContour == 'HLS') | (data.LandContour == 'Low')
        data['LandContour'] = 0
        data.loc[fil, 'LandContour'] = 1
        return data
    
    
    def tr_LandSlope(self, data):
        fil = (data.LandSlope != 'Gtl')
        data['LandSlope'] = 0
        data.loc[fil, 'LandSlope'] = 1
        return data
    
    
    def tr_MSZoning(self, data):
        data['MSZoning'] = data['MSZoning'].map({'RH': 'RM', # medium and high density
                                                 'C (all)': 'RM', # commercial and medium density
                                                 'FV': 'RM'}).fillna(data['MSZoning'])
        return data
    
    
    def tr_Alley(self, data):
        fil = (data.Alley != 'NoAlley')
        data['Alley'] = 0
        data.loc[fil, 'Alley'] = 1
        return data
    
    
    def tr_LotConfig(self, data):
        data['LotConfig'] = data['LotConfig'].map({'FR3': 'Corner', # corners have 2 or 3 free sides
                                                   'FR2': 'Corner'}).fillna(data['LotConfig'])
        return data
    
    
    def tr_BldgType(self, data):
        data['BldgType'] = data['BldgType'].map({'Twnhs' : 'TwnhsE',
                                                 '2fmCon': 'Duplex'}).fillna(data['BldgType'])
        return data
    
    
    def tr_MasVnrType(self, data):
        data['MasVnrType'] = data['MasVnrType'].map({'BrkCmn': 'BrkFace'}).fillna(data['MasVnrType'])
        return data


    def tr_HouseStyle(self, data):
        data['HouseStyle'] = data['HouseStyle'].map({'1.5Fin': '1.5Unf',
                                                     '2.5Fin': '2Story',
                                                     '2.5Unf': '2Story',
                                                     'SLvl': 'SFoyer'}).fillna(data['HouseStyle'])
        return data


    def tr_Neighborhood(self, data):
        if self.te_neig:
            data['Neighborhood'] = data['Neighborhood'].map(self.smooth_neig).fillna(self.mean_tot)
        return data
    
    def tr_MSSubClass(self, data):
        if self.te_mssc:
            data['MSSubClass'] = data['MSSubClass'].map(self.smooth_mssc).fillna(self.mean_tot)
        return data
    
    
    def transform(self, X, y=None):
        X = self.tr_GrgType(X)
        X = self.tr_LotShape(X)
        X = self.tr_LotConfig(X)
        X = self.tr_MSZoning(X)
        X = self.tr_Alley(X)
        X = self.tr_LandCont(X)
        X = self.tr_BldgType(X)
        X = self.tr_MasVnrType(X)
        X = self.tr_HouseStyle(X)
        X = self.tr_Neighborhood(X)
        X = self.tr_MSSubClass(X)
        return X
class general_cleaner(BaseEstimator, TransformerMixin):
    '''
    This class applies what we know from the documetation.
    It cleans some known missing values
    If flags the missing values

    This process is supposed to happen as first step of any pipeline
    '''
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        #LotFrontage
        X.loc[X.LotFrontage.isnull(), 'LotFrontage'] = 0
        #Alley
        X.loc[X.Alley.isnull(), 'Alley'] = "NoAlley"
        #MSSubClass
        X['MSSubClass'] = X['MSSubClass'].astype(str)
        #MissingBasement
        fil = ((X.BsmtQual.isnull()) & (X.BsmtCond.isnull()) & (X.BsmtExposure.isnull()) &
              (X.BsmtFinType1.isnull()) & (X.BsmtFinType2.isnull()))
        fil1 = ((X.BsmtQual.notnull()) | (X.BsmtCond.notnull()) | (X.BsmtExposure.notnull()) |
              (X.BsmtFinType1.notnull()) | (X.BsmtFinType2.notnull()))
        X.loc[fil1, 'MisBsm'] = 0
        X.loc[fil, 'MisBsm'] = 1 # made explicit for safety
        #BsmtQual
        X.loc[fil, 'BsmtQual'] = "NoBsmt" #missing basement
        #BsmtCond
        X.loc[fil, 'BsmtCond'] = "NoBsmt" #missing basement
        #BsmtExposure
        X.loc[fil, 'BsmtExposure'] = "NoBsmt" #missing basement
        #BsmtFinType1
        X.loc[fil, 'BsmtFinType1'] = "NoBsmt" #missing basement
        #BsmtFinType2
        X.loc[fil, 'BsmtFinType2'] = "NoBsmt" #missing basement
        #BsmtFinSF1
        X.loc[fil, 'BsmtFinSF1'] = 0 # No bsmt
        #BsmtFinSF2
        X.loc[fil, 'BsmtFinSF2'] = 0 # No bsmt
        #BsmtUnfSF
        X.loc[fil, 'BsmtUnfSF'] = 0 # No bsmt
        #TotalBsmtSF
        X.loc[fil, 'TotalBsmtSF'] = 0 # No bsmt
        #BsmtFullBath
        X.loc[fil, 'BsmtFullBath'] = 0 # No bsmt
        #BsmtHalfBath
        X.loc[fil, 'BsmtHalfBath'] = 0 # No bsmt
        #FireplaceQu
        X.loc[(X.Fireplaces == 0) & (X.FireplaceQu.isnull()), 'FireplaceQu'] = "NoFire" #missing
        #MisGarage
        fil = ((X.GarageYrBlt.isnull()) & (X.GarageType.isnull()) & (X.GarageFinish.isnull()) &
              (X.GarageQual.isnull()) & (X.GarageCond.isnull()))
        fil1 = ((X.GarageYrBlt.notnull()) | (X.GarageType.notnull()) | (X.GarageFinish.notnull()) |
              (X.GarageQual.notnull()) | (X.GarageCond.notnull()))
        X.loc[fil1, 'MisGarage'] = 0
        X.loc[fil, 'MisGarage'] = 1
        #GarageYrBlt
        X.loc[X.GarageYrBlt > 2200, 'GarageYrBlt'] = 2007 #correct mistake
        X.loc[fil, 'GarageYrBlt'] = X['YearBuilt']  # if no garage, use the age of the building
        #GarageType
        X.loc[fil, 'GarageType'] = "NoGrg" #missing garage
        #GarageFinish
        X.loc[fil, 'GarageFinish'] = "NoGrg" #missing
        #GarageQual
        X.loc[fil, 'GarageQual'] = "NoGrg" #missing
        #GarageCond
        X.loc[fil, 'GarageCond'] = "NoGrg" #missing
        #Fence
        X.loc[X.Fence.isnull(), 'Fence'] = "NoFence" #missing fence
        #Pool
        fil = ((X.PoolArea == 0) & (X.PoolQC.isnull()))
        X.loc[fil, 'PoolQC'] = 'NoPool' 
        
        # not useful features
        del X['Id']
        del X['MiscFeature']  # we already know it doesn't matter
        del X['Condition1']
        del X['Condition2']
        del X['Exterior1st']
        del X['Exterior2nd']
        del X['Functional']
        del X['Heating']
        del X['PoolQC']
        del X['RoofMatl']
        del X['RoofStyle']
        del X['SaleCondition']
        del X['SaleType']
        del X['Utilities']
        del X['BsmtFinType1']
        del X['BsmtFinType2']
        del X['BsmtFinSF1']
        del X['BsmtFinSF2']
        del X['Electrical']
        del X['Foundation']
        del X['Street']
        del X['Fence']
        del X['LandSlope']
        del X['LowQualFinSF']
        del X['FireplaceQu']
        del X['PoolArea']
        del X['MiscVal']
        del X['MoSold']
        del X['YrSold']
        
         # after model iterations
        del X['KitchenAbvGr']
        del X['GarageQual']
        del X['GarageCond'] 
        
        return X
def cv_score(df_train, y_train, kfolds, pipeline, imp_coef=False):
    oof = np.zeros(len(df_train))
    train = df_train.copy()
    feat_df = pd.DataFrame()
    
    print('Training......')
    for n_fold, (train_index, test_index) in enumerate(kfolds.split(train.values)):
            
        trn_data = train.iloc[train_index][:]
        val_data = train.iloc[test_index][:]
        
        trn_target = y_train.iloc[train_index].values.ravel()
        val_target = y_train.iloc[test_index].values.ravel()
        
        pipeline.fit(trn_data, trn_target)

        oof[test_index] = pipeline.predict(val_data).ravel()

        if imp_coef:
            try:
                fold_df = get_coef(pipeline)
            except AttributeError:
                fold_df = get_feature_importance(pipeline)
                
            fold_df['fold'] = n_fold + 1
            feat_df = pd.concat([feat_df, fold_df], axis=0)
       
    if imp_coef:
        feat_df = feat_df.groupby('feat')['score'].agg(['mean', 'std'])
        feat_df['abs_sco'] = (abs(feat_df['mean']))
        feat_df = feat_df.sort_values(by=['abs_sco'],ascending=False)
        del feat_df['abs_sco']
        return oof, feat_df
    else:    
        return oof

def get_coef(pipe):
    imp = pipe.steps[-1][1].coef_.tolist()
    feats = pipe.steps[-2][1].get_feature_names()
    result = pd.DataFrame({'feat':feats,'score':imp})
    result['abs_res'] = abs(result['score'])
    result = result.sort_values(by=['abs_res'],ascending=False)
    del result['abs_res']
    return result


def get_feature_importance(pipe):
    imp = pipe.steps[-1][1].feature_importances_.tolist() #it's a pipeline
    feats = pipe.steps[-2][1].get_feature_names()
    result = pd.DataFrame({'feat':feats,'score':imp})
    result = result.sort_values(by=['score'],ascending=False)
    return result


def make_test(train, test_size, random_state, strat_feat=None):
    if strat_feat:
        
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

        for train_index, test_index in split.split(train, train[strat_feat]):
            train_set = train.loc[train_index]
            test_set = train.loc[test_index]
            
    return train_set, test_set

def test():
    df_train=pd.read_csv('./train.csv',encoding='utf-8')
    df_train['target'] = np.log1p(df_train.SalePrice)
    del df_train['SalePrice']

    train_set, test_set = make_test(df_train, 
                                    test_size=0.2, random_state=654, 
                                    strat_feat='Neighborhood')
    folds = KFold(5, shuffle=True, random_state=541)
    y = train_set['target'].copy()
    del train_set['target']
    y_test=test_set['target'].copy()
    del test_set['target']

    numeric_forest = Pipeline([('fs', dfp.feat_sel('numeric')),
                            ('imp', dfp.df_imputer(strategy='median')),
                            ('transf', tr_numeric(SF_room=False,
                                                bedroom=False, 
                                                lot=False))])

    cat_forest = Pipeline([('fs', dfp.feat_sel('category')),
                        ('imp', dfp.df_imputer(strategy='most_frequent')), 
                        ('ord', make_ordinal(['BsmtQual', 'KitchenQual', 'ExterQual', 'HeatingQC'], 
                                            extra_cols=['BsmtExposure', 'BsmtCond', 'ExterCond'],
                                            include_extra='include')), 
                        ('recode', recode_cat()), 
                        ('dummies', dfp.dummify(drop_first=True))])


    processing_forest = dfp.FeatureUnion_df(transformer_list=[('cat', cat_forest),
                                                    ('num', numeric_forest)])

    forest_pipe = Pipeline([('gen_cl', general_cleaner()),
                        ('proc', processing_forest),
                        ('scaler', dfp.df_scaler(method='robust')),
                        ('dropper', drop_columns(forest=True))])
    model=Lasso(alpha=0.001, tol=0.005)

    
    #                        ('select', select_feature()),
                        # ('forest', RandomForestRegressor(n_estimators=1500, max_depth=30, 
                        #                                     max_features='sqrt',
                        #                                     n_jobs=-1, random_state=32))])
    # forest_oof, coefs = cv_score(train_set, y, folds, forest_pipe, imp_coef=True)

    # print(f'Train set RMSE: {round(np.sqrt(mean_squared_error(y, forest_oof)), 4)}')
    # print(f'Train set MAE: {round(mean_absolute_error(np.expm1(y), np.expm1(forest_oof)), 2)}')

    # tmp = train_set.copy()
    # d=forest_pipe.fit_transform(tmp,y)
    # model.fit(d,y)
    # # save the model to disk
    filename = 'finalized_model.sav'
    # pickle.dump(model, open(filename, 'wb'))
    loaded_model = pickle.load(open(filename, 'rb'))
    tmp2=test_set.copy()
    d2=forest_pipe.fit_transform(tmp2,y_test)
    # preds=model.predict(d2.iloc[0:1])
    preds=loaded_model.predict(d2.iloc[0:1])
    output=np.asscalar(np.expm1(preds))
    output='${:.4f}'.format(output)
    return output

    '''
    tmp = test_set.copy()
    # preds = forest_pipe.predict(tmp)
    # print(f'Test set RMSE: {round(np.sqrt(mean_squared_error(y_test, preds)), 4)}')
    # print(f'Test set MAE: {round(mean_absolute_error(np.expm1(y_test), np.expm1(preds)), 2)}')
    preds=forest_pipe.predict(tmp.iloc[0:1]) 
    output=np.asscalar(np.expm1(preds))
    return round(output,4)
    '''
# if __name__ == "__main__":
#     x=test()
#     print(x)