__author__ = 'lucabasa'
__version__ = '1.1.1'
__status__ = 'development'


import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class general_cleaner(BaseEstimator, TransformerMixin):
    '''
    This class applies what we know from the documetation.
    It cleans some known missing values
    If flags the missing values

    This process is supposed to happen as first step of any pipeline
    '''
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, df, y=None):
        X = df.copy()
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
