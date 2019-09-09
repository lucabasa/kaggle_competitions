__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'


import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class feat_sel(BaseEstimator, TransformerMixin):
	'''
	This transformer selects either numerical or categorical features.
	In this way we can build separate pipelines for separate data types.
	'''
    def __init__(self, dtype='numeric'):
        self._dtype = dtype
 
    def fit( self, X, y=None ):
        return self 
    
    def transform(self, X, y=None):
        if self._dtype == 'numeric':
            num_cols = X.columns[X.dtypes != object].tolist()
            return X[num_cols]
        elif self._dtype == 'category':
            cat_cols = X.columns[X.dtypes == object].tolist()
            return X[cat_cols]


class df_imputer(TransformerMixin):
	'''
	Just a wrapper for the SimpleImputer that keeps the dataframe structure
	'''
    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imp = None
        self.statistics_ = None

    def fit(self, X, y=None):
        self.imp = SimpleImputer(strategy=self.strategy)
        self.imp.fit(X)
        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Ximp = self.imp.transform(X)
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        return Xfilled

    
class df_scaler(TransformerMixin):
	'''
	Wrapper of StandardScaler or RobustScaler
	'''
    def __init__(self, method='standard'):
        self.scl = None
        self.scale_ = None
        self.method = method
        if self.method == 'sdandard':
            self.mean_ = None
        elif method == 'robust':
            self.center_ = None

    def fit(self, X, y=None):
        if self.method == 'standard':
            self.scl = StandardScaler()
            self.scl.fit(X)
            self.mean_ = pd.Series(self.scl.mean_, index=X.columns)
        elif self.method == 'robust':
            self.scl = RobustScaler()
            self.scl.fit(X)
            self.center_ = pd.Series(self.scl.center_, index=X.columns)
        self.scale_ = pd.Series(self.scl.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xscl = self.scl.transform(X)
        Xscaled = pd.DataFrame(Xscl, index=X.index, columns=X.columns)
        return Xscaled

    
class make_ordinal(BaseEstimator, TransformerMixin):
	'''
	Transforms ordinal features in order to have them as numeric (preserving the order)
	If unsure about converting or not a feature (maybe making dummies is better), make use of
	extra_cols and unsure_conversion
	'''
    def __init__(self, cols, extra_cols=None, unsure_conversion=True):
        self._unsure_conversion = unsure_conversion
        self.cols = cols
        self.extra_cols = extra_cols
        self.mapping = {'Po':1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        if self.extra_cols and self.unsure_conversion:
            self.cols += self.extra_cols
        for col in self.cols:
            X.loc[:, col] = X[col].map(self.mapping).fillna(0)
        return X
