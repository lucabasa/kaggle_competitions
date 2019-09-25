__author__ = 'lucabasa'
__version__ = '1.2.0'
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
        self.columns = None  # this is useful when it is the last step of a pipeline before the model

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
        self.columns = X.columns
        return Xscaled
    
    def get_feature_names(self):
        return list(self.columns)

 
 class FeatureUnion_df(TransformerMixin, BaseEstimator):
    '''
    Wrapper of FeatureUnion but returning a Dataframe, 
    the column order follows the concatenation done by FeatureUnion

    transformer_list: list of Pipelines

    '''
    def __init__(self, transformer_list):
        self.transformer_list = transformer_list
        self.feat_un = FeatureUnion(self.transformer_list)
        
    def fit(self, X, y=None):
        self.feat_un.fit(X)
        return self
    
    def transform(self, X, y=None):
        X_tr = self.feat_un.transform(X)
        columns = []
        
        for trsnf in self.transformer_list:
            cols = trsnf[1].steps[-1][1].get_features_name()
            columns += list(cols)

        X_tr = pd.DataFrame(X_tr, index=X.index, columns=columns)
        
        return X_tr
    
    def get_params(self, deep=True):  # necessary to well behave in GridSearch
        return self.feat_un.get_params(deep=deep)
