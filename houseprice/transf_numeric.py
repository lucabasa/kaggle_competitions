__author__ = 'lucabasa'
__version__ = '1.0.1'
__status__ = 'development'


import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class tr_numeric(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.columns = []  # useful to well behave with FeatureUnion
        
    def fit(self, X, y=None):
        return self
    
    def remove_skew(self, X, column):
        X[column] = np.log1p(X[column])
        return X
    
    def transform(self, X, y=None):
        for col in ['GrLivArea', '1stFlrSF']:
            X = self.remove_skew(X, col)
        self.columns = X.columns
        return X
    
    def get_features_name(self):
        return self.columns
