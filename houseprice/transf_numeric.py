__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'


import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class tr_numeric(BaseEstimator, TransformerMixin):
    def __init__(self, columns=['GrLivArea', '1stFlrSF']):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def remove_skew(self, X, column):
        X[column] = np.log1p(X[column])
        return X
    
    def transform(self, X, y=None):
        for col in self.columns:
            X = self.remove_skew(X, col)
        return X
