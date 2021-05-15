__author__ = 'lucabasa'
__version__ = '1.2.0'
__status__ = 'development'

from tubesml.base import BaseTransformer, self_columns, reset_columns

from sklearn.decomposition import PCA
import pandas as pd

class CatSimp(BaseTransformer):
    def __init__(self, cat7=True, cat6=True, cat8=True, cat4=True, cat9=True):
        super().__init__()
        self.cat7 = cat7
        self.cat6 = cat6
        self.cat8 = cat8
        self.cat4 = cat4
        self.cat9 = cat9
     
    
    def cat7_tr(self, X):
        X_tr = X.copy()
        
        if self.cat7:
            X_tr['cat7'] = X_tr['cat7'].map({'C': 'E', 
                                             'A': 'B', 
                                             'F': 'G', 
                                             'I': 'G'}).fillna(X_tr['cat7'])
        
        return X_tr
    
    
    def cat6_tr(self, X):
        X_tr = X.copy()
        
        if self.cat6:
            X_tr.loc[X_tr['cat6'] != 'A', 'cat6'] = 'B'
        
        return X_tr
    
    
    def cat8_tr(self, X):
        X_tr = X.copy()
        
        if self.cat8:
            X_tr['cat8'] = X_tr['cat8'].map({'B': 'E', 'F': 'E'}).fillna(X_tr['cat8'])
        
        return X_tr
    
    
    def cat4_tr(self, X):
        X_tr = X.copy()
        
        if self.cat4:
            X_tr['cat4'] = X_tr['cat4'].map({'D': 'A'}).fillna(X_tr['cat4'])
        
        return X_tr
    
    
    def cat9_tr(self, X):
        X_tr = X.copy()
        
        if self.cat9:
            X_tr['cat9'] = X_tr['cat9'].map({'E': 'L', 'D': 'J', 'C': 'L'}).fillna(X_tr['cat9'])
        
        return X_tr
    
    @self_columns
    def transform(self, X, y=None):
        
        Xtransf = self.cat7_tr(X)
        Xtransf = self.cat6_tr(Xtransf)
        Xtransf = self.cat8_tr(Xtransf)
        Xtransf = self.cat4_tr(Xtransf)
        Xtransf = self.cat9_tr(Xtransf)
        
        return Xtransf
    
    
class PCADf(BaseTransformer):
    def __init__(self, n_components, svd_solver='auto', random_state=24, compress=False):
        super().__init__()
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.n_components_ = 0
        self.random_state = random_state
        self.PCA = PCA(n_components=self.n_components, svd_solver=self.svd_solver, random_state=self.random_state)
        self.compress = compress
        self.original_columns = []
        
    @reset_columns
    def fit(self, X, y=None):
        
        self.PCA.fit(X)
        self.n_components_ = self.PCA.n_components_
        
        return self
    
    @self_columns
    def transform(self, X, y=None):
                
        X_tr = self.PCA.transform(X)
        X_tr = pd.DataFrame(X_tr, columns=[f'pca_{i}' for i in range(self.n_components_)])
        
        self.original_columns = X.columns
        
        if self.compress:
            X_tr = self.inverse_transform(X_tr)
        
        return X_tr
    
    
    def inverse_transform(self, X, y=None):
        
        try:
            X_tr = self.PCA.inverse_transform(X)
        except ValueError:
            return X
        X_tr = pd.DataFrame(X_tr, columns=self.original_columns)
        
        return X_tr

    
class TargetEncoder(BaseTransformer):
    # Heavily inspired by 
    # https://github.com/MaxHalford/xam/blob/93c066990d976c7d4d74b63fb6fb3254ee8d9b48/xam/feature_extraction/encoding/bayesian_target.py#L8
    def __init__(self, to_encode=None, prior_weight=100, agg_func='mean'):
        super().__init__()
        if isinstance(to_encode, str):
            self.to_encode = [to_encode]
        else:
            self.to_encode = to_encode
        self.prior_weight = prior_weight
        self.prior_ = None
        self.posteriors_ = None
        self.agg_func = agg_func
     
    
    @reset_columns
    def fit(self, X, y):
        # Encode all categorical cols by default
        if self.to_encode is None:
            self.to_encode = [c for c in X if str(X[c].dtype)=='object' or str(X[c].dtype)=='category']
        
        tmp = X.copy()
        tmp['target'] = y
        
        self.prior_ = tmp['target'].agg(self.agg_func)
        self.posteriors_ = {}
        
        for col in self.to_encode:
            
            agg = tmp.groupby(col)['target'].agg(['count', self.agg_func])
            counts = agg['count']
            data = agg[self.agg_func]
            pw = self.prior_weight
            self.posteriors_[col] = ((pw * self.prior_ + counts * data) / (pw + counts)).to_dict()
        
        del tmp
        return self
    
    
    @self_columns
    def transform(self, X, y=None):
        
        X_tr = X.copy()
        
        for col in self.to_encode:
            X_tr[col] = X_tr[col].map(self.posteriors_[col]).fillna(self.prior_).astype(float)
        
        return X_tr
