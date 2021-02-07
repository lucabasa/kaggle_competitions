__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'

from tubesml.base import BaseTransformer, self_columns, reset_columns

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
    