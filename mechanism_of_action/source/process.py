__author__ = 'lucabasa'
__version__ = '1.1.0'

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


def process_data(data, features_g, features_c):
    df = data.copy()
    
    df['g_sum'] = df[features_g].sum(axis = 1)
    df['g_mean'] = df[features_g].mean(axis = 1)
    df['g_std'] = df[features_g].std(axis = 1)
    df['g_kurt'] = df[features_g].kurtosis(axis = 1)
    df['g_skew'] = df[features_g].skew(axis = 1)
    df['c_sum'] = df[features_c].sum(axis = 1)
    df['c_mean'] = df[features_c].mean(axis = 1)
    df['c_std'] = df[features_c].std(axis = 1)
    df['c_kurt'] = df[features_c].kurtosis(axis = 1)
    df['c_skew'] = df[features_c].skew(axis = 1)
    df['gc_sum'] = df[features_g + features_c].sum(axis = 1)
    df['gc_mean'] = df[features_g + features_c].mean(axis = 1)
    df['gc_std'] = df[features_g + features_c].std(axis = 1)
    df['gc_kurt'] = df[features_g + features_c].kurtosis(axis = 1)
    df['gc_skew'] = df[features_g + features_c].skew(axis = 1)
    
    for feature in features_c:
        df[f'{feature}_squared'] = df[feature] ** 2
    
    return df


def add_pca(train_df, valid_df, test_df, g_comp, c_comp, g_feat, c_feat, add=True):
    
    # GENES
    
    pca = PCA(n_components=g_comp, random_state=1903)
    pipe = Pipeline([('scal', DfScaler(method='robust')), ('pca', pca)])
    train2 = pipe.fit_transform(train_df[g_feat])
    valid2 = pipe.transform(valid_df[g_feat])
    test2 = pipe.transform(test_df[g_feat])

    train2 = pd.DataFrame(train2, columns=[f'pca_G-{i}' for i in range(g_comp)])
    valid2 = pd.DataFrame(valid2, columns=[f'pca_G-{i}' for i in range(g_comp)])
    test2 = pd.DataFrame(test2, columns=[f'pca_G-{i}' for i in range(g_comp)])
    
    if add:
        train_df = pd.concat((train_df, train2), axis=1)
        valid_df = pd.concat((valid_df, valid2), axis=1)
        test_df = pd.concat((test_df, test2), axis=1)
    else:
        train_g = train2
        valid_g = valid2
        test_g = test2

    #CELLS

    pca = PCA(n_components=c_comp, random_state=1903)
    pipe = Pipeline([('scal', DfScaler(method='robust')), ('pca', pca)])
    train2 = pipe.fit_transform(train_df[c_feat])
    valid2 = pipe.transform(valid_df[c_feat])
    test2 = pipe.transform(test_df[c_feat])

    train2 = pd.DataFrame(train2, columns=[f'pca_C-{i}' for i in range(c_comp)])
    valid2 = pd.DataFrame(valid2, columns=[f'pca_C-{i}' for i in range(c_comp)])
    test2 = pd.DataFrame(test2, columns=[f'pca_C-{i}' for i in range(c_comp)])
    
    if add:
        train_df = pd.concat((train_df, train2), axis=1)
        valid_df = pd.concat((valid_df, valid2), axis=1)
        test_df = pd.concat((test_df, test2), axis=1)
    else:
        other_cols = [col for col in train_df if col not in c_feat and col not in g_feat]
        train_df = pd.concat([train_df[other_cols], train_g, train2], axis=1)
        valid_df = pd.concat([valid_df[other_cols], valid_g, valid2], axis=1)
        test_df = pd.concat([test_df[other_cols], test_g, test2], axis=1)
    
    return train_df, valid_df, test_df


def var_tr(train_df, valid_df, test_df, thr, cat_cols):
    
    var_thresh = VarianceThreshold(threshold=thr)

    train_transformed = var_thresh.fit_transform(train_df[[col for col in train_df if col not in cat_cols]])
    valid_transformed = var_thresh.transform(valid_df[[col for col in valid_df if col not in cat_cols]])
    test_transformed = var_thresh.transform(test_df[[col for col in test_df if col not in cat_cols]])

    train_features = train_df[cat_cols]
    train_features = pd.concat([train_features, pd.DataFrame(train_transformed)], axis=1)
    
    valid_features = valid_df[cat_cols]
    valid_features = pd.concat([valid_features, pd.DataFrame(valid_transformed)], axis=1)

    test_features = test_df[cat_cols]
    test_features = pd.concat([test_features, pd.DataFrame(test_transformed)], axis=1)
    
    return train_features, valid_features, test_features


class DfScaler(BaseEstimator, TransformerMixin):
    '''
    Wrapper of several sklearn scalers that keeps the dataframe structure
    '''
    def __init__(self, method='standard', feature_range=(0,1), n_quantiles=1000, output_distribution='uniform', random_state=345):
        super().__init__()
        self.method = method
        self._validate_input()
        self.scale_ = None
        if self.method == 'standard':
            self.scl = StandardScaler()
            self.mean_ = None
        elif method == 'robust':
            self.scl = RobustScaler()
            self.center_ = None
        elif method == 'minmax':
            self.feature_range = feature_range
            self.scl = MinMaxScaler(feature_range=self.feature_range)
            self.min_ = None
            self.data_min_ = None
            self.data_max_ = None
            self.data_range_ = None
            self.n_samples_seen_ = None
        elif method == 'quantile':
            self.n_quantiles = n_quantiles
            self.output_distribution = output_distribution
            self.random_state = random_state
            self.scl = QuantileTransformer(n_quantiles=self.n_quantiles, 
                                           output_distribution=self.output_distribution, 
                                           random_state=self.random_state)
            self.n_quantiles_ = None
            self.quantiles_ = None
            self.references_ = None

            
    def _validate_input(self):
        allowed_methods = ["standard", 'robust', 'minmax', 'quantile']
        if self.method not in allowed_methods:
            raise ValueError(f"Can only use these methods: {allowed_methods} got method={self.method}")
    
    
    def fit(self, X, y=None):
        self.scl.fit(X)
        if self.method == 'quantile':
            return self
        if self.method == 'standard':
            self.mean_ = pd.Series(self.scl.mean_, index=X.columns)
        elif self.method == 'robust':
            self.center_ = pd.Series(self.scl.center_, index=X.columns)
        elif self.method == 'minmax':
            self.min_ = pd.Series(self.scl.min_, index=X.columns)
            self.data_min_ = pd.Series(self.scl.data_min_, index=X.columns)
            self.data_max_ = pd.Series(self.scl.data_max_, index=X.columns)
            self.data_range_ = self.data_max_ - self.data_min_
            self.n_samples_seen_ = X.shape[0]
        self.scale_ = pd.Series(self.scl.scale_, index=X.columns)
        return self
    
    
    def transform(self, X, y=None):
        # assumes X is a DataFrame
        Xscl = self.scl.transform(X)
        Xscaled = pd.DataFrame(Xscl, index=X.index, columns=X.columns)
        return Xscaled
    
    
def scale_data(train, valid, test):
    
    scl = DfScaler(method='robust')
    scaled_train = scl.fit_transform(train[[col for col in train if col!='sig_id']])
    scaled_train = pd.concat([train[['sig_id']], scaled_train], axis=1)
    
    scaled_valid = scl.transform(valid[[col for col in valid if col!='sig_id']])
    scaled_valid = pd.concat([valid[['sig_id']], scaled_valid], axis=1)
    
    scaled_test = scl.transform(test[[col for col in test if col!='sig_id']])
    scaled_test = pd.concat([test[['sig_id']], scaled_test], axis=1)
    
    return scaled_train, scaled_valid, scaled_test
