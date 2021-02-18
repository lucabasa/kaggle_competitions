__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'


import pandas as pd
import numpy as np
from sklearn.base import clone


def train_model(train_df, test_df, target, trsf_pipe, estimator, cv, early_stopping=100, verbose=False):
    
    oof = np.zeros(len(train_df))
    pred = np.zeros(len(test_df))
    train = train_df.copy()
    test = test_df.copy()
    
    feat_df = pd.DataFrame()
    
    for n_fold, (train_index, test_index) in enumerate(cv.split(train.values)):
        
        trn_data = train.iloc[train_index, :]
        val_data = train.iloc[test_index, :]
        
        trn_target = target.iloc[train_index].values.ravel()
        val_target = target.iloc[test_index].values.ravel()      
        
        model = clone(estimator)
        pipe = clone(trsf_pipe)
        
        trn_set = pipe.fit_transform(trn_data, trn_target)
        val_set = pipe.transform(val_data)
        test_set = pipe.transform(test)
        
        model.fit(trn_set, trn_target, 
                  eval_set=[(trn_set, trn_target), (val_set, val_target)], 
                  eval_metric='rmse', 
                  early_stopping_rounds=early_stopping,
                  verbose=verbose
                 )
        
        oof[test_index] = model.predict(val_set).ravel()
        pred += model.predict(test_set).ravel() / cv.get_n_splits()
        
        # store feature importance
        fold_df = pd.DataFrame()
        fold_df['feat'] = trn_set.columns
        fold_df['score'] = model.feature_importances_
        feat_df = pd.concat([feat_df, fold_df], axis=0)

    # variable importance averaged over folds
    feat_df = feat_df.groupby(['feat'])['score'].agg(['mean', 'std'])
    feat_df = feat_df.sort_values(by=['mean'], ascending=False)
    feat_df['std'] = feat_df['std'] / np.sqrt(cv.get_n_splits() - 1)  # std of the mean, unbiased


    return oof, pred, feat_df
