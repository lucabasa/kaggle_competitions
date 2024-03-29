__author__ = 'lucabasa'
__version__ = '2.0.0'
__status__ = 'development'


import pandas as pd
import numpy as np

from sklearn.base import clone
from sklearn.inspection import partial_dependence
from sklearn.exceptions import NotFittedError


def get_pdp(clf, feature, data, fold):
    res = partial_dependence(clf, features=feature, 
                                   X=data, grid_resolution=50, kind='average')
    fold_tmp = pd.DataFrame({'x': res['values'][0], 'y': res['average'][0]})
    fold_tmp['feat'] = feature
    fold_tmp['fold'] = fold + 1
    
    return fold_tmp


def train_model(train_df, test_df, target, trsf_pipe, estimator, cv, early_stopping=100, eval_metric='rmse',
                weight=None, predict_proba=False, verbose=False, pdp=None, pdp_round=10):
    
    oof = np.zeros(len(train_df))
    pred = np.zeros(len(test_df))
    train = train_df.copy()
    test = test_df.copy()
    if weight is not None:
        w = pd.DataFrame({'w': weight}, index=train_df.index)
    else:
        w = pd.DataFrame({'w': [1]*len(train_df)}, index=train_df.index)
    
    rep_res = {}
    
    feat_df = pd.DataFrame()
    feat_pdp = pd.DataFrame()
    iteration = []
    
    for n_fold, (train_index, test_index) in enumerate(cv.split(train.values)):
        
        trn_data = train.iloc[train_index, :]
        val_data = train.iloc[test_index, :]
        trn_weight = w.iloc[train_index, 0].values.ravel()
        val_weight = w.iloc[test_index, 0].values.ravel()
        
        trn_target = target.iloc[train_index].values.ravel()
        val_target = target.iloc[test_index].values.ravel()      
        
        model = clone(estimator)
        pipe = clone(trsf_pipe)
        
        trn_set = pipe.fit_transform(trn_data, trn_target)
        val_set = pipe.transform(val_data)
        test_set = pipe.transform(test)
        
        try:
            model.fit(trn_set, trn_target, 
                      eval_set=[(trn_set, trn_target), (val_set, val_target)], 
                      eval_metric=eval_metric, 
                      sample_weight=trn_weight,
                      eval_sample_weight=[val_weight],
                      early_stopping_rounds=early_stopping,
                      verbose=verbose
                     )
        except TypeError:
            model.fit(trn_set, trn_target, 
                      eval_set=[(trn_set, trn_target), (val_set, val_target)], 
                      eval_metric=eval_metric, 
                      sample_weight=trn_weight,
                      #sample_weight_eval_set=[val_weight],
                      early_stopping_rounds=early_stopping,
                      verbose=verbose
                     )
        
        if predict_proba:
            oof[test_index] = model.predict_proba(val_set)[:,1]
            pred += model.predict_proba(test_set)[:,1] / cv.get_n_splits()
        else:
            oof[test_index] = model.predict(val_set).ravel()
            pred += model.predict(test_set).ravel() / cv.get_n_splits()
        
        #store iteration used
        try:
            iteration.append(model.best_iteration)
        except AttributeError:
            iteration.append(model.best_iteration_)
        
        # FIXME: pdp does not work with LightGBM
        if pdp is not None:
            pdp_set = pipe.transform(train)
            for feat in pdp:
                try:
                    fold_tmp = get_pdp(model, feat, pdp_set, n_fold)
                    feat_pdp = pd.concat([feat_pdp, fold_tmp], axis=0)
                except NotFittedError:
                    pdp = None
                    break
        
        # store feature importance
        fold_df = pd.DataFrame()
        #fold_df['feat'] = trn_set.columns
        fold_df['feat'] = pipe.steps[-1][1].get_feature_names()
        fold_df['score'] = model.feature_importances_
        feat_df = pd.concat([feat_df, fold_df], axis=0)

    # variable importance averaged over folds
    feat_df = feat_df.groupby(['feat'])['score'].agg(['mean', 'std'])
    feat_df = feat_df.sort_values(by=['mean'], ascending=False)
    feat_df['std'] = feat_df['std'] / np.sqrt(cv.get_n_splits() - 1)  # std of the mean, unbiased
    
    # pdp averaged over folds
    if pdp is not None:
        feat_pdp['x'] = round(feat_pdp['x'], pdp_round)
        feat_pdp = feat_pdp.groupby(['feat', 'x'])['y'].agg(['mean', 'std']).reset_index()
        feat_pdp['std'] = feat_pdp['std'] / np.sqrt(cv.get_n_splits() - 1)

    rep_res['feat_imp'] = feat_df
    rep_res['n_iterations'] = iteration
    rep_res['pdp'] = feat_pdp

    return oof, pred, rep_res