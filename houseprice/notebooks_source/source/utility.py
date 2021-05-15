__author__ = 'lucabasa'
__version__ = '1.3.1'
__status__ = 'development'


import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV, train_test_split

import source.report as rp


def make_test(train, test_size, random_state, strat_feat=None):
    '''
    Creates a train and test, stratified on a feature or on a list of features
    todo: allow for non-stratified splits
    '''
    if strat_feat:
        
        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

        for train_index, test_index in split.split(train, train[strat_feat]):
            train_set = train.loc[train_index]
            test_set = train.loc[test_index]
            
    else:
        train_set, test_set = train_test_split(train, test_size=test_size, random_state=random_state)
            
    return train_set, test_set


def cv_score(df_train, y_train, kfolds, pipeline, imp_coef=False, predict_proba=False):
    '''
    Train and test a pipeline in kfold cross validation
    Returns the oof predictions for the entire train set and a dataframe with the
    coefficients or feature importances, averaged across the folds, with standard deviation

    todo: report on the average score and variance too
    '''
    oof = np.zeros(len(df_train))
    train = df_train.copy()
    
    feat_df = pd.DataFrame()
    
    for n_fold, (train_index, test_index) in enumerate(kfolds.split(train.values)):
            
        trn_data = train.iloc[train_index][:]
        val_data = train.iloc[test_index][:]
        
        trn_target = y_train.iloc[train_index].values.ravel()
        val_target = y_train.iloc[test_index].values.ravel()
        
        pipeline.fit(trn_data, trn_target)
        
        if predict_proba:
            oof[test_index] = pipeline.predict_proba(val_data)[:,1]
        else:
            oof[test_index] = pipeline.predict(val_data).ravel()

        if imp_coef:
            try:
                fold_df = rp.get_coef(pipeline)
            except (AttributeError, KeyError):
                fold_df = rp.get_feature_importance(pipeline)
                
            fold_df['fold'] = n_fold + 1
            feat_df = pd.concat([feat_df, fold_df], axis=0)
       
    if imp_coef:
        feat_df = feat_df.groupby('feat')['score'].agg(['mean', 'std'])
        feat_df['abs_sco'] = (abs(feat_df['mean']))
        feat_df = feat_df.sort_values(by=['abs_sco'],ascending=False)
        del feat_df['abs_sco']
        return oof, feat_df
    else:    
        return oof


def grid_search(data, target, estimator, param_grid, scoring, cv, random=False):
    '''
    Calls a grid or a randomized search over a parameter grid
    Returns a dataframe with the results for each configuration
    Returns a dictionary with the best parameters
    Returns the best (fitted) estimator
    '''
    
    if random:
        grid = RandomizedSearchCV(estimator=estimator, param_distributions=param_grid, cv=cv, scoring=scoring, 
                                  n_iter=random, n_jobs=-1, random_state=434, iid=False, return_train_score=True)
    else:
        grid = GridSearchCV(estimator=estimator, param_grid=param_grid, 
                            cv=cv, scoring=scoring, n_jobs=-1, return_train_score=True)
    
    pd.options.mode.chained_assignment = None  # turn on and off a warning of pandas
    tmp = data.copy()
    grid = grid.fit(tmp, target)
    pd.options.mode.chained_assignment = 'warn'
    
    result = pd.DataFrame(grid.cv_results_).sort_values(by='mean_test_score', 
                                                        ascending=False).reset_index()
    
    del result['params']
    times = [col for col in result.columns if col.endswith('_time')]
    params = [col for col in result.columns if col.startswith('param_')]
    
    result = result[params + ['mean_train_score', 'std_train_score', 'mean_test_score', 'std_test_score'] + times]
    
    return result, grid.best_params_, grid.best_estimator_