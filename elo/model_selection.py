__author__ = 'lucabasa'
__version__ = '1.0'
__status__ = 'development'


import numpy as np 
import pandas as pd 

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV

import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


def lightgbm_train(train, test, target, kfolds):
    
    param = {'num_leaves': 111,
         'min_data_in_leaf': 150,
         'objective': 'regression',
         'max_depth': 9,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7083 ,
         "bagging_seed": 11,
         "metric": 'rmse', 
         "lambda_l1": 0.2634,
         "random_seed": 133,
         "verbosity": -1}
         
    '''
    param = {'num_leaves': 50,
         'min_data_in_leaf': 11,
         'objective': 'regression',
         'max_depth': 5,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.8791,
         "bagging_freq": 1,
         "bagging_fraction": 0.9238 ,
         "bagging_seed": 11,
         "metric": 'rmse', 
         "lambda_l1": 4.8679,
         "random_seed": 133,
         "verbosity": -1}
    '''

    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    comm_cols = list(set(train.columns).intersection(test.columns))

    for fold_, (trn_idx, val_idx) in enumerate(kfolds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))
        
        trn_data = lgb.Dataset(train.iloc[trn_idx][comm_cols],
                               label=target.iloc[trn_idx]
                              )
        val_data = lgb.Dataset(train.iloc[val_idx][comm_cols],
                               label=target.iloc[val_idx]
                              )

        num_round = 10000
        clf = lgb.train(param,
                        trn_data,
                        num_round,
                        valid_sets = [trn_data, val_data],
                        verbose_eval=500,
                        early_stopping_rounds = 300)
        
        oof[val_idx] = clf.predict(train.iloc[val_idx][comm_cols], num_iteration=clf.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = comm_cols
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        predictions += clf.predict(test[comm_cols], num_iteration=clf.best_iteration) / kfolds.n_splits

    print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))

    return predictions, mean_squared_error(oof, target)**0.5, feature_importance_df, oof


def lightgbm_dart(train, test, target, kfolds):
    param = {'num_leaves': 111,
         'min_data_in_leaf': 150,
         'objective': 'regression',
         'max_depth': 9,
         'learning_rate': 0.005,
         "boosting": "dart",
         "feature_fraction": 0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7083 ,
         "bagging_seed": 11,
         "metric": 'rmse', 
         "lambda_l1": 0.2634,
         "random_seed": 133,
         "verbosity": -1}

    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    comm_cols = list(set(train.columns).intersection(test.columns))

    for fold_, (trn_idx, val_idx) in enumerate(kfolds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))
        
        trn_data = lgb.Dataset(train.iloc[trn_idx][comm_cols],
                               label=target.iloc[trn_idx]
                              )
        val_data = lgb.Dataset(train.iloc[val_idx][comm_cols],
                               label=target.iloc[val_idx]
                              )

        num_round = 10000
        clf = lgb.train(param,
                        trn_data,
                        num_round,
                        valid_sets = [trn_data, val_data],
                        verbose_eval=500,
                        early_stopping_rounds = 200)
        
        oof[val_idx] = clf.predict(train.iloc[val_idx][comm_cols], num_iteration=clf.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = comm_cols
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        predictions += clf.predict(test[comm_cols], num_iteration=clf.best_iteration) / kfolds.n_splits

    print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))

    return predictions, mean_squared_error(oof, target)**0.5, feature_importance_df, oof


def lightgbm_rf(train, test, target, kfolds):
    param = {'num_leaves': 111,
         'min_data_in_leaf': 150,
         'objective': 'regression',
         'max_depth': 9,
         'learning_rate': 0.005,
         "boosting": "rf",
         "feature_fraction": 0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7083 ,
         "bagging_seed": 11,
         "metric": 'rmse', 
         "lambda_l1": 0.2634,
         "random_seed": 133,
         "verbosity": -1}

    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    comm_cols = list(set(train.columns).intersection(test.columns))

    for fold_, (trn_idx, val_idx) in enumerate(kfolds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))
        
        trn_data = lgb.Dataset(train.iloc[trn_idx][comm_cols],
                               label=target.iloc[trn_idx]
                              )
        val_data = lgb.Dataset(train.iloc[val_idx][comm_cols],
                               label=target.iloc[val_idx]
                              )

        num_round = 10000
        clf = lgb.train(param,
                        trn_data,
                        num_round,
                        valid_sets = [trn_data, val_data],
                        verbose_eval=500,
                        early_stopping_rounds = 200)
        
        oof[val_idx] = clf.predict(train.iloc[val_idx][comm_cols], num_iteration=clf.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = comm_cols
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        predictions += clf.predict(test[comm_cols], num_iteration=clf.best_iteration) / kfolds.n_splits

    print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))

    return predictions, mean_squared_error(oof, target)**0.5, feature_importance_df, oof


def xgb_train(train, test, target, kfolds):

    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    comm_cols = list(set(train.columns).intersection(test.columns))

    for fold_, (trn_idx, val_idx) in enumerate(kfolds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))
        
        trn_data = train.iloc[trn_idx][comm_cols]
        val_data = train.iloc[val_idx][comm_cols]
        
        trn_target = target.iloc[trn_idx]
        val_target = target.iloc[val_idx]
    
        clf = xgb.XGBRegressor(n_estimators=10000, 
                               learning_rate=0.05,
                               max_depth=6, 
                               n_jobs=6, 
                               subsample=0.99, 
                               random_state=408, 
                               gamma=0.0217, 
                               reg_alpha=0.9411,
                               colsample_bytree=0.3055).fit(trn_data, trn_target, 
                                                                      eval_set=[(val_data, val_target)], 
                                                                      eval_metric='rmse',
                                                                      early_stopping_rounds=200, 
                                                                      verbose=500)
        
        oof[val_idx] = clf.predict(train.iloc[val_idx][comm_cols],
                                   ntree_limit=clf.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = comm_cols
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        predictions += clf.predict(test[comm_cols], ntree_limit=clf.best_iteration) / kfolds.n_splits

    print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))

    return predictions, mean_squared_error(oof, target)**0.5, feature_importance_df, oof


def rf_train(train, test, target, kfolds):
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    comm_cols = list(set(train.columns).intersection(test.columns))
    '''
    grid_param = {'max_depth': np.arange(3,30),
                    'min_samples_split': np.arange(2, 50), 
                    'min_samples_leaf': np.arange(1,40), 
                    'max_features': ['sqrt', 'log2', None]}

    print('Optimizing parameters')

    grid = RandomizedSearchCV(RandomForestRegressor(n_estimators=300, n_jobs=4, random_state=345),
                                param_distributions=grid_param, n_iter=20, cv=kfolds, 
                                random_state=654, n_jobs=-1, scoring='neg_mean_squared_error', verbose=3)

    grid.fit(train[comm_cols], target)

    best_forest = grid.best_estimator_

    print(grid.best_params_)

    print(round( (-grid.best_score_ )**0.5 ,3))
    '''
    best_forest = RandomForestRegressor(n_estimators=1000, n_jobs=-1, random_state=32, max_depth=20, max_features='sqrt')

    for fold_, (trn_idx, val_idx) in enumerate(kfolds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))
        
        trn_data = train.iloc[trn_idx][comm_cols]
        val_data = train.iloc[val_idx][comm_cols]
        
        trn_target = target.iloc[trn_idx]
        val_target = target.iloc[val_idx]
    
        clf = best_forest.fit(trn_data, trn_target)
        
        oof[val_idx] = clf.predict(train.iloc[val_idx][comm_cols])
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = comm_cols
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        predictions += clf.predict(test[comm_cols]) / kfolds.n_splits

    print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))

    return predictions, mean_squared_error(oof, target)**0.5, feature_importance_df, oof


def extratrees_train(train, test, target, kfolds):
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    comm_cols = list(set(train.columns).intersection(test.columns))

    grid_param = {'max_depth': np.arange(3,30),
                    'min_samples_split': np.arange(2, 50), 
                    'min_samples_leaf': np.arange(1,40), 
                    'max_features': ['sqrt', 'log2', None]}

    print('Optimizing parameters')

    grid = RandomizedSearchCV(ExtraTreesRegressor(n_estimators=500, n_jobs=4, random_state=345),
                                param_distributions=grid_param, n_iter=20, cv=kfolds, 
                                random_state=654, n_jobs=-1, scoring='neg_mean_squared_error', verbose=3)

    grid.fit(train[comm_cols], target)

    best_forest = grid.best_estimator_

    print(grid.best_params_)

    print(round( (-grid.best_score_ )**0.5 ,3))

    for fold_, (trn_idx, val_idx) in enumerate(kfolds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))
        
        trn_data = train.iloc[trn_idx][comm_cols]
        val_data = train.iloc[val_idx][comm_cols]
        
        trn_target = target.iloc[trn_idx]
        val_target = target.iloc[val_idx]
    
        clf = best_forest.fit(trn_data, trn_target)
        
        oof[val_idx] = clf.predict(train.iloc[val_idx][comm_cols])
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = comm_cols
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        predictions += clf.predict(test[comm_cols]) / kfolds.n_splits

    print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))

    return predictions, mean_squared_error(oof, target)**0.5, feature_importance_df, oof



