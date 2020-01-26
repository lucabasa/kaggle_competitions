__author__ = 'lucabasa'
__version__ = '1.1.0'
__status__ = 'development'


import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import KFold, RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb



def rf_train(train, test, target, folds, full_test, tune=False):

    comm_cols = list(set(train.columns).intersection(test.columns))

    if tune:
        params = tune
        forest = RandomForestClassifier(n_estimators=700, n_jobs=4, random_state=189,
                                        max_depth=params['max_depth'], 
                                        min_samples_split=params['min_samples_split'],
                                        min_samples_leaf=params['min_samples_leaf'],
                                        max_features=params['max_features'])
    else:
        forest = RandomForestClassifier(n_estimators=700, n_jobs=4, random_state=189)

    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    full_predictions = np.zeros(len(full_test))
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))
        
        trn_data = train.iloc[trn_idx][comm_cols]
        val_data = train.iloc[val_idx][comm_cols]
        
        trn_target = target.iloc[trn_idx]
        val_target = target.iloc[val_idx]
    
        clf = forest.fit(trn_data, trn_target)
        
        oof[val_idx] = clf.predict_proba(train.iloc[val_idx][comm_cols])[:,1]
        predictions += clf.predict_proba(test[comm_cols])[:,1] / folds.n_splits
        full_predictions += clf.predict_proba(full_test[comm_cols])[:,1] / folds.n_splits
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = comm_cols
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    cv_score = log_loss(target, oof)
    full_score = log_loss(full_test['target'], full_predictions)

    return oof, predictions, feature_importance_df, cv_score, full_score


def extra_train(train, test, target, folds, full_test, tune=False):

    comm_cols = list(set(train.columns).intersection(test.columns))

    if tune:
        params = tune
        forest = ExtraTreesClassifier(n_estimators=1500, n_jobs=4, random_state=189,
                                        max_depth=params['max_depth'], 
                                        min_samples_split=params['min_samples_split'],
                                        min_samples_leaf=params['min_samples_leaf'],
                                        max_features=params['max_features'])
    else:
        forest = ExtraTreesClassifier(n_estimators=1500, n_jobs=4, random_state=189)

    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    full_predictions = np.zeros(len(full_test))
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))
        
        trn_data = train.iloc[trn_idx][comm_cols]
        val_data = train.iloc[val_idx][comm_cols]
        
        trn_target = target.iloc[trn_idx]
        val_target = target.iloc[val_idx]
    
        clf = forest.fit(trn_data, trn_target)
        
        oof[val_idx] = clf.predict_proba(train.iloc[val_idx][comm_cols])[:,1]
        predictions += clf.predict_proba(test[comm_cols])[:,1] / folds.n_splits
        full_predictions += clf.predict_proba(full_test[comm_cols])[:,1] / folds.n_splits
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = comm_cols
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    cv_score = log_loss(target, oof)
    full_score = log_loss(full_test['target'], full_predictions)

    return oof, predictions, feature_importance_df, cv_score, full_score


def lgb_train(train, test, target, folds, full_test, tune=False):

    comm_cols = list(set(train.columns).intersection(test.columns))

    param = {
            'num_leaves': 129,
            'min_data_in_leaf': 10, 
            'objective':'binary',
            'max_depth': 14,
            'learning_rate': 0.005,
            "boosting": "gbdt",
            "feature_fraction": 0.934,
            "bagging_freq": 1,
            "bagging_fraction":  0.9784,
            "bagging_seed": 541,
            "metric": 'binary_logloss',
            "lambda_l1": 1.1,
            "verbosity": -1,
            'random_seed': 41
        }

    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    full_predictions = np.zeros(len(full_test))
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))
        
        trn_data = lgb.Dataset(train.iloc[trn_idx],
                               label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train.iloc[val_idx],
                               label=target.iloc[val_idx])
    
        clf = lgb.train(param,
                        trn_data,
                        10000,
                        valid_sets = [trn_data, val_data],
                        verbose_eval=500,
                        early_stopping_rounds = 200)
        
        oof[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)
        predictions += clf.predict(test[comm_cols]) / folds.n_splits
        full_predictions += clf.predict(full_test[comm_cols]) / folds.n_splits
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = comm_cols
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    cv_score = log_loss(target, oof)
    full_score = log_loss(full_test['target'], full_predictions)

    return oof, predictions, feature_importance_df, cv_score, full_score


