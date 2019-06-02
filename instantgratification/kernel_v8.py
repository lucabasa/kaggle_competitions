__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'obsolete'

import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

import lightgbm as lgb

pd.set_option('max_columns', 200)

import utility as ut



def train_lgb(df_train, df_test, kfolds):
    train = df_train.copy()
    test = df_test.copy()

    target = train.target.copy()

    sub = test[['id']].copy()

    train, test = ut.general_processing(train, test)

    # model
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()
    
    for fold_, (trn_idx, val_idx) in enumerate(kfolds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))
        
        trn_data = lgb.Dataset(train.iloc[trn_idx], 
                               label=target.iloc[trn_idx])
        val_data = lgb.Dataset(train.iloc[val_idx], 
                               label=target.iloc[val_idx])
        
        param = {
            'bagging_freq': 3,
            'bagging_fraction': 0.8,
            'boost_from_average':'false',
            'boost': 'gbdt',
            'feature_fraction': 0.9,
            'learning_rate': 0.01,
            'max_depth': 10,  
            'metric':'auc',
            'min_data_in_leaf': 82,
            'min_sum_hessian_in_leaf': 10.0,
            'num_leaves': 20,
            'objective': 'binary', 
            'verbosity': 1,    
        }

        # param = {  # this is for v9
        #     'bagging_freq': 3,
        #     'bagging_fraction': 0.8,
        #     'boost_from_average':'false',
        #     'boost': 'gbdt',
        #     'feature_fraction': 0.8,
        #     'learning_rate': 0.001,
        #     'max_depth': 10,  
        #     'metric':'auc',
        #     'min_data_in_leaf': 100,
        #     'num_leaves': 30,
        #     'objective': 'binary', 
        #     'verbosity': 1,
        #     'n_jobs': -1
            
        # }
        num_round = 1000000
        clf = lgb.train(param, trn_data, num_round, 
                        valid_sets = [trn_data, val_data],
                        verbose_eval=500, early_stopping_rounds = 100)
        
        oof[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = train.columns
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(test, num_iteration=clf.best_iteration) / kfolds.n_splits
        
    ut.report_oof(df_train, oof)

    sub['target'] = predictions
    
    return oof, sub


if __name__ == '__main__':
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    kfolds = KFold(n_splits=5, shuffle=True, random_state=15)

    oof, sub = train_lgb(df_train, df_test, kfolds)

    sub.to_csv('submissions/v8_lgb_sub.csv',index=False)
