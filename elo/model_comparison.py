__author__ = 'lucabasa'
__version__ = 1.0
__status__ = "development"


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import datetime

import gc

from sklearn.metrics import mean_squared_error, log_loss
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

from utilities import reduce_mem_usage


def read_data(input_file):
    df = pd.read_csv(input_file)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    del df['first_active_month']
    return df


agg_loc = 'processed_data/'
aggs = {'new_tr': 'new_agg.csv',
    'full_hist': 'full_history_agg.csv', 
    'full_hist_auth': 'full_history_agg_auth.csv',
    'hist_w_2m': 'hist_agg_by2months.csv', 
    'total_agg': 'total_aggregation.csv',
    'total_agg_auth': 'total_aggregation_auth.csv',
    'FE_fractions': 'total_aggregation_with_FE.csv',
    'FE_frac_diff': 'total_aggregation_with_FE_0219.csv'}
'''
aggs = {'2_months': 'agg_h2_transactions.csv', 
    '2_months_auth': 'agg_h2_transactions_auth.csv',
    '4_months' : 'agg_h4_transactions.csv',
    '4_months_auth' : 'agg_h4_transactions_auth.csv',
    '6_months' : 'agg_h6_transactions.csv',
    '6_months_auth' : 'agg_h6_transactions_auth.csv',
    '8_months' : 'agg_h8_transactions.csv',
    '8_months_auth' : 'agg_h8_transactions_auth.csv',
    '10_months' : 'agg_h10_transactions.csv',
    '10_months_auth' : 'agg_h10_transactions_auth.csv',
    '12_months' : 'agg_h12_transactions.csv',
    '12_months_auth' : 'agg_h12_transactions_auth.csv',}
    '''


param = {'num_leaves': 130,#111,
         'min_data_in_leaf': 150,#149, 
         'objective': 'regression', # 'binary',
         'max_depth': 11,#9,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.7401546954675876,#0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7986867839650837,#0.7083 ,
         "bagging_seed": 11,
         "metric": 'rmse', # 'binary_logloss', 
         "lambda_l1": 5.118671456974949,#0.2634,
         "random_seed": 133,
         "verbosity": -1}

param_class = {'num_leaves': 111,
         'min_data_in_leaf': 149, 
         'objective': 'binary', 
         'max_depth': 9,
         'learning_rate': 0.005,
         "boosting": "gbdt",
         "feature_fraction": 0.7522,
         "bagging_freq": 1,
         "bagging_fraction": 0.7083 ,
         "bagging_seed": 11,
         "metric": 'binary_logloss',  
         "lambda_l1": 0.2634,
         "random_seed": 133,
         "verbosity": -1}



def lightgbm_train(train, test, target, kfolds):
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
                        verbose_eval=400,
                        early_stopping_rounds = 200)
        
        oof[val_idx] = clf.predict(train.iloc[val_idx][comm_cols], num_iteration=clf.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = comm_cols
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        predictions += clf.predict(test[comm_cols], num_iteration=clf.best_iteration) / kfolds.n_splits

    print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))

    return predictions, mean_squared_error(oof, target)**0.5, feature_importance_df


def lightgbm_classify(train, test, target, kfolds):
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
                        verbose_eval=400,
                        early_stopping_rounds = 200)
        
        oof[val_idx] = clf.predict(train.iloc[val_idx][comm_cols], num_iteration=clf.best_iteration)
        
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = comm_cols
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
        predictions += clf.predict(test[comm_cols], num_iteration=clf.best_iteration) / kfolds.n_splits

    print("CV score: {:<8.5f}".format(log_loss(target, oof)))

    return predictions, log_loss(target, oof), feature_importance_df


def main():
    train = read_data('raw_data/train.csv')
    test = read_data('raw_data/test.csv')

    results = {}

    for key in aggs.keys():
        print(key)
        df_tr = pd.read_csv(agg_loc + aggs[key])
        df_train = pd.merge(train, df_tr, on='card_id', how='left').fillna(0)
        df_test = pd.merge(test, df_tr, on='card_id', how='left').fillna(0)

        df_train['outlier'] = 0
        df_train.loc[df_train.target < -30, 'outlier'] = 1

        #df_train = df_train[df_train.outlier == 0].copy()

        target = df_train['target']
        #target = df_train['outlier']
        id_to_sub = df_test['card_id']
        del df_train['target']
        del df_train['card_id']
        del df_test['card_id']

        for split in ['norm']: # , 'strat'
            print(split)
            if split == 'norm':
                kfolds = KFold(5, shuffle=True, random_state=42)
            elif split == 'strat':
                kfolds = StratifiedKFold(5, shuffle=True, random_state=42)

            predictions, cv_score, feat_imp = lightgbm_train(df_train, df_test, target, kfolds)
            #predictions, cv_score, feat_imp = lightgbm_classify(df_train, df_test, target, kfolds)

            sub_df = pd.DataFrame({"card_id":id_to_sub.values})
            sub_df["target"] = predictions
            sub_df.to_csv("results/0219_"+ key + "_" + split + "_all.csv", index=False)
            feat_imp.to_csv("results/0219_" + key + "_" + split + "_featimp_all.csv", index=False)

            results[key+'_'+split] = cv_score
            print('_'*40)
            print('_'*40)

        print('\n')
        print('_'*40)
        print('_'*40)
        print('\n')

    final = pd.DataFrame.from_dict(results, orient='index', columns=['CV_score'])
    final.to_csv('results/0219_cvscores.csv')


if __name__=="__main__":
    main()
