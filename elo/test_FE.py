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
import feature_eng as fe
import model_comparison as mc 




def read_data(input_file):
    df = pd.read_csv(input_file)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days
    del df['first_active_month']
    return df


agg_loc = 'processed_data/'

'''
switcher = {'remove_2m': fe.remove_2m, 
    'spread': fe.make_spread,
    'comb_cats': fe.combine_categs,
    'trend': fe.make_trend,
    'fractions': fe.make_fractions,
    'differences': fe.make_differences,
    'trends_no2m': fe.trends_no2m,
    'spread_no2m': fe.spread_no2m}
'''

switcher = {'remove_max_min': fe.remove_max_min}#, 
            #'make_train_feats': fe.make_train_feats,
            #'clv': fe.clv}


def main():
    train = read_data('raw_data/train.csv')
    test = read_data('raw_data/test.csv')

    df_tr = pd.read_csv(agg_loc + 'total_aggregation_with_FE_0219.csv')
    train = pd.merge(train, df_tr, on='card_id', how='left').fillna(0)
    test = pd.merge(test, df_tr, on='card_id', how='left').fillna(0)

    del df_tr

    train = fe.combine_categs(train)
    test = fe.combine_categs(test)
    train = fe.clv(train)
    test = fe.clv(test)

    results = {}

    for key in switcher.keys():
        print(key)
        func = switcher.get(key)

        df_train = func(train)
        df_test = func(test)

        target = df_train['target']
        id_to_sub = df_test['card_id']
        del df_train['target']
        del df_train['card_id']
        del df_test['card_id']

        kfolds = KFold(5, shuffle=True, random_state=42)

        predictions, cv_score, feat_imp = mc.lightgbm_train(df_train, df_test, target, kfolds)

        if cv_score < 3.67855:
            results[key] = cv_score

        sub_df = pd.DataFrame({"card_id":id_to_sub.values})
        sub_df["target"] = predictions
        sub_df.to_csv("results/0219_selected_fe.csv", index=False)
        feat_imp.to_csv("results/0219_selected_fe_featimp.csv", index=False)

        print(f'{key}:\t {cv_score}')
        print('_'*40)
        print('_'*40)
        print('\n')
        

    final = pd.DataFrame.from_dict(results, orient='index', columns=['CV_score'])
    #final.to_csv('results/0219_fe_selection.csv')


if __name__=="__main__":
    main()


