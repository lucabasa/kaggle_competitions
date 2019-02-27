__author__ = 'lucabasa'
__version__ = 1.0
__status__ = "development"


import numpy as np 
import pandas as pd 

from scipy import stats

import datetime

import gc

import utilities

read_loc = 'raw_data/'
save_loc = 'processed_data/'


def process_transactions(file_name, authorized=False):
    df = pd.read_csv(read_loc + file_name, parse_dates=['purchase_date'])

    for col in ['authorized_flag', 'category_1']:
        df[col] = df[col].map({'Y':1, 'N':0})

    if authorized:
        df = df[df.authorized_flag == 1].copy()

    df['purchase_amount'] = np.round(df['purchase_amount'] / 0.00150265118 + 497.06,2)

    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']

    df['category_2'] = df['category_2'].fillna(1.0)
    df['category_3'] = df['category_3'].fillna('A')
    df['merchant_id'] = df['merchant_id'].fillna('M_ID_00a6ca8a8a')
    df['installments'] = df['installments'].replace(-1, np.nan)
    df['installments'] = df['installments'].replace(999, np.nan)

    df['month'] = df['purchase_date'].dt.month
    df['day'] = df['purchase_date'].dt.day
    df['hour'] = df['purchase_date'].dt.hour
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['weekday'] = df['purchase_date'].dt.weekday
    df['weekend'] = (df['purchase_date'].dt.weekday >=5).astype(int)

    #Christmas : December 25 2017
    df['christmas_day_2017']=(pd.to_datetime('2017-12-25')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Mothers Day: May 14 2017
    df['mothers_day_2017']=(pd.to_datetime('2017-06-04')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #fathers day: August 13 2017
    df['fathers_day_2017']=(pd.to_datetime('2017-08-13')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Childrens day: October 12 2017
    df['children_day_2017']=(pd.to_datetime('2017-10-12')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Valentine's Day : 12th June, 2017
    df['valentine_day_2017']=(pd.to_datetime('2017-06-12')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Black Friday : 24th November 2017
    df['black_friday_2017']=(pd.to_datetime('2017-11-24') - df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)
    #Mothers Day: May 13 2018
    df['mothers_day_2018']=(pd.to_datetime('2018-05-13')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    df = utilities.reduce_mem_usage(df)

    df['installm_binary'] = 0
    df.loc[df.installments > 0, 'installm_binary'] = 1

    df = pd.get_dummies(df, columns=['category_3'])

    df['cat_2_binary'] = 0
    df.loc[df.category_2 != 1, 'cat_2_binary'] = 1

    return df


def aggregate_transactions(data, name, authorized=False):

    agg_fun = {
        'category_1' : ['sum', 'mean'],
        'installments': ['mean', 'median'],
        'installm_binary': ['sum', 'mean'],
        'category_3_A': ['sum', 'mean'],
        'category_3_B': ['sum', 'mean'],
        'category_3_C': ['sum', 'mean'],
        'merchant_id': 'nunique',
        'city_id': ['nunique'],
        'merchant_category_id': ['nunique'],
        'purchase_amount': ['min', 'max', 'sum', 'mean', 'median'],
        'category_2': ['mean', 'median', 'max', 'min', 'sum'],
        'cat_2_binary': ['sum', 'mean'],
        'state_id': ['nunique'],
        'subsector_id': ['nunique']
        }

    if authorized:
        agg_fun['authorized_flag']= ['count']
    else:
        agg_fun['authorized_flag']= ['sum', 'mean', 'count']

    if 'tot' in name:
        to_add = ['christmas_day_2017', 'christmas_day_2017', 'children_day_2017', 
                  'valentine_day_2017', 'black_friday_2017', 'mothers_day_2018', ]
        for agg in to_add:
            agg_fun[agg] = ['mean', 'max']
        
        agg_fun['month'] = ['median', 'mean', 'nunique']
        agg_fun['day'] = ['nunique']
        agg_fun['weekend'] = ['mean']
        agg_fun['weekday'] = ['nunique']
        agg_fun['month_diff'] = ['max', 'min', 'mean']

    result = data.groupby('card_id', as_index=False).agg(agg_fun)
    
    result.columns = [name + '_' + '_'.join(col).strip() 
                           for col in result.columns.values]
    
    result = result.rename(columns={name + '_card_id_': 'card_id'})
    
    return result


def make_2m_features(df_hist):
    df_hist['lag_2m'] = 0
    df_hist.loc[(df_hist.month_lag > -3), 'lag_2m'] = 1

    df_hist['lag_4m'] = 0
    df_hist.loc[(df_hist.month_lag < -2) & (df_hist.month_lag > -5), 'lag_4m'] = 1

    df_hist['lag_6m'] = 0
    df_hist.loc[(df_hist.month_lag < -4) & (df_hist.month_lag > -7), 'lag_6m'] = 1

    df_hist['lag_8m'] = 0
    df_hist.loc[(df_hist.month_lag < -6) & (df_hist.month_lag > -9), 'lag_8m'] = 1

    df_hist['lag_10m'] = 0
    df_hist.loc[(df_hist.month_lag < -8) & (df_hist.month_lag > -11), 'lag_10m'] = 1

    df_hist['lag_12m'] = 0
    df_hist.loc[(df_hist.month_lag < -10), 'lag_12m'] = 1

    hist_2m = df_hist[df_hist.lag_2m == 1].copy()
    hist_4m = df_hist[df_hist.lag_4m == 1].copy()
    hist_6m = df_hist[df_hist.lag_6m == 1].copy()
    hist_8m = df_hist[df_hist.lag_8m == 1].copy()
    hist_10m = df_hist[df_hist.lag_10m == 1].copy()
    hist_12m = df_hist[df_hist.lag_12m == 1].copy()

    historical = [('h2', hist_2m), ('h4', hist_4m), 
              ('h6', hist_6m), ('h8', hist_8m),
              ('h10', hist_10m), ('h12', hist_12m)]

    return historical


if __name__=="__main__":
    authorized = True
    print('Start importing historical transactions...')
    hist = process_transactions('historical_transactions.csv', 
                                authorized=authorized)
    print('Start aggregating full history')
    agg_total = aggregate_transactions(hist, 'tot', authorized=authorized)
    if authorized:
        agg_total.to_csv(save_loc + 'full_history_agg_auth.csv', index=False)
    else:
        agg_total.to_csv(save_loc + 'full_history_agg.csv', index=False)



    print('Start making the 2 months aggregations')
    hist_list = make_2m_features(hist)

    total_aggregation = pd.DataFrame({'card_id': hist.card_id.unique()})

    del hist
    gc.collect()

    print('aggregating and merging...')
    for data in hist_list:
        print(data[0])
        df = data[1].copy()
        df = aggregate_transactions(df, data[0])
        if authorized:
            df.to_csv('processed_data/agg_'+data[0]+'_transactions_auth.csv', index=False)
        else:  
            df.to_csv('processed_data/agg_'+data[0]+'_transactions.csv', index=False)
        print('Merging with the previous...')
        total_aggregation = pd.merge(total_aggregation, df, on='card_id', how='left')
        print('\n')
        del df

    total_aggregation = pd.merge(total_aggregation, agg_total, on='card_id', how='left')

    del agg_total
    del hist_list
    gc.collect()

    print('Start importing new transactions...')
    new = process_transactions('new_merchant_transactions.csv')
    print('Aggregating it')
    agg_new = aggregate_transactions(new, 'new')
    agg_new.to_csv(save_loc + 'new_agg.csv', index=False)

    total_aggregation = pd.merge(total_aggregation, agg_new, on='card_id', how='left')

    if authorized:
        total_aggregation.to_csv(save_loc + 'total_aggregation_auth.csv', index=False)
    else:
         total_aggregation.to_csv(save_loc + 'total_aggregation.csv', index=False)   

