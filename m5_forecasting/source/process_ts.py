__author__ = 'lucabasa'
__version__ = '0.1.1'

import pandas as pd
import numpy as np


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                       df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def process_calendar(calendar):
    
    calendar = calendar.copy()
    
    calendar['date'] = pd.to_datetime(calendar['date'])
    calendar['d'] = calendar['d'].str.replace('d_', '').astype(int)
    del calendar['weekday']
    calendar['week'] = calendar['date'].dt.week
    calendar.loc[calendar.wday.isin([1,2]), 'week'] = calendar['week'] + 1 # align week with original definition
    fil_fix = (calendar.week >= 53) & (calendar.month == 1)
    calendar.loc[fil_fix, 'week'] = 1  # across years the week number can be inconsistent, this is the lesser evil
    
    for col in ['event_type_1', 'event_type_2', 'event_name_1', 'event_name_2']:
        calendar[col] = calendar[col].astype('category')
    
    calendar = reduce_mem_usage(calendar)
    
    return calendar


def melt_merge(sales, calendar, prices):
    
    calendar = process_calendar(calendar)
    
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    
    final = sales.copy()
    #del final['id']

    final.columns = id_cols + [col.replace('d_', '') for col in final if col not in id_cols]
    
    final = pd.melt(final, id_vars=id_cols, var_name='d', value_name='sales')
    final['d'] = final['d'].astype(int)
    for col in id_cols:
        final[col] = final[col].astype('category')
    
    final = pd.merge(final, calendar, on='d', how='left')
    
    final = pd.merge(final, prices, on=['wm_yr_wk', 'store_id', 'item_id'], how='left')
    del final['wm_yr_wk']
    
    final = reduce_mem_usage(final)
    
    return final


def lag_feats(data, lags):
    
    df = data.copy()
    df = df.sort_values('d', ascending=True)
    
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('id').sales.shift(lag)
    
    return df


def rolling_sum(data, window):
    
    sums = data.groupby('id')[['date', 'sales']].rolling(on='date', 
                                                         window=window, 
                                                         closed='left').sum().reset_index()
    del sums['level_1']
    sums.rename(columns={'sales': f'rol_{window}'}, inplace=True)
    sums = reduce_mem_usage(sums)
    
    return sums
    

def mean_period(data, col_name, agg_by=None, by_dimension=None, shift=1, agg='mean'):
    tmp = data.groupby(agg_by, as_index=False).sales.agg(agg)
    tmp[col_name] = tmp.groupby(by_dimension, as_index=False).sales.shift(shift)
    
    return tmp.drop('sales', axis=1)


def agg_id(data, col_name, id_col):
    ly = mean_period(data, col_name+'_last_year', [id_col, 'year'], id_col, shift=1)
    ly_m = mean_period(data, col_name+'_monht_last_year', [id_col, 'month', 'year'], [id_col, 'month'], shift=1)
    ly_w = mean_period(data, col_name+'_week_last_year', [id_col, 'week', 'year'], [id_col, 'week'], shift=1)
    lm = mean_period(data, col_name+'_last_month', [id_col, 'year', 'month'], id_col, shift=1)
    lw = mean_period(data, col_name+'_last_week', [id_col, 'year', 'week'], id_col, shift=1)
    
    return ly, ly_m, ly_w, lm, lw

