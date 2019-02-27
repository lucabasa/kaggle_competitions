__author__ = 'lucabasa'
__version__ = '1.0'
__status__ = 'development'

import numpy as np 
import pandas as pd 

import datetime

from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import KFold


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
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem,
    																			100 * (start_mem - end_mem) / start_mem))
    return df


def find_missing(data_list):
    """
    data_list: list of pandas dataframes
    """
    for data in data_list:
        print("In {}\n".format(data.name))
        cols = data.columns
        for col in cols:
            mis = data[col].isnull().sum()
            if mis > 0:
                print("{}: {} missing, {}%".format(col, mis, 
                									round(mis/data.shape[0] * 100, 3)))
        print("_"*40)
        print("_"*40)


def explore_all(data):
    cols = [col for col in data.columns if col != 'card_id']

    num = len(cols)
    rows = int(num/2) + (num % 2 > 0)
    
    fig, ax = plt.subplots(rows, 2, figsize=(15, 5 * (rows)))
    i = 0
    j = 0
    for feat in cols:
        data[feat].hist(bins=30, label=feat, ax=ax[i][j])
        ax[i][j].set_title(feat, fontsize=12)
        ax[i][j].grid(False)
        j = (j+1)%2
        i = i + 1 - j


def read_data(input_file):
    df = pd.read_csv(input_file)
    df['first_active_month'] = pd.to_datetime(df['first_active_month'])
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    del df['first_active_month']
    return df