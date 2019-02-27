__author__ = 'lucabasa'
__version__ = '1.0'
__status__ = 'development'

import numpy as np 
import pandas as pd 



def sel_all(train, test):
    return train, test
    

def sel_nomoney(train, test):
    comm_cols = list(set(train.columns).intersection(test.columns))

    to_use = [col for col in train.columns if 'purchase_amount' not in col]

    return train[to_use], test[to_use]


def sel_money(train, test):
    comm_cols = list(set(train.columns).intersection(test.columns))

    to_use = [col for col in train.columns if 'purchase_amount' in col]

    return train[to_use], test[to_use]


def sel_counts(train, test):
    comm_cols = list(set(train.columns).intersection(test.columns))

    to_use = [col for col in train.columns if 'nunique' in col]

    return train[to_use], test[to_use]


def sel_new(train, test):
    comm_cols = list(set(train.columns).intersection(test.columns))

    to_use = [col for col in train.columns if 'new' in col]

    return train[to_use], test[to_use]


def sel_hist(train, test):
    comm_cols = list(set(train.columns).intersection(test.columns))

    to_use = [col for col in train.columns if 'tot' in col]

    return train[to_use], test[to_use]


