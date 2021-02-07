__author__ = 'lucabasa'
__version__ = '1.3.1'
__status__ = 'development'


import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit, train_test_split



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
        train_set, test_set = train_test_split(train, test_size=test_size, random_state=random_state, shuffle=True)
            
    return train_set, test_set

