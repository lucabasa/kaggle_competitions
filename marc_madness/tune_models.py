__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'


import numpy as np
import pandas as pd 

from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import KFold, RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def tune_rf(train, target, verbose=False):
    grid_param = {'max_depth': np.arange(3,30),
                'min_samples_split': np.arange(2, 50), 
                'min_samples_leaf': np.arange(1,40), 
                'max_features': ['sqrt', 'log2', None]}

    kfolds = KFold(5, shuffle=True, random_state=14)

    grid = RandomizedSearchCV(RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=345),
                            param_distributions=grid_param, n_iter=20, cv=kfolds, 
                            random_state=654, n_jobs=-1, scoring='neg_log_loss')

    grid.fit(train, target)

    best_forest = grid.best_estimator_

    if verbose:
        print(grid.best_params_)
        print(round( (-grid.best_score_ ) ,3))

    return grid.best_params_


def tune_extra(train, target, verbose=False):
    grid_param = {'max_depth': np.arange(3,30),
                'min_samples_split': np.arange(2, 50), 
                'min_samples_leaf': np.arange(1,40), 
                'max_features': ['sqrt', 'log2', None]}

    kfolds = KFold(5, shuffle=True, random_state=14)

    grid = RandomizedSearchCV(ExtraTreesClassifier(n_estimators=500, n_jobs=4, random_state=345),
                            param_distributions=grid_param, n_iter=50, cv=kfolds, 
                            random_state=654, n_jobs=-1, scoring='neg_log_loss')

    grid.fit(train, target)

    best_forest = grid.best_estimator_

    if verbose:
        print(grid.best_params_)
        print(round( (-grid.best_score_ ) ,3))

    return grid.best_params_
