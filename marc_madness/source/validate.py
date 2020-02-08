__author__ = 'lucabasa'
__version__ = '1.4.1'
__status__ = 'development'


import pandas as pd
import numpy as np

import source.utility as ut


def random_split(data, model, kfolds, target, test_size=0.2, predict_proba=False):
    
    train, test = ut.make_test(data, test_size=test_size, random_state=324)
    
    y_train = train[target]
    y_test = test[target]
    
    for col in ['target', 'target_points', 'ID', 'DayNum', 'Team1', 'Team2', 'Season']:
        try:
            del train[col]
            del test[col]
        except KeyError:
            pass
    
    oof, imp_coef = ut.cv_score(train, y_train, kfolds, model, imp_coef=True, predict_proba=predict_proba)
    
    fit_model = model.fit(train, y_train)    
    
    return fit_model, oof, imp_coef, train, y_train, test, y_test


def yearly_split(data, model, kfolds, target, predict_proba=False):
    
    fit_model = {}
    oof = {}
    imp_coef = {}
    train = {}
    test = {}
    y_train = {}
    y_test = {}    
    
    for year in data.Season.unique():
        yr = str(year)
        train[yr] = data[data.Season != year].copy()
        test[yr] = data[data.Season == year].copy()
    
        y_train[yr] = train[yr][target]
        y_test[yr] = test[yr][target]

        for col in ['target', 'target_points', 'ID', 'DayNum', 'Team1', 'Team2', 'Season']:
            try:
                del train[yr][col]
                del test[yr][col]
            except KeyError:
                pass
    
        oof[yr], imp_coef[yr] = ut.cv_score(train[yr], y_train[yr], kfolds, model, imp_coef=True, predict_proba=predict_proba)
    
        fit_model[yr] = model.fit(train[yr], y_train[yr])    
    
    return fit_model, oof, imp_coef, train, y_train, test, y_test

