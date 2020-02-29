__author__ = 'lucabasa'
__version__ = '1.3.0'
__status__ = 'development'


import pandas as pd
import numpy as np

from scipy.interpolate import UnivariateSpline

import source.utility as ut


def _clean_columns(train, test):
    for col in ['target', 'target_points', 'ID', 'DayNum', 'Team1', 'Team2', 'Season']:
        try:
            del train[col]
            del test[col]
        except KeyError:
            pass
    return train, test


def _make_preds(train, y_train, test, model, kfolds, predict_proba):
    oof, imp_coef = ut.cv_score(train, y_train, kfolds, model, imp_coef=True, predict_proba=predict_proba)
    
    fit_model = model.fit(train, y_train)
    if predict_proba:
        predictions = fit_model.predict_proba(test)[:,1]
    else:
        predictions = fit_model.predict(test)
    
    return fit_model, oof, imp_coef, predictions


def point_to_proba(oof, y_train, preds):
    dat = list(zip(np.clip(oof, -30, 30), np.where(y_train > 0, 1, 0)))
    dat = sorted(dat, key = lambda x: x[0])
    datdict = {dat[k][0]: dat[k][1] for k in range(len(dat))}

    spline_model = UnivariateSpline(list(datdict.keys()), list(datdict.values()))  
    spline_oof = spline_model(np.clip(oof, -30, 30))
    spline_test = spline_model(np.clip(preds, -30, 30))
    
    return np.clip(spline_oof, 0.03, 0.97), np.clip(spline_test, 0.03, 0.97)


def random_split(data, model, kfolds, target, test_size=0.2, predict_proba=False):
    
    train, test = ut.make_test(data, test_size=test_size, random_state=324)
    
    y_train = train[target]
    y_test = test[target]
    
    train, test = _clean_columns(train, test)
    
    fit_model, oof, imp_coef, predictions = _make_preds(train, y_train, test, model, kfolds, predict_proba)
    
    return fit_model, oof, predictions, imp_coef, train, y_train, test, y_test


def yearly_split(data, model, kfolds, target, predict_proba=False):
    
    fit_model = {}
    oof = {}
    imp_coef = {}
    train = {}
    test = {}
    y_train = {}
    y_test = {}
    predictions = {}
    
    years = [2015, 2016, 2017, 2018, 2019]
    
    for year in years:
        yr = str(year)
        train[yr] = data[data.Season != year].copy()
        test[yr] = data[data.Season == year].copy()
    
        y_train[yr] = train[yr][target]
        y_test[yr] = test[yr][target]

        train[yr], test[yr] = _clean_columns(train[yr], test[yr])
        
        fit_model[yr], oof[yr], imp_coef[yr], predictions[yr] = _make_preds(train[yr], 
                                                                            y_train[yr], 
                                                                            test[yr], 
                                                                            model, 
                                                                            kfolds, 
                                                                            predict_proba)
    
    return fit_model, oof, predictions, imp_coef, train, y_train, test, y_test
