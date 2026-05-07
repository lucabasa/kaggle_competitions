__author__ = "lucabasa"
__version__ = "2.0.0"
__status__ = "development"


import pandas as pd
import numpy as np

from scipy.interpolate import UnivariateSpline

from sklearn.model_selection import GridSearchCV

import tubesml as tml

from source.train_funcs import train_model


def _clean_columns(train, test):
    for col in ["target", "target_points", "ID", "DayNum", "Team1", "Team2",
                "T1_region", "T2_region",
                "Season", "competitive", "competitive_score",
                "delta_def_rating_diff", "delta_impact_diff",
                "T1_def_rating_diff", "T2_def_rating_diff"]:
        try:
            del train[col]
            del test[col]
        except KeyError:
            pass
    return train, test


def _make_preds(train, y_train, test, model, kfolds, predict_proba, fit_params=None, early_stopping=False, regression=True):

    cv_score = tml.CrossValidate(data=train, target=y_train, test=test,
                                 estimator=model, cv=kfolds, fit_params=fit_params, early_stopping=early_stopping,
                                 imp_coef=True, predict_proba=predict_proba, regression=regression)
    
    oof, pred, result_dict = cv_score.score()
    
    return  oof, result_dict, pred


def random_split(data, model, kfolds, target, test_size=0.2, boost=False, predict_proba=False, tune=False, param_grid=None, **kwargs):
    
    train, test = tml.make_test(data, test_size=test_size, strat_feat="Season", random_state=324)
    
    y_train = train[target]
    y_test = test[target]
    
    train, test = _clean_columns(train, test)
    
    if tune and not boost:
        if predict_proba:
            grid = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, 
                                cv=5, scoring="neg_log_loss")
        else:
            grid = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, 
                                cv=5, scoring="neg_mean_absolute_error")
        grid.fit(train, y_train)
        model = grid.best_estimator_
        print(grid.best_score_)
        print(grid.best_params_)
    
    if boost:
        oof, result_dict, preds = train_model(train, test, y_train, cv=kfolds, predict_proba=predict_proba, **kwargs)
        return oof, preds, result_dict, train, y_train, test, y_test
    else:    
        oof, result_dict, pred = _make_preds(train, y_train, test, model, kfolds, predict_proba, **kwargs)

        return oof, pred, result_dict, train, y_train, test, y_test


def yearly_split(data, model, kfolds, target, boost=False, predict_proba=False, tune=False, param_grid=None, **kwargs):

    oof = {}
    train = {}
    test = {}
    y_train = {}
    y_test = {}
    predictions = {}
    result_dicts = {}
    
    years = [2022, 2023, 2024, 2025]
    
    for year in years:
        yr = str(year)
        train[yr] = data[data.Season != year].copy()
        test[yr] = data[data.Season == year].copy()
    
        y_train[yr] = train[yr][target]
        y_test[yr] = test[yr][target]

        train[yr], test[yr] = _clean_columns(train[yr], test[yr])
        
        if tune and not boost:
            if predict_proba:
                grid = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, 
                                    cv=5, scoring="neg_log_loss")
            else:
                grid = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, 
                                    cv=5, scoring="neg_mean_absolute_error")
            grid.fit(train[yr], y_train[yr])
            model = grid.best_estimator_
            print(grid.best_score_)
            print(grid.best_params_)
            
        if boost:
            oof[yr], result_dicts[yr], predictions[yr] = train_model(train[yr], test[yr], y_train[yr],
                                                                 cv=kfolds, predict_proba=predict_proba, **kwargs)
        else:
            oof[yr], result_dicts[yr], predictions[yr] = _make_preds(train[yr],
                                                                     y_train[yr],
                                                                     test[yr],
                                                                     model,
                                                                     kfolds,
                                                                     predict_proba, **kwargs)
    
    return oof, predictions, result_dicts, train, y_train, test, y_test

