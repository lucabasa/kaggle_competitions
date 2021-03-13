__author__ = 'lucabasa'
__version__ = '3.0.0'
__status__ = 'development'


import pandas as pd
import numpy as np

import tubesml as tml

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, max_error
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from scipy.interpolate import UnivariateSpline

from datetime import date
from os.path import exists



def _plot_proba(score, label, spline, ax):
    plot_df = pd.DataFrame({"pred": score, 
                            "label": np.where(label > 0, 1, 0), 
                            "spline": spline})
    plot_df["pred_int"] = (plot_df["pred"]).astype(int)
    plot_df = plot_df.groupby('pred_int', as_index=False)[['spline','label']].mean()
    
    ax.plot(plot_df.pred_int,plot_df.spline, label='Spline')
    ax.plot(plot_df.pred_int,plot_df.label, label='Score')
    ax.axhline(0.5, color='r', linestyle='--', alpha=0.5)
    ax.legend()
    ax.set_xlabel('Predicted score')
    ax.set_ylabel('Predicted probability')
    
    return ax


def _point_to_proba(oof, y_train, preds):
    dat = list(zip(np.clip(oof, -30, 30), np.where(y_train > 0, 1, 0)))
    dat = sorted(dat, key = lambda x: x[0])
    datdict = {dat[k][0]: dat[k][1] for k in range(len(dat))}

    spline_model = UnivariateSpline(list(datdict.keys()), list(datdict.values()))  
    spline_oof = spline_model(np.clip(oof, -30, 30))
    spline_test = spline_model(np.clip(preds, -30, 30))
    
    return np.clip(spline_oof, 0.03, 0.97), np.clip(spline_test, 0.03, 0.97)


def plot_pred_prob(oof, test, y_train, y_test):
    
    fig, ax = plt.subplots(1,2, figsize=(15, 6))
    
    df = pd.DataFrame()
    df['true'] = np.where(y_train > 0, 1, 0)
    df['Prediction'] = oof
    
    df[df.true==1]['Prediction'].hist(bins=50, ax=ax[0], alpha=0.5, color='g', label='Victory')
    df[df.true==0]['Prediction'].hist(bins=50, ax=ax[0], alpha=0.5, color='r', label='Loss')
    
    df = pd.DataFrame()
    df['true'] = np.where(y_test > 0, 1, 0)
    df['Prediction'] = test

    df[df.true==1]['Prediction'].hist(bins=50, ax=ax[1], alpha=0.5, color='g', label='Victory')
    df[df.true==0]['Prediction'].hist(bins=50, ax=ax[1], alpha=0.5, color='r', label='Loss')
    
    ax[0].axvline(0.5, color='k', linestyle='--')
    ax[1].axvline(0.5, color='k', linestyle='--')
    
    ax[0].set_title('Training data')
    ax[1].set_title('Test data')
    ax[0].grid(False)
    ax[1].grid(False)
    ax[0].legend()
    ax[1].legend()
    fig.suptitle('Probabilities of victory', fontsize=15)


def report_points(train, test, y_train, y_test, oof, preds, plot=True):
    mae_oof = round(mean_absolute_error(y_true=y_train, y_pred=oof), 4)
    mae_test = round(mean_absolute_error(y_true=y_test, y_pred=preds), 4)
    mse_oof = round(np.sqrt(mean_squared_error(y_true=y_train, y_pred=oof)), 4)
    mse_test = round(np.sqrt(mean_squared_error(y_true=y_test, y_pred=preds)), 4)
    acc_oof = round(accuracy_score(y_true=(y_train>0).astype(int), y_pred=(oof>0).astype(int)),4)
    acc_test = round(accuracy_score(y_true=(y_test>0).astype(int), y_pred=(preds>0).astype(int)),4)
    n_unsure_oof = round((abs(oof) < 2).mean() * 100, 2)
    n_unsure_test = round((abs(preds) < 2).mean() * 100, 2)
    
    # transform into probabilities
    spline_oof, spline_test = _point_to_proba(oof, y_train, preds)
    
    logloss_oof = round(log_loss(y_true=np.where(y_train > 0, 1, 0), y_pred=spline_oof), 4)
    logloss_test = round(log_loss(y_true=np.where(y_test > 0, 1, 0), y_pred=spline_test), 4)
    
    if plot:
        # plot_proba
        fig, ax = plt.subplots(1,2, figsize=(15, 6))

        ax[0] = _plot_proba(oof, y_train, spline_oof, ax[0])
        ax[0].set_title('Training data')

        ax[1] = _plot_proba(preds, y_test, spline_test, ax[1])
        ax[1].set_title('Test data')

        fig.suptitle('Probabilities of victory via splines', fontsize=15)

        # plot predictions
        tml.plot_regression_predictions(train, y_train, oof, savename=None)
        tml.plot_regression_predictions(test, y_test, preds, savename=None)
        
        plot_pred_prob(spline_oof, spline_test, y_train, y_test)
    
    print(f'MAE train: \t\t\t {mae_oof}')
    print(f'MAE test: \t\t\t {mae_test}')
    print(f'RMSE train: \t\t\t {mse_oof}')
    print(f'RMSE test: \t\t\t {mse_test}')
    print(f'Accuracy train: \t\t {acc_oof}')
    print(f'Accuracy test: \t\t\t {acc_test}')
    print(f'Logloss train: \t\t\t {logloss_oof}')
    print(f'Logloss test: \t\t\t {logloss_test}')
    print(f'Unsure train: \t\t\t {n_unsure_oof}%')
    print(f'Unsure test: \t\t\t {n_unsure_test}%')
    
    return pd.DataFrame({'mae_oof': [mae_oof], 'mae_test': [mae_test], 
                         'mse_oof': [mse_oof], 'mse_test': [mse_test], 
                         'acc_oof': [acc_oof], 'acc_test': [acc_test], 
                         'logloss_oof': [logloss_oof], 'logloss_test': [logloss_test], 
                         'unsure_oof': [n_unsure_oof], 'unsure_test': [n_unsure_test]})

    
def report_victory(y_train, y_test, oof, preds, probs=True, plot=True):
    
    if probs:
        oof = np.clip(oof, 0.03, 0.97)
        preds = np.clip(preds, 0.03, 0.97)
        acc_oof = round(accuracy_score(y_true=y_train, y_pred=(oof>0.5).astype(int)),4)
        acc_test = round(accuracy_score(y_true=y_test, y_pred=(preds>0.5).astype(int)),4)
        n_unsure_oof = round((abs(oof - 0.5) < 0.1).mean() * 100, 4)
        n_unsure_test = round((abs(preds - 0.5) < 0.1).mean() * 100, 4)
        logloss_oof = round(log_loss(y_true=y_train, y_pred=oof), 4)
        logloss_test = round(log_loss(y_true=y_test, y_pred=preds), 4)
        
        if plot:
            plot_pred_prob(oof, preds, y_train, y_test)
    
    print(f'Accuracy train: \t\t {acc_oof}')
    print(f'Accuracy test: \t\t\t {acc_test}')
    print(f'Logloss train: \t\t\t {logloss_oof}')
    print(f'Logloss test: \t\t\t {logloss_test}')
    print(f'Unsure train: \t\t\t {n_unsure_oof}%')
    print(f'Unsure test: \t\t\t {n_unsure_test}%')
    
    return pd.DataFrame({'acc_oof': [acc_oof], 'acc_test': [acc_test], 
                         'logloss_oof': [logloss_oof], 'logloss_test': [logloss_test], 
                         'unsure_oof': [n_unsure_oof], 'unsure_test': [n_unsure_test]})
    

def yearly_wrapper(train, test, y_train, y_test, oof, preds, min_yr=2015, points=True):
    y_train_total = []
    y_test_total = []
    oof_total = []
    preds_total = []
    full_train = []
    full_test = []
    res_summary = []
    for yr in train.keys():
        print(yr)
        print('\n')
        if points:
            res = report_points(train[yr], test[yr], y_train[yr], y_test[yr], oof[yr], preds[yr], plot=False)
        else:
            res = report_victory(y_train[yr], y_test[yr], oof[yr], preds[yr], probs=True, plot=False)
        print('\n')
        print('_'*40)
        print('\n')
        if int(yr) >= min_yr:
            y_train_total.append(y_train[yr])
            y_test_total.append(y_test[yr])
            oof_total += list(oof[yr])
            preds_total += list(preds[yr])
            full_train.append(train[yr])
            full_test.append(test[yr])
            res['year'] = yr
            res_summary.append(res)
        
    print('Total predictions')
    print('\n')
    y_train_total = pd.concat(y_train_total, ignore_index=True)
    y_test_total = pd.concat(y_test_total, ignore_index=True)
    full_train = pd.concat(full_train, ignore_index=True)
    full_test = pd.concat(full_test, ignore_index=True)
    oof_total = pd.Series(oof_total)
    preds_total = pd.Series(preds_total)
    res_summary = pd.concat(res_summary, ignore_index=True)
    if points:
        _ = report_points(full_train, full_test, y_train_total, y_test_total, oof_total, preds_total, plot=True)
    else:
        _ = report_victory(y_train_total, y_test_total, oof_total, preds_total, probs=True, plot=True)
        
    return res_summary
    
