__author__ = 'lucabasa'
__version__ = '2.0.0'
__status__ = 'development'


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, max_error
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from scipy.interpolate import UnivariateSpline

from datetime import date
from os.path import exists


def _plot_diagonal(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    low = min(xmin, xmax)
    high = max(xmin, xmax)
    scl = (high - low) / 100
    
    line = pd.DataFrame({'x': np.arange(low, high ,scl), # small hack for a diagonal line
                         'y': np.arange(low, high ,scl)})
    ax.plot(line.x, line.y, color='black', linestyle='--')
    
    return ax


def plot_predictions(data, true_label, pred_label, feature=None, hue=None, legend=False, savename='test.png'):
    '''
    Plot prediction vs true label or a specific feature. It also plots the residuals plot
    '''
    
    tmp = data.copy()
    tmp['Prediction'] = pred_label
    tmp['True Label'] = true_label
    tmp['Residual'] = tmp['True Label'] - tmp['Prediction']
    
    diag = False
    alpha = 0.7
    label = ''
    
    fig, ax = plt.subplots(1,2, figsize=(15,6))
    
    if feature is None:
        feature = 'True Label'
        diag = True
    else:
        legend = 'full'
        sns.scatterplot(x=feature, y='True Label', data=tmp, ax=ax[0], label='True',
                         hue=hue, legend=legend, alpha=alpha)
        label = 'Predicted'
        alpha = 0.4

    sns.scatterplot(x=feature, y='Prediction', data=tmp, ax=ax[0], label=label,
                         hue=hue, legend=legend, alpha=alpha)
    if diag:
        ax[0] = _plot_diagonal(ax[0])
    
    sns.scatterplot(x=feature, y='Residual', data=tmp, ax=ax[1], 
                    hue=hue, legend=legend, alpha=0.7)
    ax[1].axhline(y=0, color='r', linestyle='--')
    
    ax[0].set_title(f'{feature} vs Predictions')
    ax[1].set_title(f'{feature} vs Residuals')
    
    if savename:
        plt.savefig('plots/' + savename)
    plt.show()


def high_low_errors(data, *, res_list=None, n_samples=50,
                    target=None, pred_list=None, mean=False, 
                    abs_err=True, common=False):
    '''
    Report on the difference between high and low errors by doing a difference 
    between the results of a pandas describe method

    If the residuals are not provided, they can be calculated with the use of 
    both target and pred_list

    It is possible to compute the mean across the various models and to focus
    on the absoulte errors

    todo: implement the procedure that takes the common top n_samples errors

    '''
    
    df = data.copy()
    if pred_list:
        res_list = []
        for col in pred_list:
            name = col + '_res'
            res_list.append(name)
            df[name] = df[target] - df[col]
    
    errors = {}
    
    if mean:
        df['mean_res'] = df[res_list].mean(axis=1)
        res_list += ['mean_res']

    for col in res_list:
        if abs_err:
            if col == 'abs_err':
                name = 'abs_err'
            else:
                name = 'abs_' + col
            df[name] = abs(df[col])
        else:
            name = col
        
        high_err = df.sort_values(name, ascending=False).head(n_samples)
        low_err = df.sort_values(name, ascending=False).tail(n_samples)
        
        try:
            errors[name] = high_err.describe(include='all').drop(index=['top', 'count', 'freq']).fillna(0) - \
                        low_err.describe(include='all').drop(index=['top', 'count', 'freq']).fillna(0)
        except KeyError:
            errors[name] = high_err.describe().fillna(0) - low_err.describe().fillna(0)
        
    return errors


def make_results(label, prediction, model, parameters, target_name, variables, instances, verbose=False):
    results=pd.DataFrame({'Date': [date.today().strftime("%d/%m/%Y")], 
                          'Model': [model],
                          'Parameters': [parameters], 
                          'Target': target_name, 
                          'Variables': variables, 
                          'N_instances': instances})
    
    results['MAE'] = mean_absolute_error(y_true=label, y_pred=prediction)
    results['MSE'] = mean_squared_error(y_true=label, y_pred=prediction)
    results['Max_error'] = max_error(y_true=label, y_pred=prediction)
    results['Explained_var'] = explained_variance_score(y_true=label, y_pred=prediction)

    if verbose:
        print(f'MAE: \t\t {round(results["MAE"].values[0], 5)}')
        print(f'MSE: \t\t {round(results["MSE"].values[0], 5)}')
        print(f'Max Error: \t {round(results["Max_error"].values[0], 5)}')
        print(f'Expl Variance: \t {round(results["Explained_var"].values[0], 5)}')
    
    return results


def store_results(file_loc, label, prediction, model, parameters, target_name, variables, instances, verbose=False):
    
    results = make_results(label, prediction, model, parameters, target_name, variables, instances, verbose)
    
    if not exists(file_loc):
        results.to_csv(file_loc, index=False)
    else:
        old_results = pd.read_csv(file_loc)
        results = pd.concat([old_results, results])
        results.to_csv(file_loc, index=False) 
    return


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
    auc_oof = round(roc_auc_score(y_true=(y_train>0).astype(int), y_score=(oof>0).astype(int)),4)
    auc_test = round(roc_auc_score(y_true=(y_test>0).astype(int), y_score=(preds>0).astype(int)),4)
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
        plot_predictions(train, y_train, oof, savename=False)
        plot_predictions(test, y_test, preds, savename=False)
        
        plot_pred_prob(spline_oof, spline_test, y_train, y_test)
    
    print(f'MAE train: \t\t\t {mae_oof}')
    print(f'MAE test: \t\t\t {mae_test}')
    print(f'RMSE train: \t\t\t {mse_oof}')
    print(f'RMSE test: \t\t\t {mse_test}')
    print(f'Accuracy train: \t\t {acc_oof}')
    print(f'Accuracy test: \t\t\t {acc_test}')
    print(f'AUC ROC train: \t\t\t {auc_oof}')
    print(f'AUC ROC test: \t\t\t {auc_test}')
    print(f'Logloss train: \t\t\t {logloss_oof}')
    print(f'Logloss test: \t\t\t {logloss_test}')
    print(f'Unsure train: \t\t\t {n_unsure_oof}%')
    print(f'Unsure test: \t\t\t {n_unsure_test}%')

    
def report_victory(y_train, y_test, oof, preds, probs=True, plot=True):
    
    if probs:
        oof = np.clip(oof, 0.03, 0.97)
        preds = np.clip(preds, 0.03, 0.97)
        acc_oof = round(accuracy_score(y_true=y_train, y_pred=(oof>0.5).astype(int)),4)
        acc_test = round(accuracy_score(y_true=y_test, y_pred=(preds>0.5).astype(int)),4)
        auc_oof = round(roc_auc_score(y_true=y_train, y_score=(oof>0.5).astype(int)),4)
        auc_test = round(roc_auc_score(y_true=y_test, y_score=(preds>0.5).astype(int)),4)
        n_unsure_oof = round((abs(oof - 0.5) < 0.1).mean() * 100, 4)
        n_unsure_test = round((abs(preds - 0.5) < 0.1).mean() * 100, 4)
        logloss_oof = round(log_loss(y_true=y_train, y_pred=oof), 4)
        logloss_test = round(log_loss(y_true=y_test, y_pred=preds), 4)
        
        if plot:
            plot_pred_prob(oof, preds, y_train, y_test)
    
    print(f'Accuracy train: \t\t {acc_oof}')
    print(f'Accuracy test: \t\t\t {acc_test}')
    print(f'AUC ROC train: \t\t\t {auc_oof}')
    print(f'AUC ROC test: \t\t\t {auc_test}')
    print(f'Logloss train: \t\t\t {logloss_oof}')
    print(f'Logloss test: \t\t\t {logloss_test}')
    print(f'Unsure train: \t\t\t {n_unsure_oof}%')
    print(f'Unsure test: \t\t\t {n_unsure_test}%')    
    

def yearly_wrapper(train, test, y_train, y_test, oof, preds, min_yr=2015, points=True):
    y_train_total = []
    y_test_total = []
    oof_total = []
    preds_total = []
    full_train = []
    full_test = []
    for yr in train.keys():
        print(yr)
        print('\n')
        if points:
            report_points(train[yr], test[yr], y_train[yr], y_test[yr], oof[yr], preds[yr], plot=False)
        else:
            report_victory(y_train[yr], y_test[yr], oof[yr], preds[yr], probs=True, plot=False)
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
        
    print('Total predictions')
    print('\n')
    y_train_total = pd.concat(y_train_total, ignore_index=True)
    y_test_total = pd.concat(y_test_total, ignore_index=True)
    full_train = pd.concat(full_train, ignore_index=True)
    full_test = pd.concat(full_test, ignore_index=True)
    oof_total = pd.Series(oof_total)
    preds_total = pd.Series(preds_total)
    if points:
        report_points(full_train, full_test, y_train_total, y_test_total, oof_total, preds_total, plot=True)
    else:
        report_victory(y_train_total, y_test_total, oof_total, preds_total, probs=True, plot=True)
    
