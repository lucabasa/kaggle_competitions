__author__ = 'lucabasa'
__version__ = '1.3.0'
__status__ = 'development'

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.tri as tri
from source.report import _plot_diagonal

import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve


def plot_hyperparameter(result, param_name, pretty_name, negative=True, save=False, uncertainty=True):
    
    if negative:
        res = result.copy()
        res['mean_train_score'] = -res['mean_train_score']
        res['mean_test_score'] = -res['mean_test_score']
    else:
        res = result.copy()

    fig, ax = plt.subplots(1,2, figsize=(15,6))
    
    try:
        X_axis = res[param_name].astype(float)
    except ValueError:
        X_axis = res[param_name]

    ax[0].plot(X_axis, res['mean_train_score'], label='Train', color='r', alpha=.6)
    ax[0].plot(X_axis, res['mean_test_score'], label='Test', color='g', alpha=.6)
    if uncertainty:
        ax[0].fill_between(X_axis, (res['mean_train_score'] - res['std_train_score']).astype(float),
                                (res['mean_train_score'] + res['std_train_score']).astype(float), alpha=0.1, color='r')
        ax[0].fill_between(X_axis, (res['mean_test_score'] - res['std_test_score']).astype(float),
                                (res['mean_test_score'] + res['std_test_score']).astype(float), alpha=0.1, color='g')

    ax[1].plot(X_axis, res['mean_fit_time'], label='Fit', color='r', alpha=.6)
    ax[1].fill_between(X_axis, (res['mean_fit_time'] - res['std_fit_time']).astype(float),
                            (res['mean_fit_time'] + res['std_fit_time']).astype(float), alpha=0.1, color='r')
    ax[1].plot(X_axis, res['mean_score_time'], label='Score', color='g', alpha=.6)
    ax[1].fill_between(X_axis, (res['mean_score_time'] - res['std_score_time']).astype(float),
                            (res['mean_score_time'] + res['std_score_time']).astype(float), alpha=0.1, color='g')

    ax[0].legend()
    ax[1].legend()
    ax[0].set_title('Score', fontsize=14)
    ax[1].set_title('Time', fontsize=14)
    fig.suptitle(f'{pretty_name}, Scores and Times', fontsize=18)
    
    if save:
        plt.savefig('plots/' + save)

    plt.show()


def plot_two_hyperparms(result, param_x, param_y, pretty_name, negative=True, save=False):
    
    if negative:
        res = result.copy()
        res['mean_test_score'] = -res['mean_test_score']
    else:
        res = result.copy()

    fig, ax = plt.subplots(1,2, figsize=(15,6))

    X_axis = res[param_x].astype(float)
    Y_axis = res[param_y].astype(float)

    xg, yg = np.meshgrid(np.linspace(X_axis.min(), X_axis.max(), 100),
                         np.linspace(Y_axis.min(), Y_axis.max(), 100))
    
    triangles = tri.Triangulation(X_axis, Y_axis)
    tri_interp = tri.CubicTriInterpolator(triangles, res['mean_test_score'])
    zg = tri_interp(xg, yg)
    
    ax[0].contourf(xg, yg, zg, 
                   norm=plt.Normalize(vmax=res['mean_test_score'].max(), vmin=res['mean_test_score'].min()),
                   cmap=plt.cm.terrain)
    
    tri_interp = tri.CubicTriInterpolator(triangles, res['mean_fit_time'])
    zg = tri_interp(xg, yg)
    
    ax[1].contourf(xg, yg, zg, 
                   norm=plt.Normalize(vmax=res['mean_fit_time'].max(), vmin=res['mean_fit_time'].min()), 
                   cmap=plt.cm.terrain)
    
    ax[0].set_xlabel(param_x.split('__')[-1].title(), fontsize=12)
    ax[1].set_xlabel(param_x.split('__')[-1].title(), fontsize=12)
    ax[0].set_ylabel(param_y.split('__')[-1].title(), fontsize=12)
    ax[1].set_ylabel(param_y.split('__')[-1].title(), fontsize=12)
    ax[0].set_title('Test Score', fontsize=14)
    ax[1].set_title('Fit Time', fontsize=14)
    fig.suptitle(f'{pretty_name}', fontsize=18)
    
    if save:
        plt.savefig('plots/' + save)
    
    plt.show()
    
    
def _label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))
    return ax


def plot_coefficients(target_name, est_coefs, coefs_real=None, annotate=False):
    if coefs_real is None:
        coefs_real = pd.read_pickle('data/simulated/coefficients.pkl')
        coefs_real = coefs_real[target_name]       
    else:
        coefs_real = pd.read_csv(coefs_real)
        coefs_real.rename(columns={'variable': 'feat', 'coefficient': 'coef'}, inplace=True)

    comparison = pd.merge(coefs_real, est_coefs.reset_index(), on='feat', how='left').fillna(0)
    
    fig, ax = plt.subplots(1,2, figsize=(13,6))
    
    ax[0].scatter(comparison.coef, comparison['mean'], color='k')
    ax[0] = _plot_diagonal(ax[0])
    if annotate:
        ax[0] = _label_point(comparison.coef, comparison['mean'], comparison.feat, ax[0])
        
    ax[0].set_xlabel('True Coefficient', fontsize=12)
    ax[0].set_ylabel('Estimated Coefficient', fontsize=12)
    ax[0].set_title('True vs Estimated', fontsize=14)
        
    ax[1].scatter(comparison.feat, comparison.coef, color='g', alpha=0.7, label='True')
    ax[1].scatter(comparison.feat, comparison['mean'], color='r', alpha=0.7, label='Est.')
    ax[1].errorbar(comparison.feat, comparison['mean'], yerr=comparison['std'], 
                   ls='none', color='r', alpha=0.3)
    ax[1].legend()
    
    ax[1].set_xticklabels(comparison.feat, rotation=70)
    ax[1].set_title('Coefficient values', fontsize=14)
    
    plt.show()
    
    
def plot_learning_curve(estimator, title, X, y, scoring=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10)):
    
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))

    train_sizes, train_scores, test_scores, fit_times, score_times = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       scoring=scoring,
                       train_sizes=train_sizes,
                       return_times=True)
    
    if not scoring is None:
        if 'neg' in scoring:
            train_scores = -train_scores
            test_scores = -test_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    score_times_mean = np.mean(score_times, axis=1)
    score_times_std = np.std(score_times, axis=1)

    # Plot learning curve
    ax[0][0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    ax[0][0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    ax[0][0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    ax[0][0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    ax[0][0].legend(loc="best")
    ax[0][0].set_title('Train and test scores', fontsize=14)
    if ylim is not None:
        ax[0][0].set_ylim(*ylim)
    ax[0][0].set_xlabel("Training examples")
    ax[0][0].set_ylabel("Score")

    # Plot n_samples vs fit_times
    ax[0][1].plot(train_sizes, fit_times_mean, 'o-')
    ax[0][1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    ax[0][1].set_xlabel("Training examples")
    ax[0][1].set_ylabel("fit_times")
    ax[0][1].set_title("Scalability of the model", fontsize=14)

    # Plot fit_time vs score
    ax[1][0].plot(fit_times_mean, test_scores_mean, 'o-')
    ax[1][0].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    ax[1][0].set_xlabel("fit_times")
    ax[1][0].set_ylabel("Score")
    ax[1][0].set_title("Fit time vs test score", fontsize=14)
    
    # Plot fit_time vs fit_score
    ax[1][1].plot(fit_times_mean, train_scores_mean, 'o-')
    ax[1][1].fill_between(fit_times_mean, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1)
    ax[1][1].set_xlabel("fit_times")
    ax[1][1].set_ylabel("Score")
    ax[1][1].set_title("Fit time vs train score", fontsize=14)
    
    fig.suptitle(f'{title}', fontsize=18)
    
    plt.show()
    
    
def plot_coef_est(coefs_est):
    plt.figure(figsize=(14, 12))
    sns.barplot(x="mean", y="feat", 
                data=coefs_est.head(50).reset_index(), 
                xerr=coefs_est.head(50)['std'])
    plt.show()

