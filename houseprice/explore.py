__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns



def list_missing(data, verbose=True):
    mis_cols = [col for col in data.columns if data[col].isna().any()]
    if not verbose:
        return mis_cols
    tot_rows = len(data)
    for col in mis_cols:
        print(f'Column {col}: {round(data[col].isna().sum()*100/tot_rows, 2)}% missing')
    return mis_cols


def plot_correlations(data, target=None, limit=50, figsize=(12,10), **kwargs):
    corr = data.corr()
    if target:
        cor_target = abs(corr[target]).sort_values(ascending=False)
        cor_target = cor_target[:limit]
        corr = corr.loc[cor_target.index, cor_target.index]
    plt.figure(figsize=figsize)
    ax = sns.heatmap(corr, cmap='RdBu_r', **kwargs)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    return cor_target


def plot_distribution(data, column, bins=50, correlation=None):
    plt.figure(figsize=(12,8))
    data[column].hist(bins=bins)
    if not correlation is None:
        value = correlation[column]
        column = column + f' - {round(value,2)}'
    plt.title(f'Distribution of {column}', fontsize=18)
    plt.grid(False)


def plot_bivariate(data, x, y, hue=None, **kwargs):
    plt.figure(figsize=(12,8))
    sns.scatterplot(data=data, x=x, y=y, hue=hue, **kwargs)
    if hue:
        plt.title(f'{x} vs {y}, by {hue}', fontsize=18)
    else:
        plt.title(f'{x} vs {y}', fontsize=18)


def corr_target(data, target, cols, x_estimator=None):
    print(data[cols+[target]].corr())
    num = len(cols)
    rows = int(num/2) + (num % 2 > 0)
    cols = list(cols)
    y = data[target]
    fig, ax = plt.subplots(rows, 2, figsize=(12, 5 * (rows)))
    i = 0
    j = 0
    for feat in cols:
        x = data[feat]
        if (rows > 1):
            sns.regplot(x=x, y=y, ax=ax[i][j], x_estimator=x_estimator)
            j = (j+1)%2
            i = i + 1 - j
        else:
            sns.regplot(x=x, y=y, ax=ax[i], x_estimator=x_estimator)
            i = i+1

def find_cats(data, target, thrs=0.1, agg_func='mean', frac=1):
    cats = []
    tar_std = data[target].std()
    for col in data.select_dtypes(include=['object']).columns:
        counts = data[col].value_counts(dropna=False, 
                                        normalize=True)
        tmp = data.loc[data[col].isin(counts[counts > thrs].index), 
                       :].groupby(col)[target].agg(agg_func).std()
        if tmp >= tar_std*frac:
            cats.append(col)
    return cats


def segm_target(data, cat, target):
    df = data.groupby(cat)[target].agg(['count', 'mean', 'max', 
                                        'min', 'median', 'std'])
    fig, ax = plt.subplots(1,2, figsize=(12, 5))
    sns.boxplot(cat, target, data=data, ax=ax[0])
    for val in data[cat].unique():
        tmp = data[data[cat] == val]
        sns.distplot(tmp[target], hist=False, kde=True,
                 kde_kws = {'linewidth': 3},
                 label = val, ax=ax[1])  
    return df

 