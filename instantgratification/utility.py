__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'


import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report


def report_oof(df_train, oof):
    acc = accuracy_score((oof > 0.5).astype(int), df_train.target)
    f1 = f1_score((oof > 0.5).astype(int), df_train.target)
    roc = roc_auc_score(df_train.target, oof)
    print(f'Oof accuracy: \t {acc}')
    print(f'Oof f1 score: \t {f1}')
    print(f'Oof area under the roc curve: \t {roc}')
    print('Classification report: ')
    print(classification_report((oof > 0.5).astype(int), df_train.target))


def clean_cols(data, col_list):
    df = data.copy()
    for col in col_list:
        try:
            del df[col]
        except KeyError:
            pass

    return df


def general_processing(train, test):
    # cleaning up unused columns
    to_drop = ['id', 'target']
    train = clean_cols(train, to_drop)
    test = clean_cols(test, to_drop)
    
    train['wheezy-copper-turtle-magic'] = train['wheezy-copper-turtle-magic'].astype('category')
    test['wheezy-copper-turtle-magic'] = test['wheezy-copper-turtle-magic'].astype('category')
    
    return train, test


def plot_results(oof, preds, df_train, save_name):
    if not save_name.endswith('.png'):
        save_name += '.png'

    pd.Series(oof).hist(bins=50, label='oof', alpha=0.7)
    pd.Series(preds).hist(bins=50, label='prediction')
    plt.grid(False)
    plt.legend()
    plt.savefig('plots/preds_' + save_name)
    plt.close()

    err = df_train.copy()
    err['oof'] = oof
    plt.figure(figsize=(15,8))
    sns.scatterplot(data=err, y='oof', x='wheezy-copper-turtle-magic', hue='target', alpha=0.7)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.savefig('plots/oof_' + save_name)
    plt.close()


def subs(df_train, df_test, oof, preds, save_name, n_folds, sel='var', sample=250):
    train = df_train[['id', 'target']].copy()
    test = df_test[['id']].copy()

    train[save_name] = oof
    test[save_name] = preds

    train.to_csv(f'off_preds/{save_name}_{sel}_{n_folds}_{sample}_oof.csv', index=False)
    test.to_csv(f'oof_preds/{save_name}_{sel}_{n_folds}_{sample}_preds.csv', index=False)
