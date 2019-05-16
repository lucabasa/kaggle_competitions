__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'


import numpy as np
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report


def plot_importance(feature_importance_df, save_name):
    cols = (feature_importance_df[["feature", "importance"]]
                    .groupby("feature")
                    .mean().abs()
                    .sort_values(by="importance", ascending=False)[:50].index)

    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

    plt.figure(figsize=(15,8))

    sns.barplot(x="importance",
                y="feature",
                data=best_features.sort_values(by="importance",
                                               ascending=False))

    if not save_name.endswith('.png'):
        save_name += '.png'
    plt.savefig('plots/' + save_name)
    plt.close()


def report_oof(df_train, oof):
    acc = accuracy_score(oof, df_train.Survived)
    f1 = f1_score(oof, df_train.Survived)
    roc = roc_auc_score(oof, df_train.Survived)
    print(f'Oof accuracy: \t {acc}')
    print(f'Oof f1 score: \t {f1}')
    print(f'Oof area under the roc curve: \t {roc}')
    print('Classification report: ')
    print(classification_report(oof, df_train.Survived))
    