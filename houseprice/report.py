__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_squared_log_error, mean_absolute_error


def print_scores(train, test, inf, oof, target):
    print(f'Train set RMSE: {mean_squared_error(train[target], inf)}')
    print(f'Test set RMSE: {mean_squared_error(test[target], oof)}')
    print(f'Train set logRMSE: {mean_squared_log_error(train[target], inf)}')
    print(f'Test set logRMSE: {mean_squared_log_error(test[target], oof)}')
    print(f'Train set MAE: {mean_absolute_error(train[target], inf)}')
    print(f'Test set MAE: {mean_absolute_error(test[target], oof)}')
    

def plot_predictions(train, test, target, prediction, hue=None, legend=False, savename='test.png'):
    fig, ax = plt.subplots(1,2, figsize=(15,6))

    sns.scatterplot(x=target, y=prediction, data=train, 
                    hue=hue, ax=ax[0], legend=legend, alpha=0.7)
    sns.scatterplot(x=target, y=prediction, data=test, 
                    hue=hue, ax=ax[1], legend=legend, alpha=0.7)
    
    ax[0].set_title('In fold prediction')
    ax[1].set_title('Oof prediction')

    if not save_name.endswith('.png'):
        save_name += '.png'
    plt.savefig('plots/' + save_name)
    plt.close()