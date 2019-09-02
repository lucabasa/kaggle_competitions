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
    

def plot_diagonal(ax):
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    low = min(xmin, xmax)
    high = max(xmin, xmax)
    scl = (high - low) / 100
    
    line = pd.DataFrame({'x': np.arange(low, high ,scl), # small hack for a diagonal line
                         'y': np.arange(low, high ,scl)})
    ax.plot(line.x, line.y, color='black', linestyle='--')
    
    return ax

    
def plot_predictions(train, test, true_label, pred_label, hue=None, legend=False, savename='test.png'):
    
    fig, ax = plt.subplots(1,2, figsize=(15,6))

    sns.scatterplot(x=true_label, y=pred_label, data=train, 
                    hue=hue, ax=ax[0], legend=legend, alpha=0.7)
    sns.scatterplot(x=true_label, y=pred_label, data=test, 
                    hue=hue, ax=ax[1], legend=legend, alpha=0.7)
    
    ax[0].set_title('In fold prediction')
    ax[1].set_title('Oof prediction')
    
    ax[0] = plot_diagonal(ax[0])
    ax[1] = plot_diagonal(ax[1])

    if not save_name.endswith('.png'):
        save_name += '.png'
    plt.savefig('plots/' + save_name)
    plt.close()
