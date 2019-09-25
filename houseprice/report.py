__author__ = 'lucabasa'
__version__ = '1.1.0'
__status__ = 'development'


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error


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

    
def plot_predictions(data, true_label, pred_label, hue=None, legend=False, savename='test.png'):
    
    tmp = data.copy()
    tmp['Prediction'] = pred_label
    tmp['True Label'] = true_label

    plt.figure(figsize=(15,6))

    ax = sns.scatterplot(x='True Label', y='Prediction', data=tmp, 
                         hue=hue, legend=legend, alpha=0.7)
    ax = _plot_diagonal(ax)
    
    if not savename.endswith('.png'):
        savename += '.png'
    plt.savefig('plots/' + savename)
    plt.close()


def get_coef(pipe):
    '''
    Get dataframe with coefficients of a model in Pipeline
    The step before the model has to have a get_feature_name method
    '''
    imp = pipe.steps[-1][1].coef_.tolist()
    feats = pipe.steps[-2][1].get_feature_names()

    result = pd.DataFrame({'feat':feats,'score':imp})
    result = result.sort_values(by=['score'],ascending=False)

    return result


def evaluate(y_true, y_pred, data, hue=None, legend=False, savename='test.png'):

    print(f'RMSE: {round(mean_squared_error(y_true, y_pred), 4)}')
    print(f'MAE: {round(mean_absolute_error(np.expm1(y_true), np.expm1(y_pred)), 4)}')

    plot_predictions(data, y_true, y_pred, hue, legend, savename)




