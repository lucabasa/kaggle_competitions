__author__ = 'lucabasa'
__version__  = '0.0.1'


import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, roc_curve, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

import warnings


def plot_confusion_matrix(true_label, pred_label, ax=None, thrs=0.5):
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(12,12))
        end_action = True
    else:
        end_action = False
        
    if len(set(pred_label)) > 2:
        preds = pred_label > thrs
    else:
        preds = pred_label
    cm = confusion_matrix(y_true=true_label, y_pred=preds, normalize='all')
    sns.heatmap(cm, cbar=False, ax=ax,
                annot=True, fmt=".2%", annot_kws={"size":15},
                linewidths=.5, cmap="coolwarm")
    ax.set_xlabel('Predicted labels', fontsize=14)
    ax.set_ylabel('True labels', fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    
    if end_action:
        plt.show()
    else:
        return ax


def plot_classification_probs(data, true_label, pred_label, thrs=0.5, sample=None, feat=None, hue_feat=None, savename=None):
    '''
    Plot prediction vs true label when the prediction is a probability
    Plots also a confusion matrix, a 
    Data, true_label, and pred_label must be of compatible size
    hue_feat is ignored when the unique values are more than 5 for readability
    '''
    
    # prepare data
    tmp = data.copy()
    tmp['Prediction'] = pred_label
    tmp['True Label'] = true_label
    # the x axis of the third plot has to be a continuos feature
    # If missing, make one
    if feat is None:  
        tmp['DUMMY_FEAT'] = np.arange(0, len(tmp))
        feat = 'DUMMY_FEAT'
    
    fig, ax = plt.subplots(2,2, figsize=(15,12))
    
    # Plotting probability vs true label
    tmp[tmp['True Label']==1]['Prediction'].hist(bins=50, ax=ax[0][0], alpha=0.5, color='g', label='1')
    tmp[tmp['True Label']==0]['Prediction'].hist(bins=50, ax=ax[0][0], alpha=0.5, color='r', label='0')
    ax[0][0].axvline(thrs, color='k', linestyle='--')
    ax[0][0].grid(False)
    ax[0][0].legend()
    ax[0][0].set_title('Predicted Probability vs True Label', fontsize=14)
    
    # Confusion matrix
    ax[0][1] = plot_confusion_matrix(tmp['True Label'], tmp['Prediction'], ax=ax[0][1], thrs=thrs)
    
    # This is to allow for segmenting the data by a categorical feature
    addition=''
    if hue_feat is not None:
        if tmp[hue_feat].nunique() > 5:
            warnings.warn(f'{hue_feat} has more than 5 unique values, hue will be ignored', UserWarning)
            hue_feat = None
            sizes = None
        else:
            addition = f' by {hue_feat}'
            # The hue_feature will be displayed with size
            n_vals = tmp[hue_feat].nunique()
            sizes = {val: (i+1)*100/n_vals for i, val in enumerate(np.sort(tmp[hue_feat].unique()))}
    else:
        sizes = None
    
    # Barplot of mean target vs mean prediction
    if hue_feat is None:
        tmp[['True Label', 'Prediction']].mean().plot(kind='bar', 
                                                      ax=ax[1][1], color=['r', 'g'], alpha=0.7)
    else:
        tmp.groupby(hue_feat)[['True Label', 'Prediction']].mean().plot(kind='bar', 
                                                                        ax=ax[1][1], color=['r', 'g'], alpha=0.7)
    
    ax[1][1].axhline(tmp['True Label'].mean(), color='r', linestyle='--')
    ax[1][1].axhline(tmp['Prediction'].mean(), color='g', linestyle='--')
    ax[1][1].set_xticklabels(ax[1][1].get_xticklabels(), rotation=0)
    ax[1][1].set_ylim((0,1))
    ax[1][1].set_title(f'Mean Label vs Mean Prediction{addition}', fontsize=14)
    
    if sample:  # to make the plot more readable
        tmp = tmp.sample(sample)
    
    # Plot continuous feature vs prediction, the true label is the hue
    # Possible to add size to also see the role of categorical feature
    sns.scatterplot(x=feat, y='Prediction', data=tmp, ax=ax[1][0], palette=['r', 'g'],
                    size=hue_feat, sizes=sizes,
                    hue='True Label', legend=True, alpha=0.3, edgecolor=None)
    
    
    ax[1][0].axhline(thrs, color='k', linestyle='--')
    ax[1][0].set_title(f'True Label vs Prediction{addition}', fontsize=14)
    
    if savename is not None:
        plt.savefig(savename)
        plt.close()
    else:
        plt.show()


def eval_classification(data, target, preds, proba=False, thrs=0.5, feats=None, plot=True, **kwargs):
    if proba:
        preds_bin = preds >= thrs
        fpr, tpr, _ = roc_curve(target, preds)
        if plot > 0:
            plot_classification_probs(data, target, preds, thrs=thrs, **kwargs)
        if plot > 1:
            plt.plot(fpr, tpr)
            plt.title('ROC', fontsize=14)
    else:
        preds_bin = preds
        if plot > 0:
            plot_confusion_matrix(target, preds)
    
    print(f'Accuracy score: \t{round(accuracy_score(target, preds_bin), 4)}')
    print(f'AUC ROC: \t\t{round(roc_auc_score(target, preds), 4)}')
    print(classification_report(target, preds_bin))
    