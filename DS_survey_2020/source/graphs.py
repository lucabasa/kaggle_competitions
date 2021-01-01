__author__ = 'lucabasa'
__version__ = '0.0.1'


import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec

import pandas as pd


def newline(ax, p1, p2, color='black'):
    l = mlines.Line2D([p1[1],p2[1]], [p1[0],p2[0]], color=color)
    ax.add_line(l)
    return ax


def plot_frame(ax):
    ax.set_facecolor('#292525')
    ax.spines['bottom'].set_color('w')
    ax.tick_params(axis='x', colors='w')
    ax.xaxis.label.set_color('w')
    ax.spines['left'].set_color('w')
    ax.tick_params(axis='y', colors='w')
    ax.yaxis.label.set_color('w')
    return ax


def annotate_bar(ax, data):
    rects = ax.patches
    labels = [f'n = {count}' for count in data['count']]
    if data['mean'].max() > 1000:
        x_pos = 1000
    else:
        x_pos = 0.1
    for rect, label in zip(rects, labels):
        y_value = rect.get_y() + rect.get_height() / 2
        ax.text(x_pos, y_value, label, color='w', va='center')
    return ax


def plot_means(data, target, question, title, sub_title, target_title):
    fig = plt.figure(figsize=(15, 14), facecolor='#292525')
    fig.subplots_adjust(top=0.94)
    fig.suptitle(title, fontsize=18, color='w')

    gs = GridSpec(2, 9, figure=fig)
    ax0 = fig.add_subplot(gs[0, :4])
    ax1 = fig.add_subplot(gs[0, 5:])
    ax2 = fig.add_subplot(gs[1, :])

    tmp = data.groupby(question)[target].agg(['mean', 'count'])
    tmp['mean'].plot(kind='barh', ax=ax0, color='#3A3FDC')
    ax0.axvline(data[target].mean(), color='w', linestyle='--')
    ax0 = annotate_bar(ax0, tmp)
    ax0.set_title(f'Mean {target_title} by {sub_title}', color='w', fontsize=14)
    
    tmp = data.groupby('Q2')[target].agg(['mean', 'count'])
    tmp['mean'].plot(kind='barh', ax=ax1, color='#3A3FDC')
    ax1.axvline(data[target].mean(), color='w', linestyle='--')
    ax1 = annotate_bar(ax1, tmp)
    ax1.set_title(f'Mean {target_title} by Gender', color='w', fontsize=14)

    tmp = data.groupby([question, 'Q2'])[target].mean().unstack().reset_index()

    counts = data.groupby([question, 'Q2']).size().unstack()

    ax2.scatter(y=tmp['Man'], x=tmp[question], color='#C3C92E', s=100, alpha=0.8, label='Men')
    ax2.scatter(y=tmp['Woman'], x=tmp[question], color='#C93D2E', s=100, alpha=0.8, label='Women')
    ax2.scatter(y=tmp['Other'], x=tmp[question], color='#41DA5B', s=100, alpha=0.8, label='Other')

    ax2.axhline(data[data.Q2 == 'Man'][target].mean(), color='#C3C92E', linestyle='--', alpha=0.5)
    ax2.axhline(data[data.Q2 == 'Woman'][target].mean(), color='#C93D2E', linestyle='--', alpha=0.5)
    ax2.axhline(data[data.Q2 == 'Other'][target].mean(), color='#41DA5B', linestyle='--', alpha=0.5)
    legend = ax2.legend(facecolor='#292525', edgecolor='#292525')
    plt.setp(legend.get_texts(), color='w')
    ax2.set_title(f'Mean {target_title} by {sub_title} and Gender', color='w', fontsize=14)
    plt.draw()
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=35)
    

    for i, p1, p2 in zip(tmp[question], tmp['Man'], tmp['Woman']):
            ax2 = newline(ax2, [p1, i], [p2, i], color='silver')
    for i, p1, p2 in zip(tmp[question], tmp['Man'], tmp['Other']):
            ax2 = newline(ax2, [p1, i], [p2, i], color='silver')
            

    for ax in [ax0, ax1, ax2]:
        ax = plot_frame(ax)
        ax.set_xlabel('')
        ax.set_ylabel('')
        
    ax2.set_ylabel(f'Mean {target_title}', color='w', fontsize=10)

    plt.show()