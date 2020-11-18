__author__ = 'lucabasa'
__version__ = '1.0.0'

import matplotlib.pyplot as plt


def plot_learning(train_losses, valid_losses, fold):
    
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    
    ax.plot(train_losses[2:], color='g')
    ax.plot(valid_losses[2:], color='r')
    
    plt.savefig(f'plots/learning_fold_{fold}.png')
    plt.close()