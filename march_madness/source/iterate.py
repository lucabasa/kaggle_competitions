__author__ = 'lucabasa'
__version__ = '1.0.0'

import numpy as np


class Iterator:
    def __init__(self, train, test, target_cols, seeds, n_folds, train_func, train_args, verbose=False):
        self.train = train
        self.test = test
        self.target_cols = target_cols
        self.seeds = seeds
        self.n_folds = n_folds
        self.train_func = train_func
        self.train_args = train_args
        self.verbose = verbose
        

    def it_seeds(self):

        oof = np.zeros((len(self.train), len(self.target_cols)))
        predictions = np.zeros((len(self.test), len(self.target_cols)))

        for seed in self.seeds:
            if self.verbose:
                print(f"SEED: {seed}")

            oof_seed, predictions_seed = self.it_folds(seed=seed)
            oof += oof_seed / len(self.seeds)
            predictions += predictions_seed / len(self.seeds)

        return oof, predictions


    def it_folds(self, seed):
        oof = np.zeros((len(self.train), len(self.target_cols)))
        predictions = np.zeros((len(self.test), len(self.target_cols)))

        for fold in range(self.n_folds):
            if self.verbose:
                print(f'\t FOLD: {fold}')
            
            oof_fold, pred_fold = self.train_func(seed=seed, fold=fold, verbose=self.verbose, **self.train_args)

            predictions += pred_fold / self.n_folds
            oof += oof_fold

        return oof, predictions
