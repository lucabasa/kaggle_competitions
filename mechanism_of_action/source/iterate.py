__author__ = 'lucabasa'
__version__ = '1.0.0'

import numpy as np


def it_seeds(train, target_cols, test, seeds, folds_func, verbose=False, **kwargs):
    
    oof = np.zeros((len(train), len(target_cols)))
    predictions = np.zeros((len(test), len(target_cols)))

    for seed in seeds:
        if verbose:
            print(f"SEED: {seed}")

        oof_seed, predictions_seed = folds_func(seed, **kwargs)
        oof += oof_seed / len(seeds)
        predictions += predictions_seed / len(seeds)
        
    return oof, predictions