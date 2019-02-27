__author__ = 'lucabasa'
__version__ = "1.0.0"

"""
This script sets the baseline result to benchmark the results of the models
"""

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold


def baseline(data, cv=5):
    """
    data: training data 
    cv = cross-validation scheme to use, integer 

    """

    kfolds = KFold(n_splits=cv, shuffle=True, random_state=14)
    target = np.log1p(data['SalePrice'])
    old_target = data['SalePrice']


    scores = []
    or_scores = []

    for i,(train_index, test_index) in enumerate(kfolds.split(target)):
        print(f"Fold {i} in progress")

        base = target.iloc[train_index].median()
        result = target.iloc[test_index]
        or_result = old_target.iloc[test_index]

        tmp = pd.DataFrame({'result' : result, 'original_result': or_result})
        tmp['prediction'] = base 

        print("Predicting with median {}".format(round(base,3)))
        score = mean_squared_error(y_pred= tmp['prediction'] , 
                                    y_true=tmp['result'])
        or_score = mean_absolute_error(y_pred=np.expm1(tmp['prediction']), 
                                    y_true=tmp['original_result'])

        print(f"Scoring {score}")
        print(f"MAE {or_score}$")

        scores.append(score)
        or_scores.append(or_score)
        print("_"*40)
        
    print("Baseline: {} +- {}".format(round(np.mean(scores),3), round(np.std(scores),3)))
    print('MAE: {} +- {}'.format(round(np.mean(or_scores),3), round(np.std(or_scores),3)))

    return scores, or_scores




if __name__ =='__main__':
    df_train = pd.read_csv('../train.csv')
    scores, or_scores = baseline(df_train, cv=10)

