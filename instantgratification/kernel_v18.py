__author__ = 'lucabasa'
__version__ = '1.1.0'
__status__ = 'obsolete'


import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

pd.set_option('max_columns', 200)

import utility as ut


def train_svc(df_train, df_test):
    train = df_train.copy()
    test = df_test.copy()

    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

    for i in range(512):
        train2 = train[train['wheezy-copper-turtle-magic']==i]
        test2 = test[test['wheezy-copper-turtle-magic']==i]
        idx1 = train2.index; idx2 = test2.index
        train2.reset_index(drop=True,inplace=True)

        sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
        train3 = sel.transform(train2[cols])
        test3 = sel.transform(test2[cols])

        skf = StratifiedKFold(n_splits=25, random_state=15)
        for train_index, test_index in skf.split(train3, train2['target']):

            clf = Pipeline([('scaler', StandardScaler()),
                            ('svc', SVC(probability=True,kernel='poly',degree=4,gamma='auto'))])
            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
            oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
            preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
        if i%25==0: 
            print(i)    
  
    ut.report_oof(df_train, oof)
    
    return oof, preds


def train_logit(df_train, df_test):
    train = df_train.copy()
    test = df_test.copy()

    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

    for i in range(512):
        train2 = train[train['wheezy-copper-turtle-magic']==i]
        test2 = test[test['wheezy-copper-turtle-magic']==i]
        idx1 = train2.index; idx2 = test2.index
        train2.reset_index(drop=True,inplace=True)

        sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
        train3 = sel.transform(train2[cols])
        test3 = sel.transform(test2[cols])

        skf = StratifiedKFold(n_splits=25, random_state=15)
        for train_index, test_index in skf.split(train3, train2['target']):

            clf = Pipeline([('scaler', StandardScaler()),
                            ('logit', LogisticRegression(solver='saga', penalty='l1', C=1))])
            
            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
            
            oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
            preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
        
            
    ut.report_oof(df_train, oof)
    
    return oof, preds


if __name__ == '__main__':
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    oof_svc, preds_svc = train_svc(df_train, df_test)

    oof_logit, preds_logit = train_logit(df_train, df_test)

    scores = []

    for i in np.arange(0.001, 1, 0.001):
        score_temp = roc_auc_score(df_train.target, i * oof_svc + (1 - i) * oof_logit)
        scores.append(score_temp)

    max_score = pd.DataFrame({'i': np.arange(0.001, 1, 0.001), 'score': scores})
    use_max = max_score[max_score.score == max(max_score.score)]['i'].values[0]

    sub = pd.read_csv('data/sample_submission.csv')
    sub['target'] =  use_max * preds_svc + (1- use_max) * preds_logit
    sub.to_csv('submissions/v18_svclogit_sub.csv',index=False)