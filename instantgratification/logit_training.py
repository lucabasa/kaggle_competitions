__author__ = 'lucabasa'
__version__ = '1.1.0'
__status__ = 'development'


import numpy as np 
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA

pd.set_option('max_columns', 200)

import utility as ut


def train_logit(df_train, df_test, C=1, pca=False):
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

        if pca:
            data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
            #data2 = StandardScaler().fit_transform(PCA(n_components=40, random_state=51).fit_transform(data[cols]))
            data2 = StandardScaler().fit_transform(PCA(svd_solver='full',n_components='mle').fit_transform(data[cols]))
            train3 = data2[:train2.shape[0]]
            test3 = data2[train2.shape[0]:]
        else:
            sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
            train3 = sel.transform(train2[cols])
            test3 = sel.transform(test2[cols])

        skf = StratifiedKFold(n_splits=25, random_state=15)
        for train_index, test_index in skf.split(train3, train2['target']):

            clf = Pipeline([('scaler', StandardScaler()),
                            ('logit', LogisticRegression(solver='saga', penalty='l1', C=C))])
            
            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
            
            oof[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
            preds[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
            
    ut.report_oof(df_train, oof)
    
    return oof, preds


def main():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    oof_logit, preds_logit = train_logit(df_train, df_test, pca=True)
    ut.plot_results(oof_logit, preds_logit, df_train, 'logit_'+str(c))


if __name__ == '__main__':
    main()