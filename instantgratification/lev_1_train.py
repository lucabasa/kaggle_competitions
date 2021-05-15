__author__ = 'lucabasa'
__version__ = '1.1.0'
__status__ = 'development'

import numpy as np 
import pandas as pd 

from sklearn.model_selection import StratifiedKFold

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, NuSVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold

import utility as ut 


def train_all(df_train, df_test, n_folds, pca=False):
    train = df_train.copy()
    test = df_test.copy()

    oof_svc = np.zeros(len(train))
    oof_nusvc = np.zeros(len(train))
    oof_logit = np.zeros(len(train))
    oof_knn = np.zeros(len(train))
    oof_qda = np.zeros(len(train))
    preds_svc = np.zeros(len(test))
    preds_nusvc = np.zeros(len(test))
    preds_logit = np.zeros(len(test))
    preds_knn = np.zeros(len(test))
    preds_qda = np.zeros(len(test))

    cols = [c for c in train.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]

    for i in range(512):
        train2 = train[train['wheezy-copper-turtle-magic']==i].copy()
        test2 = test[test['wheezy-copper-turtle-magic']==i].copy()
        if len(train2) == 0:
            continue
        idx1 = train2.index; idx2 = test2.index
        train2.reset_index(drop=True,inplace=True)
        
        if pca:
            data = pd.concat([pd.DataFrame(train2[cols]), pd.DataFrame(test2[cols])])
            #data2 = StandardScaler().fit_transform(PCA(n_components=40, random_state=51).fit_transform(data[cols]))
            data2 = StandardScaler().fit_transform(PCA(svd_solver='full',n_components='mle').fit_transform(data[cols]))
            train3 = data2[:train2.shape[0]]; test3 = data2[train2.shape[0]:]
        else:
            sel = VarianceThreshold(threshold=1.5).fit(train2[cols])
            train3 = sel.transform(train2[cols])
            test3 = sel.transform(test2[cols])

        skf = StratifiedKFold(n_splits=n_folds, random_state=15)

        for train_index, test_index in skf.split(train3, train2['target']):

            clf = Pipeline([('scaler', StandardScaler()),
                            ('svn', SVC(probability=True,kernel='poly',degree=4,gamma='auto'))])
            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
            oof_svc[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
            preds_svc[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits
            
            clf = Pipeline([('scaler', StandardScaler()),
                            ('svn', NuSVC(probability=True, kernel='poly', degree=4, 
                                          gamma='auto', random_state=745, nu=0.59, coef0=0.053))])
            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
            oof_nusvc[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
            preds_nusvc[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

            clf = Pipeline([('scaler', StandardScaler()),
                            ('logit', LogisticRegression(solver='saga', penalty='l1', C=0.5))])
            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
            oof_logit[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
            preds_logit[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

            clf = Pipeline([('scaler', StandardScaler()),
                            ('KNN', KNeighborsClassifier(n_neighbors=17, p=2.9))])
            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
            oof_knn[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
            preds_knn[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

            clf = QuadraticDiscriminantAnalysis(reg_param=0.6)
            clf.fit(train3[train_index,:],train2.loc[train_index]['target'])
            oof_qda[idx1[test_index]] = clf.predict_proba(train3[test_index,:])[:,1]
            preds_qda[idx2] += clf.predict_proba(test3)[:,1] / skf.n_splits

                 
    ut.report_oof(df_train, oof_svc)
    ut.report_oof(df_train, oof_nusvc)
    ut.report_oof(df_train, oof_logit)
    ut.report_oof(df_train, oof_knn)
    ut.report_oof(df_train, oof_kda)

    return oof_svc, preds_svc, oof_nusvc, preds_nusvc, oof_logit, preds_logit, oof_knn, preds_knn, oof_qda, preds_qda


def main():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    sample_train = 100
    n_folds=25
    if sample_train:
        import random
        ints = random.sample(range(0, 512), sample_train)

        df_train = df_train[df_train['wheezy-copper-turtle-magic'].isin(ints)].copy()
        df_test = df_test[df_test['wheezy-copper-turtle-magic'].isin(ints)].copy()
        df_train.reset_index(drop=True,inplace=True)
        df_test.reset_index(drop=True,inplace=True)

        print(df_train.shape, df_test.shape)

    oof_svc, preds_svc, oof_nusvc, preds_nusvc, oof_logit, preds_logit, oof_knn, preds_knn, oof_qda, preds_qda = train_all(df_train, df_test, n_folds)

    ut.subs(df_train, df_test, oof_svc, preds_svc, 'svc', n_folds=n_folds, sel='var', sample=sample_train)
    ut.subs(df_train, df_test, oof_nusvc, preds_nusvc, 'nu_svc', n_folds=n_folds, sel='var', sample=sample_train)
    ut.subs(df_train, df_test, oof_logit, preds_logit, 'logit', n_folds=n_folds, sel='var', sample=sample_train)
    ut.subs(df_train, df_test, oof_knn, preds_knn, 'knn', n_folds=n_folds, sel='var', sample=sample_train)
    ut.subs(df_train, df_test, oof_qda, preds_qda, 'qda', n_folds=n_folds, sel='var', sample=sample_train)

    oof_svc, preds_svc, oof_nusvc, preds_nusvc, oof_logit, preds_logit, oof_knn, preds_knn, oof_qda, preds_qda = train_all(df_train, df_test, n_folds, pca=True)

    ut.subs(df_train, df_test, oof_svc, preds_svc, 'svc', n_folds=n_folds, sel='pca', sample=sample_train)
    ut.subs(df_train, df_test, oof_nusvc, preds_nusvc, 'nu_svc', n_folds=n_folds, sel='pca', sample=sample_train)
    ut.subs(df_train, df_test, oof_logit, preds_logit, 'logit', n_folds=n_folds, sel='pca', sample=sample_train)
    ut.subs(df_train, df_test, oof_knn, preds_knn, 'knn', n_folds=n_folds, sel='pca', sample=sample_train)
    ut.subs(df_train, df_test, oof_qda, preds_qda, 'qda', n_folds=n_folds, sel='pca', sample=sample_train)

if __name__ == '__main__':
    main()
