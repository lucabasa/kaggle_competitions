__author__ = 'lucabasa'
__version__ = '1.1.2'
__status__ = 'development'


import numpy as np
import pandas as pd 

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import processing as pr
import report as rep


def tune_rf(train, target, verbose=False):
    grid_param = {'max_depth': np.arange(3,30),
                'min_samples_split': np.arange(2, 50), 
                'min_samples_leaf': np.arange(1,40), 
                'max_features': ['sqrt', 'log2', None]}

    kfolds = KFold(5, shuffle=True, random_state=14)

    grid = RandomizedSearchCV(RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=345),
                            param_distributions=grid_param, n_iter=200, cv=kfolds, 
                            random_state=654, n_jobs=-1, scoring='roc_auc')

    grid.fit(train, target)

    best_forest = grid.best_estimator_

    if verbose:
        print(grid.best_params_)
        print(round( grid.best_score_  ,3))

    return grid.best_params_


def general_processing(train, test):
    # processing train and test outside the cv loop
    train['Sex'] = train.Sex.map({'male': 1, 'female': 0}).astype(int)
    test['Sex'] = test.Sex.map({'male': 1, 'female': 0}).astype(int)

    # flagging missing data
    train = pr.flag_missing(train, ['Age', 'Cabin'])
    test = pr.flag_missing(test, ['Age', 'Cabin'])

    # fam size
    train['fam_size'] = train['SibSp'] + train['Parch'] + 1
    test['fam_size'] = test['SibSp'] + test['Parch'] + 1

    # Gender and class
    train = pr.gen_clas(train)
    test = pr.gen_clas(test)

    # FamSize and Class
    train['fs_cl'] = train.fam_size * train.Pclass
    test['fs_cl'] = test.fam_size * test.Pclass

    # isAlone  
    train['is_alone'] = 0
    train.loc[train.fam_size==1, 'is_alone'] = 1
    test['is_alone'] = 0
    test.loc[test.fam_size==1, 'is_alone'] = 1

    #big families
    train['big_fam'] = 0
    train.loc[train.fam_size > 5, 'big_fam'] = 1
    test['big_fam'] = 0
    test.loc[test.fam_size > 5, 'big_fam'] = 1

    # Missing cabin and gender
    train = pr.gen_cab(train)
    test = pr.gen_cab(test)

    # cleaning up unused columns
    to_drop = ['Survived', 'Name', 'Ticket', 'PassengerId', 'Cabin']
    train = pr.clean_cols(train, to_drop)
    test = pr.clean_cols(test, to_drop)
    
    return train, test


def impute_test(train, test):
    # using train data to impute the missing test entries
    test.loc[test.Age.isna(), 'Age'] = train.Age.median()
    test.loc[test.Fare.isna(), 'Fare'] = train.Fare.median()

    test = pr.baby(test)

    test = pd.get_dummies(test)

    return test


def process_fold(trn_fold, val_fold):
    # impute only with the training data of the fold
    trn_fold.loc[trn_fold.Age.isna(), 'Age'] = trn_fold.Age.median()
    trn_fold.loc[trn_fold.Embarked.isna(), 'Embarked'] =trn_fold.Embarked.mode().values[0]
    
    val_fold.loc[val_fold.Age.isna(), 'Age'] = trn_fold.Age.median()
    val_fold.loc[val_fold.Embarked.isna(), 'Embarked'] = trn_fold.Embarked.mode().values[0]

    trn_fold = pr.baby(trn_fold)
    val_fold = pr.baby(val_fold)
    
    trn_fold = pd.get_dummies(trn_fold)
    val_fold = pd.get_dummies(val_fold)

    return trn_fold, val_fold


def train_rf(df_train, df_test, kfolds):
    # preparing data with minimal features
    train = df_train.copy()
    test = df_test.copy()

    target = train.Survived.copy()

    sub = test[['PassengerId']].copy()

    train, test = general_processing(train, test)

    # model
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    test = impute_test(train, test)

    for fold_, (trn_idx, val_idx) in enumerate(kfolds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))

        trn_data = train.iloc[trn_idx].copy()
        val_data = train.iloc[val_idx].copy()

        trn_target = target.iloc[trn_idx]
        val_target = target.iloc[val_idx]
        
        trn_data, val_data = process_fold(trn_data, val_data)

        params = tune_rf(trn_data, trn_target, verbose=True)
        forest = RandomForestClassifier(n_estimators=700, n_jobs=4, random_state=189,
                                        max_depth=params['max_depth'], 
                                        min_samples_split=params['min_samples_split'],
                                        min_samples_leaf=params['min_samples_leaf'],
                                        max_features=params['max_features'])

        clf = forest.fit(trn_data, trn_target)

        oof[val_idx] = clf.predict_proba(val_data)[:,1]
        predictions += clf.predict_proba(test)[:,1] / kfolds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = trn_data.columns
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    rep.report_oof(df_train, (oof > 0.5).astype(int))

    rep.plot_importance(feature_importance_df, 'rf_fe_featimp')

    sub['Survived'] = (predictions > 0.5).astype(int) 

    sub.to_csv('submissions/rf_feat_eng.csv', index=False)

    rf_oof = df_train.copy()
    rf_oof['oof'] = oof
    rf_oof.to_csv('oof_pred/random_forest.csv', index=False)
    sub.to_csv('oof_pred/random_forest_test.csv', index=False)


def main():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    kfolds = KFold(n_splits=5, shuffle=True, random_state=498)

    train_rf(df_train, df_test, kfolds)


if __name__ == '__main__':
    main()
