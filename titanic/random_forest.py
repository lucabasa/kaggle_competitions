__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'


import numpy as np
import pandas as pd 

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, RandomizedSearchCV

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

from sklearn.ensemble import RandomForestClassifier


def tune_rf(train, target, verbose=False):
    grid_param = {'max_depth': np.arange(3,30),
                'min_samples_split': np.arange(2, 50), 
                'min_samples_leaf': np.arange(1,40), 
                'max_features': ['sqrt', 'log2', None]}

    kfolds = KFold(5, shuffle=True, random_state=14)

    grid = RandomizedSearchCV(RandomForestClassifier(n_estimators=300, n_jobs=4, random_state=345),
                            param_distributions=grid_param, n_iter=50, cv=kfolds, 
                            random_state=654, n_jobs=-1, scoring='roc_auc')

    grid.fit(train, target)

    best_forest = grid.best_estimator_

    if verbose:
        print(grid.best_params_)
        print(round( grid.best_score_  ,3))

    return grid.best_params_


def quick_forest(df_train, df_test, kfolds):
    # preparing data with minimal features
    train = df_train.dropna(axis=1).copy()
    test = df_test.dropna(axis=1).copy()

    target = train.Survived.copy()

    sub = test[['PassengerId']].copy()

    del train['Survived']
    del train['Name']
    del train['Ticket']
    del train['PassengerId']
    del test['PassengerId']
    del test['Name']
    del test['Embarked']

    train['Sex'] = train.Sex.map({'male': 1, 'female': 0}).astype(int)
    test['Sex'] = test.Sex.map({'male': 1, 'female': 0}).astype(int)

    # model 
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    comm_cols = list(set(train.columns).intersection(test.columns))

    for fold_, (trn_idx, val_idx) in enumerate(kfolds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))

        trn_data = train.iloc[trn_idx][comm_cols]
        val_data = train.iloc[val_idx][comm_cols]

        trn_target = target.iloc[trn_idx]
        val_target = target.iloc[val_idx]
        
        params = tune_rf(trn_data, trn_target, verbose=True)
        forest = RandomForestClassifier(n_estimators=700, n_jobs=4, random_state=189,
                                            max_depth=params['max_depth'], 
                                            min_samples_split=params['min_samples_split'],
                                            min_samples_leaf=params['min_samples_leaf'],
                                            max_features=params['max_features'])

        clf = forest.fit(trn_data, trn_target)

        oof[val_idx] = clf.predict(train.iloc[val_idx][comm_cols])
        predictions += clf.predict_proba(test[comm_cols])[:,1] / kfolds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = comm_cols
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        
    print(accuracy_score(oof, df_train.Survived))
    print(f1_score(oof, df_train.Survived))
    print(roc_auc_score(oof, df_train.Survived))
    print(classification_report(oof, df_train.Survived))

    cols = (feature_importance_df[["feature", "importance"]]
                    .groupby("feature")
                    .mean().abs()
                    .sort_values(by="importance", ascending=False)[:50].index)

    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

    sns.barplot(x="importance",
                y="feature",
                data=best_features.sort_values(by="importance",
                                               ascending=False))

    plt.savefig('plots/rf_simple_featimp.png')
    plt.close()

    sub['Survived'] = (predictions > 0.5).astype(int) 

    sub.to_csv('submissions/rf_simple.csv', index=False)


def forest_w_imputation(df_train, df_test, kfolds):
    # preparing data with minimal features
    train = df_train.dropna(axis=1).copy()
    test = df_test.dropna(axis=1).copy()

    target = train.Survived.copy()

    sub = test[['PassengerId']].copy()

    train['Sex'] = train.Sex.map({'male': 1, 'female': 0}).astype(int)
    test['Sex'] = test.Sex.map({'male': 1, 'female': 0}).astype(int)

    # flagging missing data
    train['MisAge'] = 0
    train.loc[train.Age.isna(), 'MisAge'] = 1
    train['MisCab'] = 0
    train.loc[train.Cabin.isna(), 'MisCab'] = 1

    test['MisAge'] = 0
    test.loc[test.Age.isna(), 'MisAge'] = 1
    test['MisCab'] = 0
    test.loc[test.Cabin.isna(), 'MisCab'] = 1

    del train['Survived']
    del train['Name']
    del train['Ticket']
    del train['PassengerId']
    del train['Cabin']
    del test['PassengerId']
    del test['Name']
    del test['Ticket']
    del test['Cabin']
   
    # model
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()

    # using train data to impute the missing test entries
    test.loc[test.Age.isna(), 'Age'] = train.Age.median()
    test.loc[test.Fare.isna(), 'Fare'] = train.Fare.median()

    test = pd.get_dummies(test)

    for fold_, (trn_idx, val_idx) in enumerate(kfolds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))

        trn_data = train.iloc[trn_idx].copy()
        val_data = train.iloc[val_idx].copy()

        trn_target = target.iloc[trn_idx]
        val_target = target.iloc[val_idx]
        
        # impute only with the training data of the fold
        trn_data.loc[trn_data.Age.isna(), 'Age'] = trn_data.Age.median()
        trn_data.loc[trn_data.Embarked.isna(), 'Embarked'] =trn_data.Embarked.mode().values[0]
        
        val_data.loc[val_data.Age.isna(), 'Age'] = trn_data.Age.median()
        val_data.loc[val_data.Embarked.isna(), 'Embarked'] = trn_data.Embarked.mode().values[0]
        
        trn_data = pd.get_dummies(trn_data)
        val_data = pd.get_dummies(val_data)

        params = tune_rf(trn_data, trn_target, verbose=True)
        forest = RandomForestClassifier(n_estimators=700, n_jobs=4, random_state=189,
                                        max_depth=params['max_depth'], 
                                        min_samples_split=params['min_samples_split'],
                                        min_samples_leaf=params['min_samples_leaf'],
                                        max_features=params['max_features'])

        clf = forest.fit(trn_data, trn_target)

        oof[val_idx] = clf.predict(val_data)
        predictions += clf.predict_proba(test)[:,1] / kfolds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = trn_data.columns
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print(accuracy_score(oof, df_train.Survived))
    print(f1_score(oof, df_train.Survived))
    print(roc_auc_score(oof, df_train.Survived))
    print(classification_report(oof, df_train.Survived))

    cols = (feature_importance_df[["feature", "importance"]]
                    .groupby("feature")
                    .mean().abs()
                    .sort_values(by="importance", ascending=False)[:50].index)

    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

    sns.barplot(x="importance",
                y="feature",
                data=best_features.sort_values(by="importance",
                                               ascending=False))

    plt.savefig('plots/rf_imputed_featimp.png')
    plt.close()

    sub['Survived'] = (predictions > 0.5).astype(int) 

    sub.to_csv('submissions/rf_imputed.csv', index=False)



def main():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    kfolds = KFold(n_splits=5, shuffle=True, random_state=498)

    quick_forest(df_train, df_test, kfolds)

    forest_w_imputation(df_train, df_test, kfolds)


if __name__ == '__main__':
    main()
