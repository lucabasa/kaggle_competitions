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


def clean_cols(data, col_list):
    df = data.copy()
    for col in col_list:
        try:
            del df[col]
        except KeyError:
            pass

    return df


def general_processing(train, test):
    # processing train and test outside the cv loop
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

    # fam size
    train['FamSize'] = train['SibSp'] + train['Parch'] + 1
    test['FamSize'] = test['SibSp'] + test['Parch'] + 1

    # Gender and class
    train.loc[(train.Sex == 1) & (train.Pclass == 1), 'se_cl'] = 'male_1'
    train.loc[(train.Sex == 1) & (train.Pclass == 2), 'se_cl'] = 'male_2'
    train.loc[(train.Sex == 1) & (train.Pclass == 3), 'se_cl'] = 'male_3'
    train.loc[(train.Sex == 0) & (train.Pclass == 1), 'se_cl'] = 'female_1'
    train.loc[(train.Sex == 0) & (train.Pclass == 2), 'se_cl'] = 'female_2'
    train.loc[(train.Sex == 0) & (train.Pclass == 3), 'se_cl'] = 'female_3'
    test.loc[(test.Sex == 1) & (test.Pclass == 1), 'se_cl'] = 'male_1'
    test.loc[(test.Sex == 1) & (test.Pclass == 2), 'se_cl'] = 'male_2'
    test.loc[(test.Sex == 1) & (test.Pclass == 3), 'se_cl'] = 'male_3'
    test.loc[(test.Sex == 0) & (test.Pclass == 1), 'se_cl'] = 'female_1'
    test.loc[(test.Sex == 0) & (test.Pclass == 2), 'se_cl'] = 'female_2'
    test.loc[(test.Sex == 0) & (test.Pclass == 3), 'se_cl'] = 'female_3'

    # FamSize and Class
    train['fs_cl'] = train.FamSize * train.Pclass
    test['fs_cl'] = test.FamSize * test.Pclass
    
    to_drop = ['Survived', 'Name', 'Ticket', 'PassengerId', 'Cabin']
    
    train = clean_cols(train, to_drop)
    test = clean_cols(test, to_drop)
    
    return train, test


def impute_test(train, test):
    # using train data to impute the missing test entries
    test.loc[test.Age.isna(), 'Age'] = train.Age.median()
    test.loc[test.Fare.isna(), 'Fare'] = train.Fare.median()

    return test


def process_fold(trn_fold, val_fold):
    # impute only with the training data of the fold
    trn_fold.loc[trn_fold.Age.isna(), 'Age'] = trn_fold.Age.median()
    trn_fold.loc[trn_fold.Embarked.isna(), 'Embarked'] =trn_fold.Embarked.mode().values[0]
    
    val_fold.loc[val_fold.Age.isna(), 'Age'] = trn_fold.Age.median()
    val_fold.loc[val_fold.Embarked.isna(), 'Embarked'] = trn_fold.Embarked.mode().values[0]
    
    trn_fold = pd.get_dummies(trn_fold)
    val_fold = pd.get_dummies(val_fold)

    return trn_fold, val_fold


def plot_importance(feature_importance_df, save_name):
    cols = (feature_importance_df[["feature", "importance"]]
                    .groupby("feature")
                    .mean().abs()
                    .sort_values(by="importance", ascending=False)[:50].index)

    best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

    plt.figure(figsize=(15,8))

    sns.barplot(x="importance",
                y="feature",
                data=best_features.sort_values(by="importance",
                                               ascending=False))

    if not save_name.endswith('.png'):
        save_name += '.png'
    plt.savefig('plots/' + save_name)
    plt.close()


def report_oof(df_train, oof):
    acc = accuracy_score(oof, df_train.Survived)
    f1 = f1_score(oof, df_train.Survived)
    roc = roc_auc_score(oof, df_train.Survived)
    print(f'Oof accuracy: \t {acc}')
    print(f'Oof f1 score: \t {f1}')
    print(f'Oof area under the roc curve: \t {roc}')
    print('Classification report: ')
    print(classification_report(oof, df_train.Survived))


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

    test = pd.get_dummies(test)

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

        oof[val_idx] = clf.predict(val_data)
        predictions += clf.predict_proba(test)[:,1] / kfolds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = trn_data.columns
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    report_oof(df_train, oof)

    plot_importance(feature_importance_df, 'rf_fe_featimp')

    sub['Survived'] = (predictions > 0.5).astype(int) 

    sub.to_csv('submissions/rf_feat_eng.csv', index=False)



def main():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    kfolds = KFold(n_splits=5, shuffle=True, random_state=498)

    train_rf(df_train, df_test, kfolds)


if __name__ == '__main__':
    main()
