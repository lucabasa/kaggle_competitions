__author__ = 'lucabasa'
__version__ = '1.0.2'
__status__ = 'development'


import numpy as np
import pandas as pd 

from sklearn.model_selection import KFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

import processing as pr
import report as rep


def tune_logit(train, target, verbose=False):
    grid_param = {'logit__C': np.arange(0.01, 2, 0.01)}

    kfolds = KFold(5, shuffle=True, random_state=14)
    
    scl = ('scl', RobustScaler())
    pipe = Pipeline([scl, ('logit', LogisticRegression(solver='lbfgs', max_iter=5000))])

    grid = GridSearchCV(pipe, param_grid=grid_param, n_jobs=-1, 
                                cv=5, scoring='roc_auc')

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
    to_drop = ['Survived', 'Name', 'Ticket', 'PassengerId', 'Cabin',  # the next line is for choices specific to logit
    			'Embarked', 'mis_age', 'Parch', 'SibSp', 'fam_size', 'Pclass']
    train = pr.clean_cols(train, to_drop)
    test = pr.clean_cols(test, to_drop)
    
    return train, test

def impute_test(train, test):
    # using train data to impute the missing test entries
    test.loc[test.Age.isna(), 'Age'] = train.Age.median()
    test.loc[test.Fare.isna(), 'Fare'] = train.Fare.median()

    test = pr.baby(test)
    test = pr.young(test)

    del test['Age']  # the oof showed a clear pattern in the prediction, I prefer to drop it until I know why

    test = pd.get_dummies(test, drop_first=True)

    return test


def process_fold(trn_fold, val_fold):
    # impute only with the training data of the fold
    trn_fold.loc[trn_fold.Age.isna(), 'Age'] = trn_fold.Age.median()
    
    val_fold.loc[val_fold.Age.isna(), 'Age'] = trn_fold.Age.median()

    trn_fold = pr.baby(trn_fold)
    val_fold = pr.baby(val_fold)

    trn_fold = pr.young(trn_fold)
    val_fold = pr.young(val_fold)

    del trn_fold['Age']
    del val_fold['Age']

    trn_fold = pd.get_dummies(trn_fold, drop_first=True)
    val_fold = pd.get_dummies(val_fold, drop_first=True)

    return trn_fold, val_fold


def train_logit(df_train, df_test, kfolds):
    train = df_train.copy()
    test = df_test.copy()

    target = train.Survived.copy()

    sub = test[['PassengerId']].copy()

    train, test = general_processing(train, test)

    # model
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    feature_importance_df = pd.DataFrame()
    
    scl = ('scl', RobustScaler())
    
    test = impute_test(train, test)
    
    train['Fare'] = np.log1p(train.Fare)  # linear model, very skewed feature
    test['Fare'] = np.log1p(test.Fare)
    
    for fold_, (trn_idx, val_idx) in enumerate(kfolds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))

        trn_data = train.iloc[trn_idx].copy()
        val_data = train.iloc[val_idx].copy()

        trn_target = target.iloc[trn_idx]
        val_target = target.iloc[val_idx]
        
        trn_data, val_data = process_fold(trn_data, val_data)
        
        params = tune_logit(trn_data, trn_target, verbose=True)
        pipe = Pipeline([scl, ('logit', LogisticRegression(solver='lbfgs', max_iter=5000,
                                                          C=params['logit__C']))])

        clf = pipe.fit(trn_data, trn_target)

        oof[val_idx] = clf.predict_proba(val_data)[:,1]
        predictions += clf.predict_proba(test)[:,1] / kfolds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = trn_data.columns
        fold_importance_df["coefficient"] = clf.steps[-1][1].coef_[0]
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


    rep.report_oof(df_train, (oof > 0.5).astype(int))

    rep.plot_importance(feature_importance_df, 'logit_fe_featimp', 'coefficient')

    sub['Survived'] = (predictions > 0.5).astype(int)

    sub.to_csv('submissions/logit_fe.csv', index=False)

    rf_oof = df_train.copy()
    rf_oof['oof'] = oof
    rf_oof.to_csv('oof_pred/logistic_regression.csv', index=False)
    sub.to_csv('oof_pred/logistic_regression_test.csv', index=False)


def main():
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    kfolds = KFold(n_splits=5, shuffle=True, random_state=498)

    train_logit(df_train, df_test, kfolds)


if __name__ == '__main__':
    main()
