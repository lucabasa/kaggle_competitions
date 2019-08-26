__author__ = 'lucabasa'
__version__ = '1.0.1'
__status__ = 'development'


import numpy as np
import pandas as pd 


def flag_missing(data, col_list):
    df = data.copy()
    for col in col_list:
        df['mis_' + col.lower()] = 0
        df.loc[df[col].isna(), 'mis_' + col.lower()] = 1

    return df


def gen_clas(data):
    df = data.copy()

    df.loc[(df.Sex == 1) & (df.Pclass == 1), 'se_cl'] = 'male_1'
    df.loc[(df.Sex == 1) & (df.Pclass == 2), 'se_cl'] = 'male_23' # to help with the misclassification of men
    df.loc[(df.Sex == 1) & (df.Pclass == 3), 'se_cl'] = 'male_23'
    df.loc[(df.Sex == 0) & (df.Pclass == 1), 'se_cl'] = 'female_1'
    df.loc[(df.Sex == 0) & (df.Pclass == 2), 'se_cl'] = 'female_2'
    df.loc[(df.Sex == 0) & (df.Pclass == 3), 'se_cl'] = 'female_3'

    return df


def gen_cab(data):
    df = data.copy()

    df.loc[((df.Sex == 1) & (df.mis_cabin == 0)) , 'se_ca'] = 'male_nocab'
    df.loc[((df.Sex == 1) & (df.mis_cabin == 1)) , 'se_ca'] = 'male_cab'
    df.loc[((df.Sex == 0) & (df.mis_cabin == 0)) , 'se_ca'] = 'female_nocab'
    df.loc[((df.Sex == 0) & (df.mis_cabin == 1)) , 'se_ca'] = 'female_cab'

    return df


def general_processing(train, test):
    # processing train and test outside the cv loop
    train['Sex'] = train.Sex.map({'male': 1, 'female': 0}).astype(int)
    test['Sex'] = test.Sex.map({'male': 1, 'female': 0}).astype(int)

    # flagging missing data
    train = flag_missing(train, ['Age', 'Cabin'])
    test = flag_missing(test, ['Age', 'Cabin'])

    # fam size
    train['fam_size'] = train['SibSp'] + train['Parch'] + 1
    test['fam_size'] = test['SibSp'] + test['Parch'] + 1

    # Gender and class
    train = gen_clas(train)
    test = gen_clas(test)

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
    train = gen_cab(train)
    test = gen_cab(test)

    return train, test


def clean_cols(data, col_list):
    df = data.copy()
    for col in col_list:
        try:
            del df[col]
        except KeyError:
            pass

    return df


def baby(data):
    df = data.copy()

    df['is_baby'] = 0
    df.loc[df.Age < 10, 'is_baby'] = 1

    return df

def young(data):
    df = data.copy()

    df['is_young'] = 0
    df.loc[(df.Age > 9) & (df.Age < 20), 'is_young'] = 1

    return df

