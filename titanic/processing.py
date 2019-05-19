__author__ = 'lucabasa'
__version__ = '1.0.1'
__status__ = 'development'


import numpy as np
import pandas as pd 


def clean_cols(data, col_list):
    df = data.copy()
    for col in col_list:
        try:
            del df[col]
        except KeyError:
            pass

    return df


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

