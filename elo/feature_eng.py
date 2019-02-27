__author__ = 'lucabasa'
__version__ = 1.0
__status__ = "development"

import pandas as pd 
import numpy as np

def combine_categs(data):
    result = data.copy()

    result['feat1_hist_sum'] = result['feature_1'] * result['tot_purchase_amount_sum']
    result['feat2_hist_sum'] = result['feature_2'] * result['tot_purchase_amount_sum']
    result['feat3_hist_sum'] = result['feature_3'] * result['tot_purchase_amount_sum']

    result['feat1_hist_mean'] = result['feature_1'] * result['new_purchase_amount_mean']
    result['feat2_hist_mean'] = result['feature_2'] * result['new_purchase_amount_mean']
    result['feat3_hist_mean'] = result['feature_3'] * result['new_purchase_amount_mean']

    result['feat1_new_sum'] = result['feature_1'] * result['new_purchase_amount_sum']
    result['feat2_new_sum'] = result['feature_2'] * result['new_purchase_amount_sum']
    result['feat3_new_sum'] = result['feature_3'] * result['new_purchase_amount_sum']

    result['feat1_new_mean'] = result['feature_1'] * result['new_purchase_amount_mean']
    result['feat2_new_mean'] = result['feature_2'] * result['new_purchase_amount_mean']
    result['feat3_new_mean'] = result['feature_3'] * result['new_purchase_amount_mean']

    return result


def remove_2m(data):
    result = data.copy()

    to_drop = [col for col in result.columns if col.startswith('h2_')]
    to_drop += [col for col in result.columns if col.startswith('h4_')]
    to_drop += [col for col in result.columns if col.startswith('h6_')]
    to_drop += [col for col in result.columns if col.startswith('h8_')]
    to_drop += [col for col in result.columns if col.startswith('h10_')]
    to_drop += [col for col in result.columns if col.startswith('h12_')]

    for col in to_drop:
        del result[col]

    return result


def make_spread(data):
    to_use = [col for col in data.columns if col.endswith('_min')]
    result = data.copy()

    for col in to_use:
        name = col.split('_min')[0]
        result[name+'_spread'] = result[name+'_max'] - result[col]

    return result


def make_trend(data):
    cols = [col.split('tot_')[1] for col in data.columns if 'tot_' in col] 
    result = data.copy()

    for col in cols:
        try:
            # creating the time spread features
            result['2_4_'+col] = result['h2_'+col] - result['h4_'+col]
            result['4_6_'+col] = result['h4_'+col] - result['h6_'+col]
            result['6_8_'+col] = result['h6_'+col] - result['h8_'+col]
            result['8_10_'+col] = result['h8_'+col] - result['h10_'+col]
            result['10_12_'+col] = result['h10_'+col] - result['h12_'+col]
            result['time_spread_'+col] = (result['2_4_'+col] + result['4_6_'+col] + result['6_8_'+col] 
                               + result['8_10_'+col] + result['10_12_'+col]) / 2
            result['pos_trend_'+col] = (result[['2_4_'+col, '4_6_'+col, 
                                          '6_8_'+col, '8_10_'+col, '10_12_'+col]] < 0).astype(int).sum(axis=1)
            result['neg_trend_'+col] = (result[['2_4_'+col, '4_6_'+col, 
                                              '6_8_'+col, '8_10_'+col, '10_12_'+col]] > 0).astype(int).sum(axis=1)
            result['tot_trend_'+col] = result['pos_trend_'+col] + result['neg_trend_'+col]
            del result['2_4_'+col]
            del result['4_6_'+col]
            del result['6_8_'+col]
            del result['8_10_'+col]
            del result['10_12_'+col]
        except KeyError:
            pass

    return result



def make_fractions(data):
    cols = [col.split('tot_')[1] for col in data.columns if 'tot_' in col] 
    result = data.copy()

    for col in cols:
        try:
            result['t_v_n_'+col] = result['new_'+col] / result['tot_'+col]
            result['t_v_n_'+col] = result['t_v_n_'+col].replace(np.inf, 0)
            result['t_v_n_'+col] = result['t_v_n_'+col].replace(-np.inf, 0)
        except KeyError:
            pass

    return result


def make_differences(data):
    cols = [col.split('tot_')[1] for col in data.columns if 'tot_' in col] 
    result = data.copy()

    for col in cols:
        try:
            result['t_diff_n_'+col] = result['tot_'+col] - result['new_'+col]
        except KeyError:
            pass

    return result


def trends_no2m(data):
    result = data.copy()
    result = make_trend(result)
    result = remove_2m(result)
    return result

def spread_no2m(data):
    result = data.copy()
    result = make_spread(result)
    result = remove_2m(result)
    return result


def remove_max_min(data):
    result = data.copy()
    to_drop = [col for col in data.columns if col.endswith('_max')]
    to_drop += [col for col in data.columns if col.endswith('_min')]
    for col in to_drop:
        del result[col]
    return result


def make_train_feats(data):
    result = data.copy()

    result['days_feature1'] = result['elapsed_time'] * result['feature_1']
    result['days_feature2'] = result['elapsed_time'] * result['feature_2']
    result['days_feature3'] = result['elapsed_time'] * result['feature_3']

    result['days_feature1_ratio'] = result['feature_1'] / result['elapsed_time']
    result['days_feature2_ratio'] = result['feature_2'] / result['elapsed_time']
    result['days_feature3_ratio'] = result['feature_3'] / result['elapsed_time']

    result['feature_sum'] = result['feature_1'] + result['feature_2'] + result['feature_3']
    result['feature_mean'] = result['feature_sum']/3

    return result


def clv(data):
    result = data.copy()

    #result['new_CLV'] = result['new_authorized_flag_count'] * result['new_purchase_amount_sum'] / result['new_month_diff_mean']
    result['hist_CLV'] = result['tot_authorized_flag_count'] * result['tot_purchase_amount_sum'] / result['tot_month_diff_mean']
    #result['CLV_ratio'] = result['new_CLV'] / result['hist_CLV']

    return result
