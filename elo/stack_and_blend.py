__author__ = 'lucabasa'
__version__ = '1.0'
__status__ = 'development'

import numpy as np 
import pandas as pd 

from utilities import read_data
import feature_eng as fe
import feature_selection as fs
import model_selection as ms

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error



agg_loc = 'processed_data/'

agg_name = 'total_aggregation_with_FE_0219.csv'

save_loc = 'results/stack_n_blend/'


model_list = {'lightGBM': ms.lightgbm_train, 
                'XGB': ms.xgb_train,
                'lightGMBrf': ms.lightgbm_rf,
                'RF': ms.rf_train,  
                'extra': ms.extratrees_train}

sel_list = {'only_hist': fs.sel_hist, 
            'only_new': fs.sel_new, 
            'only_money': fs.sel_money, 
            'only_counts': fs.sel_counts, 
            'no_money': fs.sel_nomoney, 
            'full': fs.sel_all}


def stack():
    train = pd.read_csv('results/stack_n_blend/oof_predictions.csv')
    del train['Unnamed: 0']
    test = pd.read_csv('results/stack_n_blend/all_predictions.csv')

    target = train['target']
    id_to_sub = test.card_id

    kfolds = KFold(5, shuffle=True, random_state=42)

    
    predictions, cv_score, feat_imp, oof = ms.rf_train(train, test, target, kfolds)

    print(f'random forest:\t {cv_score}')

    sub_df = pd.DataFrame({"card_id":id_to_sub.values})
    sub_df['target'] = predictions
    sub_df.to_csv('stack_rf.csv', index=False)
    feat_imp.to_csv('stack_rf_featimp.csv', index=False)

    predictions, cv_score, feat_imp, oof = ms.extratrees_train(train, test, target, kfolds)

    print(f'Extra trees:\t {cv_score}')

    sub_df = pd.DataFrame({"card_id":id_to_sub.values})
    sub_df['target'] = predictions
    sub_df.to_csv('stack_extratrees.csv', index=False)
    feat_imp.to_csv('stack_extratrees_featimp.csv', index=False)
    
    predictions, cv_score, feat_imp, oof = ms.lightgbm_train(train, test, target, kfolds)

    print(f'lightGBM:\t {cv_score}')

    sub_df = pd.DataFrame({"card_id":id_to_sub.values})
    sub_df['target'] = predictions
    sub_df.to_csv(save_loc + 'stack_lightgbm.csv', index=False)
    feat_imp.to_csv(save_loc + 'stack_lightgbm_featimp.csv', index=False)


def blend():
    train = pd.read_csv('results/stack_n_blend/oof_predictions.csv')
    del train['Unnamed: 0']
    test = pd.read_csv('results/stack_n_blend/all_predictions.csv')

    target = train['target']
    id_to_sub = test.card_id

    kfolds = KFold(5, shuffle=True, random_state=42)

    del train['target']

    train['oof_score'] = train.mean(axis=1)

    print('Full blend: ', mean_squared_error(train.oof_score, target)**0.5)

    del train['oof_score']

    scores = pd.read_csv('results/stack_n_blend/single_cvscores.csv')
    scores = scores.rename(columns={'Unnamed: 0': 'models'})

    for num in np.arange(1, 15):
        best_blends = scores.sort_values(by='CV_score').head(num).models.values

        train['oof_score'] = train[best_blends].mean(axis=1)

        print(f'Best {num} blends: ', mean_squared_error(train.oof_score, target)**0.5)

        del train['oof_score']

    

    tot_score = scores.CV_score.sum()

    for model in scores.models.unique():
        train[model] = train[model] * (scores[scores.models == model].CV_score.values[0] / tot_score)

    train['oof_score'] = train.sum(axis=1)

    print('Weighted blend: ', mean_squared_error(train.oof_score, target)**0.5)



def single_model():
    train = read_data('raw_data/train.csv')
    test = read_data('raw_data/test.csv')

    df_tr = pd.read_csv(agg_loc + agg_name)
    train = pd.merge(train, df_tr, on='card_id', how='left').fillna(0)
    test = pd.merge(test, df_tr, on='card_id', how='left').fillna(0)

    del df_tr

    train = fe.combine_categs(train)
    test = fe.combine_categs(test)

    kfolds = KFold(5, shuffle=True, random_state=42)

    results = {}

    for_second_level = pd.DataFrame({'target': train['target']})

    for model in model_list.keys():
        to_train = model_list.get(model)



        for selection in sel_list:
            to_select = sel_list.get(selection)

            print(f'{model}_{selection}')

            df_train = train.copy()
            df_test = test.copy()

            target = df_train['target']
            id_to_sub = df_test['card_id']
            del df_train['target']
            del df_train['card_id']
            del df_test['card_id']

            df_train, df_test = to_select(df_train, df_test)

            predictions, cv_score, feat_imp, oof = to_train(df_train, df_test, target, kfolds)

            results[model + '_' + selection] = cv_score

            for_second_level[model + '_' + selection] = oof

            sub_df = pd.DataFrame({"card_id":id_to_sub.values})
            sub_df["target"] = predictions
            sub_df.to_csv(save_loc + model + '_' + selection + '.csv', index=False)
            feat_imp.to_csv(save_loc + model + '_' + selection + "_featimp.csv", index=False)

            for_second_level.to_csv(save_loc + 'oof_predictions.csv')

            print(f'{model}_{selection}:\t {cv_score}')
            print('_'*40)
            print('_'*40)
            print('\n')

    final = pd.DataFrame.from_dict(results, orient='index', columns=['CV_score'])
    final.to_csv(save_loc + 'single_cvscores.csv')
    for_second_level.to_csv(save_loc + 'oof_predictions.csv')


def stack_with_features():
    train = read_data('raw_data/train.csv')
    test = read_data('raw_data/test.csv')

    df_tr = pd.read_csv(agg_loc + agg_name)
    train = pd.merge(train, df_tr, on='card_id', how='left').fillna(0)
    test = pd.merge(test, df_tr, on='card_id', how='left').fillna(0)

    del df_tr

    train = fe.combine_categs(train)
    test = fe.combine_categs(test)

    train = train[['card_id', 'target'] + [col for col in train.columns if 'purchase' in col or 'month' in col]]
    test = test[['card_id'] + [col for col in train.columns if 'purchase' in col or 'month' in col]]

    print(train.columns)

    stacked = pd.read_csv('results/stack_n_blend/oof_predictions.csv')
    del stacked['Unnamed: 0']
    del stacked['target']
    st_test = pd.read_csv('results/stack_n_blend/all_predictions.csv')

    #stacked = stacked[[col for col in stacked.columns if 'lightGBM_' in col]]
    #st_test = st_test[[col for col in stacked.columns if 'lightGBM_' in col] + ['card_id']]

    train = pd.concat([train, stacked], axis=1)
    test = pd.merge(test, st_test, on='card_id', how='left')

    del train['lightGBM_full']
    del test['lightGBM_full']

    target = train['target']
    id_to_sub = test.card_id
    del train['target']
    del train['card_id']
    del test['card_id']

    kfolds = KFold(10, shuffle=True, random_state=42)

    predictions, cv_score, feat_imp, oof = ms.lightgbm_train(train, test, target, kfolds)

    sub_df = pd.DataFrame({"card_id":id_to_sub.values})
    sub_df["target"] = predictions
    sub_df.to_csv(save_loc + 'stacked_with_feats.csv', index=False)
    feat_imp.to_csv(save_loc + "stacked_with_feats_featimp.csv", index=False)

    print(cv_score)


if __name__=="__main__":
    single_model()
    stack()
    blend()
    stack_with_features()
