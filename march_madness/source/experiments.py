__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'

import numpy as np 
import pandas as pd 

from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import KFold

from models import rf_train, extra_train, lgb_train

from tune_models import tune_rf, tune_extra


save_loc = 'model_results/'
read_loc = 'processed_data/'

model_list = {#'rf' : rf_train,
               # 'extra': extra_train,
                'lgbm': lgb_train}

feat_list = {'all': 1}


def exp_prob(df_train, df_test, tune=False):

    for model in model_list.keys():
        print(model)
        trainer = model_list.get(model)

        for feat_sel in feat_list.keys():
            selector = feat_list.get(feat_sel)

            # select stuff

            if tune:
                print('Tuning...')
                to_tune = df_train.sample(10000).copy()
                tune_target = to_tune.target
                to_drop = ['target','target_points','Team1','Team2', 'ID', 
                            'DayNum', 'Season', 'predicted_prob']
                for col in to_drop:
                    try:
                        del to_tune[col]
                    except KeyError:
                        pass

                #params = tune_rf(to_tune, tune_target, verbose=True)
                params = tune_extra(to_tune, tune_target, verbose=True)

            else:
                params = False


            sss = []
            cv_score = []
            test_score = []
            full_score = []

            for season in df_train.Season.unique():
                print(season)
                train = df_train[df_train.Season == season].copy()
                test = df_test[df_test.Season == season].copy()

                target = train['target']
                test_target = test['target']

                to_drop = ['target','target_points','Team1','Team2', 'ID', 
                            'DayNum', 'Season', 'predicted_prob']

                for col in to_drop:
                    try:
                        del train[col]
                        del test[col]
                    except KeyError:
                        pass
                    
                kfolds = KFold(5, shuffle=True, random_state=984)

                season_oof, season_pred, feat_imp, oof_score, all_test_score = trainer(train, test, target, 
                                                                                kfolds, df_test, tune=params)

                print("Season {} \t CV score: {:<8.5f}".format(season, oof_score))
                print('\t Prediction score: {}'.format(log_loss(test_target, season_pred)))
                print('\t Full prediction: {}'.format(all_test_score))
                
                df_test.loc[df_test.Season == season, 'predicted_prob'] = season_pred
                df_train.loc[df_train.Season == season, 'predicted_prob'] = season_oof
                
                sss.append(season)
                cv_score.append(oof_score)
                test_score.append(log_loss(test_target, season_pred))
                full_score.append(all_test_score)
                
                results = pd.DataFrame({'season': sss, 'cv_score': cv_score, 
                                        'test_score': test_score, 'full_score': full_score})
                results.to_csv(save_loc + model + '_' + feat_sel + '_prob_scores.csv', index=False)
                feat_imp.to_csv(save_loc + 'feat_imp/'+ model + '_' + feat_sel + '_' + str(season) + '_prob_FI.csv', index=False)
                
            print('Final CV score: {}'.format(log_loss(df_train['target'], df_train['predicted_prob'])))
            print('Final prediction score: {}'.format(log_loss(df_test['target'], df_test['predicted_prob'])))

            df_train.to_csv(save_loc + 'oof_df/' + model + '_' + feat_sel + '_prob_oof.csv', index=False)
            df_test.to_csv(save_loc + 'pred_df/' + model + '_' + feat_sel + '_prob_pred.csv', index=False)


if __name__=='__main__':
    #df_train = pd.read_csv(read_loc + 'training_reg_full.csv')
    df_test = pd.read_csv(read_loc + 'testing_playoff.csv')

    exp_prob(df_train, df_test, tune=False)

