__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin



class transformation(TransformerMixin, BaseEstimator):
    def __init__(self, mean_weight=10):
        self.columns = None
        self.mean_weight = mean_weight
        self.smooth_team = {}
        
    def fit(self, X, y=None):
        return self
    
    
    def stats_by_play(self, data):
        avg_by_play = data.groupby(['PlayId', 
                                    'Team', 
                                    'offense_team'], as_index=False)[['PlayerHeight', 
                                                                      'PlayerWeight',
                                                                      'age',
                                                                      'S', 'A']].mean()
        spread = data.groupby(['PlayId', 
                               'Team', 
                               'offense_team'])[['X', 'Y']].std().reset_index()
        tot_momentum = data.groupby(['PlayId', 
                                     'Team', 
                                     'offense_team'], as_index=False)[['X_speed', 'Y_speed',
                                                                       'PlayerWeight',
                                                                       'X_acceleration','Y_acceleration']].sum()
        
        tot_momentum['x_momentum'] = tot_momentum['X_speed'] * tot_momentum['PlayerWeight']
        tot_momentum['y_momentum'] = tot_momentum['Y_speed'] * tot_momentum['PlayerWeight']
        tot_momentum['x_force'] = tot_momentum['X_acceleration'] * tot_momentum['PlayerWeight']
        tot_momentum['y_force'] = tot_momentum['Y_acceleration'] * tot_momentum['PlayerWeight']
        tot_momentum.drop(['X_speed', 'Y_speed',
                           'PlayerWeight',  
                           'X_acceleration','Y_acceleration'], axis=1, inplace=True)

        avg_by_play = pd.merge(avg_by_play, tot_momentum, on=['PlayId', 'Team', 'offense_team'])
        avg_by_play = pd.merge(avg_by_play, spread, on=['PlayId', 'Team', 'offense_team'])

        poss_team = avg_by_play[avg_by_play.Team == avg_by_play.offense_team].copy()
        def_team = avg_by_play[avg_by_play.Team != avg_by_play.offense_team].copy()

        poss_team.rename(columns={'PlayerHeight': 'poss_avg_height', 
                                  'PlayerWeight': 'poss_avg_weight',
                                  'age': 'poss_avg_age',
                                  'X': 'poss_std_X',
                                  'Y': 'poss_std_Y',
                                  'S': 'poss_avg_S', 
                                  'A': 'poss_avg_A', 
                                  'x_momentum': 'poss_x_momentum', 
                                  'y_momentum': 'poss_y_momentum', 
                                  'x_force': 'poss_x_force', 
                                  'y_force': 'poss_y_force'}, inplace=True)
        def_team.rename(columns={'PlayerHeight': 'def_avg_height', 
                                  'PlayerWeight': 'def_avg_weight', 
                                  'age': 'def_avg_age',
                                  'X': 'def_std_X',
                                  'Y': 'def_std_Y',
                                  'S': 'def_avg_S', 
                                  'A': 'def_avg_A',
                                  'x_momentum': 'def_x_momentum', 
                                  'y_momentum': 'def_y_momentum', 
                                  'x_force': 'def_x_force', 
                                  'y_force': 'def_y_force'}, inplace=True)

        avg_by_play = pd.merge(poss_team.drop('Team', axis=1), 
                               def_team.drop('Team', axis=1), on=['PlayId', 'offense_team'])
        
        avg_by_play['tot_x_momenumt'] = avg_by_play['poss_x_momentum'] - avg_by_play['def_x_momentum']
        avg_by_play['tot_x_force'] = avg_by_play['poss_x_force'] - avg_by_play['def_x_force']
        avg_by_play['height_diff'] = avg_by_play['poss_avg_height'] - avg_by_play['def_avg_height']
        avg_by_play['weight_diff'] = avg_by_play['poss_avg_weight'] - avg_by_play['def_avg_weight']
        avg_by_play['age_diff'] = avg_by_play['poss_avg_age'] - avg_by_play['def_avg_age']
        avg_by_play['X_diff'] = avg_by_play['poss_std_X'] - avg_by_play['def_std_X']
        avg_by_play['Y_diff'] = avg_by_play['poss_std_Y'] - avg_by_play['def_std_Y']

        return avg_by_play
    
    
    def process_play(self, X):
        cols_by_play = ['GameId', 'PlayId', 'YardLine', 
                'Quarter', 'GameClock', 'Down', 'Distance',
                'OffenseFormation', 'DefendersInTheBox',  
                'Location', 'StadiumType', 'Turf', 
                'GameWeather','Temperature', 'Humidity', 'WindSpeed', 'WindDirection', 
                'PlayDirection', 'HomeScoreBeforePlay', 'VisitorScoreBeforePlay']
        train_play = X[cols_by_play].drop_duplicates()
        avg_by_play = self.stats_by_play(X)
        train_play = pd.merge(train_play, avg_by_play.drop('offense_team', axis=1), on=['PlayId'])

        return train_play
    
    
    def transform(self, X, y=None):
        train_play = self.process_play(X)
        carriers = X[X.has_ball].copy()

        to_drop = ['GameId', 'NflId', 'Team', 'Orientation','YardLine', 'Quarter', 'GameClock', 'PossessionTeam',
           'Down', 'FieldPosition', 'HomeScoreBeforePlay',
           'VisitorScoreBeforePlay', 'NflIdRusher', 'OffensePersonnel','DefensePersonnel',
               'PlayDirection', 'Position', 'HomeTeamAbbr',
           'VisitorTeamAbbr', 'Location', 'StadiumType', 'GameWeather',
           'Temperature', 'Humidity', 'WindSpeed', 'WindDirection', 'to_left',
           'has_ball', 'offense_team', 'Distance',
           'OffenseFormation', 'DefendersInTheBox', 'Turf']

        carriers.drop(to_drop, axis=1, inplace=True)

        full_train = pd.merge(carriers, train_play, on='PlayId')

        full_train.drop(['GameId', 'WindDirection', 'WindSpeed', 'GameWeather', 
                         'PlayDirection', 'StadiumType', 'Turf', 'Location', 'GameClock'], axis=1, inplace=True)
        
        self.columns = full_train.columns

        return full_train
    
    
    def get_features_name(self):
        return self.columns
