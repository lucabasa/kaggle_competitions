__author__ = 'lucabasa'
__version__ = '1.2.0'
__status__ = 'development'


import pandas as pd 
import numpy as np 


def make_teams_target(data, league):
    if league == 'men':
        limit = 2003
    else:
        limit = 2010

    df = data[data.Season >= limit].copy()

    df['Team1'] = np.where((df.WTeamID < df.LTeamID), df.WTeamID, df.LTeamID)
    df['Team2'] = np.where((df.WTeamID > df.LTeamID), df.WTeamID, df.LTeamID)
    df['target'] = np.where((df['WTeamID'] < df['LTeamID']),1,0)
    df['target_points'] = np.where((df['WTeamID'] < df['LTeamID']),df.WScore - df.LScore,df.LScore - df.WScore)
    df.loc[df.WLoc == 'N', 'LLoc'] = 'N'
    df.loc[df.WLoc == 'H', 'LLoc'] = 'A'
    df.loc[df.WLoc == 'A', 'LLoc'] = 'H'
    df['T1_Loc'] = np.where((df.WTeamID < df.LTeamID), df.WLoc, df.LLoc)
    df['T2_Loc'] = np.where((df.WTeamID > df.LTeamID), df.WLoc, df.LLoc)
    df['T1_Loc'] = df['T1_Loc'].map({'H': 1, 'A': -1, 'N': 0})
    df['T2_Loc'] = df['T2_Loc'].map({'H': 1, 'A': -1, 'N': 0})

    reverse = data[data.Season >= limit].copy()
    reverse['Team1'] = np.where((reverse.WTeamID > reverse.LTeamID), reverse.WTeamID, reverse.LTeamID)
    reverse['Team2'] = np.where((reverse.WTeamID < reverse.LTeamID), reverse.WTeamID, reverse.LTeamID)
    reverse['target'] = np.where((reverse['WTeamID'] > reverse['LTeamID']),1,0)
    reverse['target_points'] = np.where((reverse['WTeamID'] > reverse['LTeamID']),
                                        reverse.WScore - reverse.LScore,
                                        reverse.LScore - reverse.WScore)
    reverse.loc[reverse.WLoc == 'N', 'LLoc'] = 'N'
    reverse.loc[reverse.WLoc == 'H', 'LLoc'] = 'A'
    reverse.loc[reverse.WLoc == 'A', 'LLoc'] = 'H'
    reverse['T1_Loc'] = np.where((reverse.WTeamID > reverse.LTeamID), reverse.WLoc, reverse.LLoc)
    reverse['T2_Loc'] = np.where((reverse.WTeamID < reverse.LTeamID), reverse.WLoc, reverse.LLoc)
    reverse['T1_Loc'] = reverse['T1_Loc'].map({'H': 1, 'A': -1, 'N': 0})
    reverse['T2_Loc'] = reverse['T2_Loc'].map({'H': 1, 'A': -1, 'N': 0})
    
    df = pd.concat([df, reverse], ignore_index=True)

    to_drop = ['WScore','WTeamID', 'LTeamID', 'LScore', 'WLoc', 'LLoc', 'NumOT']
    for col in to_drop:
        del df[col]
    
    df.loc[:,'ID'] = df.Season.astype(str) + '_' + df.Team1.astype(str) + '_' + df.Team2.astype(str)
    return df


def make_training_regular(reg, origins, save_loc):
#    for key in origins.keys():
#        print(key)
    key = 'reg_full'
    details = pd.read_csv(save_loc + origins[key])

    tmp = details.copy()
    tmp.columns = ['Season', 'DayNum', 'Team1'] + \
                ['T1_'+col for col in tmp.columns if col not in ['Season', 'DayNum', 'TeamID']]
    total = pd.merge(reg, tmp, on=['Season', 'DayNum', 'Team1'], how='left')

    tmp = details.copy()
    tmp.columns = ['Season', 'DayNum', 'Team2'] + \
                ['T2_'+col for col in tmp.columns if col not in ['Season', 'DayNum', 'TeamID']]
    total = pd.merge(total, tmp, on=['Season', 'DayNum', 'Team2'], how='left')

    total = total.rename(columns={'T1_Loc_x': 'T1_Loc', 'T2_Loc_x': 'T2_Loc', 
                                    'T1_Loc_y': 'T1_Loc_mean', 'T2_Loc_y': 'T2_Loc_mean'})

    if total.isnull().any().any():
        if save_loc.endswith('/men/'):
            raise ValueError('Something went wrong')
        else:
            print('Some games are missing')
            before = total.shape[0]
            total = total.dropna()
            print(f'{before - total.shape[0]} games dropped')

    total.to_csv(save_loc + 'training_' + key + '.csv', index=False)


def _add_rank(total):
    ranks = pd.read_csv('raw_data/mens-machine-learning-competition-2019/MasseyOrdinals.csv')
    ranks = ranks[['Season', 'RankingDayNum', 
           'TeamID', 'OrdinalRank']].groupby(['Season', 'RankingDayNum','TeamID']).mean().reset_index()
    ranks = ranks[ranks.RankingDayNum == 133]
    ranks = ranks.rename(columns={'TeamID': 'Team1'})
    total = pd.merge(total, ranks[['Season', 'Team1', 'OrdinalRank']], on=['Season', 'Team1'], how='left')
    ranks = ranks.rename(columns={'Team1': 'Team2'})
    total = pd.merge(total, ranks[['Season', 'Team2', 'OrdinalRank']], on=['Season', 'Team2'], how='left')
    total = total.rename(columns={'OrdinalRank_x': 'T1_rank', 'OrdinalRank_y': 'T2_rank'})

    return total


def _add_seed(save_loc, total):
    if save_loc.endswith('/men/'):
        seeds = pd.read_csv('raw_data/mens-machine-learning-competition-2019/DataFiles/NCAATourneySeeds.csv')
        seeds = seeds[seeds.Season >= 2003].copy()
    else:
        seeds = pd.read_csv('raw_data/womens-machine-learning-competition-2019/WDataFiles/WNCAATourneySeeds.csv')
        seeds = seeds[seeds.Season >= 2010].copy()

    seeds['seed_num'] = seeds.Seed.apply(lambda x: int(x[1:3]))
    seeds = seeds.rename(columns={'TeamID': 'Team1', 'seed_num': 'T1_seed'})
    total = pd.merge(total, seeds[['Season', 'Team1', 'T1_seed']], on=['Season', 'Team1'], how='left')
    seeds = seeds.rename(columns={'Team1': 'Team2', 'T1_seed': 'T2_seed'})
    total = pd.merge(total, seeds[['Season', 'Team2', 'T2_seed']], on=['Season', 'Team2'], how='left')
    return total


def _add_stage(save_loc, total):
    #if save_loc.endswith('/men/'):
    total['stage'] = '68'
    total.loc[(total.DayNum == 136) | (total.DayNum == 136), 'stage'] = '64'
    total.loc[(total.DayNum == 138) | (total.DayNum == 139), 'stage'] = '32'
    total.loc[(total.DayNum == 143) | (total.DayNum == 144), 'stage'] = '16'
    total.loc[(total.DayNum == 145) | (total.DayNum == 146), 'stage'] = '8'
    total.loc[(total.DayNum == 152), 'stage'] = '4'
    total.loc[(total.DayNum == 154), 'stage'] = 'Final'

    total = pd.get_dummies(total, columns=['stage'])

    del total['stage_68']

    return total


def make_testing_playoff(play, origins, save_loc):
    details = pd.read_csv(save_loc + origins['play_season'])

    tmp = details.copy()
    tmp.columns = ['Season', 'Team1'] + \
                ['T1_'+col for col in tmp.columns if col not in ['Season', 'TeamID']]
    total = pd.merge(play, tmp, on=['Season', 'Team1'], how='left')

    tmp = details.copy()
    tmp.columns = ['Season', 'Team2'] + \
                ['T2_'+col for col in tmp.columns if col not in ['Season', 'TeamID']]
    total = pd.merge(total, tmp, on=['Season', 'Team2'], how='left')

    if total.isnull().any().any():
        if save_loc.endswith('/men/'):
            raise ValueError('Something went wrong')
        else:
            print('Some games are missing')
            before = total.shape[0]
            total = total.dropna()
            print(f'{before - total.shape[0]} games dropped')

    #total = total.rename(columns={'T1_Loc_x': 'T1_Loc', 'T2_Loc_x': 'T2_Loc', 
    #                                'T1_Loc_y': 'T1_Loc_mean', 'T2_Loc_y': 'T2_Loc_mean'})

    total = _add_stage(save_loc, total)

    if save_loc.endswith('/men/'):
        total = _add_rank(total)

    total = _add_seed(save_loc, total)

    #del total['T1_Loc']
    #del total['T2_Loc']

    stats = [col[3:] for col in total.columns if 'T1_' in col]

    for stat in stats:
        total['delta_'+stat] = total['T1_'+stat] - total['T2_'+stat]

    if total.isnull().any().any():
        raise ValueError('Something went wrong')

    total.to_csv(save_loc + 'testing_playoff.csv', index=False)


def make_train_test(league):
    save_loc = 'processed_data/' + league + '/'

    if league == 'women':
        raw_loc = 'raw_data/womens-machine-learning-competition-2019/WDataFiles/'

        origins = {'reg_full': 'teams_byday_reg_all_past.csv', 
                    'reg_last5': 'teams_byday_reg_last_5.csv', 
                    'reg_last10': 'teams_byday_reg_last_10.csv',
                    'play_season': 'teams_full_reg.csv'}

        results = {'reg': 'WRegularSeasonCompactResults.csv',
                    'play': 'WNCAATourneyCompactResults.csv'}
    else:
        raw_loc = 'raw_data/mens-machine-learning-competition-2019/DataFiles/'

        origins = {'reg_full': 'teams_byday_reg_all_past.csv', 
                    'reg_last5': 'teams_byday_reg_last_5.csv', 
                    'reg_last10': 'teams_byday_reg_last_10.csv',
                    'play_season': 'teams_full_reg.csv'}

        results = {'reg': 'RegularSeasonCompactResults.csv',
                    'play': 'NCAATourneyCompactResults.csv'}

    compact = pd.read_csv(raw_loc + results['reg'])
    compact = make_teams_target(compact, league)
    make_training_regular(compact, origins, save_loc)

    compact = pd.read_csv(raw_loc + results['play'])
    compact = make_teams_target(compact, league)
    make_testing_playoff(compact, origins, save_loc)


if __name__=='__main__':
    make_train_test('men')
    make_train_test('women')
    






