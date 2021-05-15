__author__ = 'lucabasa'
__version__ = '1.0.0'
__status__ = 'development'


import pandas as pd 
import numpy as np

import statsmodels.api as sm


def big_wins(data, rank_loc=None):
    '''
    If Rank is available, use it to display a win against a top team or an upset
    Either way, add win in overtime and away from home
    
    data: game stats in the W/L format (the original one)
    rank_loc: csv location (optional)
    '''
    df = data.copy()
    
    if rank_loc:
        ranks = pd.read_csv(rank_loc)
        # exclude ranks that are on very different value ranges
        ranks = ranks[~(ranks.SystemName.isin(['AP', 'USA', 'DES', 'LYN', 'ACU', 
                                               'TRX', 'D1A', 'JNG', 'BNT']))].copy()
        mean_ranks = ranks.groupby(['Season', 'TeamID', 'RankingDayNum'], as_index=False).OrdinalRank.mean()

        df = pd.merge(df, mean_ranks.rename(columns={'TeamID': 'WTeamID', 
                                                    'RankingDayNum':'DayNum', 
                                                    'OrdinalRank': 'WRank'}), 
                    on=['Season', 'WTeamID', 'DayNum'], how='left')

        df = pd.merge(df, mean_ranks.rename(columns={'TeamID': 'LTeamID', 
                                                        'RankingDayNum':'DayNum', 
                                                        'OrdinalRank': 'LRank'}), 
                        on=['Season', 'LTeamID', 'DayNum'], how='left')

        df = df.fillna(1000)

        df['Wtop_team'] = 0
        df.loc[df.LRank <= 30, 'Wtop_team'] = 1

        df['Wupset'] = 0
        df.loc[df.WRank - df.LRank > 15, 'Wupset'] = 1
        
        del df['WRank']
        del df['LRank']
    
    df['WOT_win'] = 0
    df.loc[df.NumOT > 0, 'WOT_win'] = 1
    
    df['WAway'] = 0
    df.loc[df.WLoc!='H', 'WAway'] = 1
    
    return df


def perc_OT_win(data):
    '''
    Creates a dataframe with the percentage of OT wins per team in a Season
    '''
    df = data[['Season', 'TeamID', 'NumOT', 'OT_win']].copy()
    df['has_OT'] = np.where(df.NumOT > 0, 1, 0)
    
    df = df.groupby(['Season', 'TeamID', 'has_OT'], as_index=False).OT_win.mean()
    df = df[df.has_OT > 0].copy()
    del df['has_OT']
    
    return df.rename(columns={'OT_win': 'OT_win_perc'})


def add_seed(seed_location, total):
    '''
    Read the Seed csv and add to the Seasonal stats the seed and region
    
    seed_location: csv location
    total: Seasonal stats, it must have a TeamID and a Season column
    '''
    seed_data = pd.read_csv(seed_location)
    seed_data['region'] = seed_data['Seed'].apply(lambda x: x[0])
    seed_data['Seed'] = seed_data['Seed'].apply(lambda x: int(x[1:3]))
    total = pd.merge(total, seed_data, how='left', on=['TeamID', 'Season'])
    return total


def add_rank(rank_location, total):
    '''
    Read the Rank csv, exclude some of them, aggregate the ranks by day and team
    Then take the Rank on the last regular season day and add it to the Seasonal stats
    
    rank_location: csv location
    total: Seasonal stats, it must have a TeamID and a Season column
    '''
    ranks = pd.read_csv(rank_location)
    ranks = ranks[~(ranks.SystemName.isin(['AP', 'USA', 'DES', 'LYN', 'ACU', 
                                           'TRX', 'D1A', 'JNG', 'BNT']))].copy()
    ranks = ranks.groupby(['Season', 'TeamID', 'RankingDayNum'], as_index=False).OrdinalRank.mean()
    ranks = ranks[ranks.RankingDayNum == 133]
    del ranks['RankingDayNum']

    total = pd.merge(total, ranks.rename(columns={'OrdinalRank': 'Rank'}), 
                     on=['Season', 'TeamID'], how='left')

    return total


def highlow_seed(reg_stats, game_data, seed):
    '''
    Use the seed information to determine if each win and loss was again an high or low seed
    Create then the % of wins against high and low seed per team and Season
    Merge it back to the regular season stats
    
    reg_stats: seasonal stats per season and team
    game_data: game details in the original W/L format
    seed: csv location
    '''
    tmp = pd.read_csv(seed)
    tmp['Seed'] = tmp['Seed'].apply(lambda x: int(x[1:3]))

    df = pd.merge(game_data, tmp.rename(columns={'TeamID': 'WTeamID',
                                          'Seed': 'WSeed'}), 
                        on=['Season', 'WTeamID'], how='left')
    df = pd.merge(df, tmp.rename(columns={'TeamID': 'LTeamID',
                                          'Seed': 'LSeed'}), 
                        on=['Season', 'LTeamID'], how='left')

    tmp = df[['Season', 'LTeamID', 'WTeamID', 'LSeed', 'WSeed']].copy()

    df.loc[df.LSeed <= 8, 'Whigh_seed'] = 1
    df.loc[df.WSeed <= 8, 'Lhigh_seed'] = 0
    df.loc[(df.LSeed >= 9) | (df.LSeed.isna()), 'Wlow_seed'] = 1
    df.loc[(df.WSeed >= 9) | (df.WSeed.isna()), 'Llow_seed'] = 0
    
    tmp = df[['Season', 'LTeamID', 'Lhigh_seed', 'Llow_seed']].copy()
    tmp.columns = ['Season', 'TeamID', 'high_seed', 'low_seed']
    df = df[['Season', 'WTeamID', 'Whigh_seed', 'Wlow_seed']].copy()
    df.columns = ['Season', 'TeamID', 'high_seed', 'low_seed']

    df = pd.concat([df, tmp], ignore_index=True)

    df = df.groupby(['Season', 'TeamID'], as_index=False)[['high_seed', 'low_seed']].mean()
    
    reg_stats = pd.merge(reg_stats, df, on=['Season', 'TeamID'], how='left')
    reg_stats.high_seed = reg_stats.high_seed.fillna(0)
    
    return reg_stats


def add_stage(data):
    '''
    Given Seed and region, determine the stage of a given game
    Dummify the result
    
    data: training data with Region and seed for T1 and T2
    '''
    data.loc[(data.T1_region == 'W') & (data.T2_region == 'X'), 'stage'] = 'finalfour'
    data.loc[(data.T1_region == 'X') & (data.T2_region == 'W'), 'stage'] = 'finalfour'
    data.loc[(data.T1_region == 'Y') & (data.T2_region == 'Z'), 'stage'] = 'finalfour'
    data.loc[(data.T1_region == 'Z') & (data.T2_region == 'Y'), 'stage'] = 'finalfour'
    data.loc[(data.T1_region == 'W') & (data.T2_region.isin(['Y', 'Z'])), 'stage'] = 'final'
    data.loc[(data.T1_region == 'X') & (data.T2_region.isin(['Y', 'Z'])), 'stage'] = 'final'
    data.loc[(data.T1_region == 'Y') & (data.T2_region.isin(['W', 'X'])), 'stage'] = 'final'
    data.loc[(data.T1_region == 'Z') & (data.T2_region.isin(['W', 'X'])), 'stage'] = 'final'
    data.loc[(data.T1_region == data.T2_region) & (data.T1_Seed + data.T2_Seed == 17), 'stage'] = 'Round1'
    
    fil = data.stage.isna()
    
    data.loc[fil & (data.T1_Seed.isin([1, 16])) & (data.T2_Seed.isin([8, 9])), 'stage'] = 'Round2'
    data.loc[fil & (data.T1_Seed.isin([8, 9])) & (data.T2_Seed.isin([1, 16])), 'stage'] = 'Round2'
    data.loc[fil & (data.T1_Seed.isin([5, 12])) & (data.T2_Seed.isin([4, 13])), 'stage'] = 'Round2'
    data.loc[fil & (data.T1_Seed.isin([4, 13])) & (data.T2_Seed.isin([5, 12])), 'stage'] = 'Round2'
    data.loc[fil & (data.T1_Seed.isin([6, 11])) & (data.T2_Seed.isin([3, 14])), 'stage'] = 'Round2'
    data.loc[fil & (data.T1_Seed.isin([3, 14])) & (data.T2_Seed.isin([6, 11])), 'stage'] = 'Round2'
    data.loc[fil & (data.T1_Seed.isin([7, 10])) & (data.T2_Seed.isin([2, 15])), 'stage'] = 'Round2'
    data.loc[fil & (data.T1_Seed.isin([2, 15])) & (data.T2_Seed.isin([7, 10])), 'stage'] = 'Round2'
    
    fil = data.stage.isna()
    
    data.loc[fil & (data.T1_Seed.isin([1, 16, 8, 9])) & (data.T2_Seed.isin([4, 5, 12, 13])), 'stage'] = 'Round3'
    data.loc[fil & (data.T1_Seed.isin([4, 5, 12, 13])) & (data.T2_Seed.isin([1, 16, 8, 9])), 'stage'] = 'Round3'
    data.loc[fil & (data.T1_Seed.isin([3, 6, 11, 14])) & (data.T2_Seed.isin([2, 7, 10, 15])), 'stage'] = 'Round3'
    data.loc[fil & (data.T1_Seed.isin([2, 7, 10, 15])) & (data.T2_Seed.isin([3, 6, 11, 14])), 'stage'] = 'Round3'
    
    fil = data.stage.isna()
    
    data.loc[fil & (data.T1_Seed.isin([1, 16, 8, 9, 4, 5, 12, 13])) & 
             (data.T2_Seed.isin([3, 6, 11, 14, 2, 7, 10, 15])), 'stage'] = 'Round4'
    data.loc[fil & (data.T1_Seed.isin([3, 6, 11, 14, 2, 7, 10, 15])) & 
             (data.T2_Seed.isin([1, 16, 8, 9, 4, 5, 12, 13])), 'stage'] = 'Round4'
    
    data.loc[data.stage.isna(), 'stage'] = 'impossible'
    
    data = pd.get_dummies(data, columns=['stage'])
    
    del data['T1_region']
    del data['T2_region']
    
    return data


def team_quality(season, data):
    '''
    Use a GLM to estimate the team quality in a season
    '''
    formula = 'win~-1+T1_TeamID+T2_TeamID'
    glm = sm.GLM.from_formula(formula=formula, 
                              data=data.loc[data.Season==season,:], 
                              family=sm.families.Binomial()).fit()
    
    quality = pd.DataFrame(glm.params).reset_index()
    quality.columns = ['TeamID','quality']
    quality['Season'] = season
    quality['quality'] = np.exp(quality['quality'])
    quality = quality.loc[quality.TeamID.str.contains('T1_')].reset_index(drop=True)
    quality['TeamID'] = quality['TeamID'].apply(lambda x: x[10:14]).astype(int)
    return quality


def add_quality(data, reg):
    '''
    Take the regular season game data in the original W/L format
    Double the dataframe inverting the team as usual
    Call the team_quality function per season
    Merge the result with the training data in the T1/T2 format
    
    data: training data in the T1/T2 format
    reg: regular season data in the W/L format
    '''
    reg = reg[['Season', 'WTeamID', 'LTeamID', 'WScore', 'LScore']].copy()
    reg_inv = reg.copy()
    
    reg.columns = [x.replace('W','T1_').replace('L','T2_') for x in list(reg.columns)]
    reg_inv.columns = [x.replace('L','T1_').replace('W','T2_') for x in list(reg_inv.columns)]

    reg = pd.concat([reg, reg_inv]).sort_index().reset_index(drop = True)
    
    reg['PointDiff'] = reg['T1_Score'] - reg['T2_Score']
    reg['T1_TeamID'] = reg['T1_TeamID'].astype(str)
    reg['T2_TeamID'] = reg['T2_TeamID'].astype(str)
    reg['win'] = np.where(reg['PointDiff']>0,1,0)
    
    all_quality = []
    for year in reg.Season.unique():
        all_quality.append(team_quality(year, reg))
                           
    all_quality = pd.concat(all_quality, ignore_index=True)
    
    team_quality_T1 = all_quality[['TeamID','Season','quality']]
    team_quality_T1.columns = ['Team1','Season','T1_quality']
    team_quality_T2 = all_quality[['TeamID','Season','quality']]
    team_quality_T2.columns = ['Team2','Season','T2_quality']

    data['Team1'] = data['Team1'].astype(int)
    data['Team2'] = data['Team2'].astype(int)
    data = data.merge(team_quality_T1, on = ['Team1','Season'], how = 'left')
    data = data.merge(team_quality_T2, on = ['Team2','Season'], how = 'left')
    
    return data


def add_days(data, info, date=True):
    df = data.copy()
    seasons = pd.read_csv(info)
    
    df = pd.merge(df, seasons[['Season', 'DayZero']], on='Season')
    df['DayZero'] = pd.to_datetime(df.DayZero)
    
    if date:
        df['GameDay'] = df.apply(lambda x: x['DayZero'] + pd.offsets.DateOffset(days=x['DayNum']), 1)
    else:
        df['DayNum'] = (df['GameDay'] - df['DayZero']).dt.days
    
    del df['DayZero']
    
    return df
