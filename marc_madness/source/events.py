__author__ = 'lucabasa'
__version__ = '1.1.0'
__status__ = 'development'


import pandas as pd
import numpy as np

import gc


def make_scores(data):
    to_keep = ['made1', 'made2', 'made3', 'miss1', 'miss2', 'miss3', 'reb', 'turnover', 'assist', 'steal', 'block']
    df = data[data.EventType.isin(to_keep)].copy()
    to_drop = ['EventPlayerID', 'EventSubType', 'X', 'Y', 'Area']
    df.drop(to_drop, axis=1, inplace=True)
    
    df['tourney'] = np.where(df.DayNum >= 132, 1, 0)
    
    df['points_made'] = 0
    df.loc[df.EventType == 'made1', 'points_made'] = 1
    df.loc[df.EventType == 'made2', 'points_made'] = 2
    df.loc[df.EventType == 'made3', 'points_made'] = 3
    df['tmp_gameID'] = df['DayNum'].astype(str) + '_' + df['WTeamID'].astype(str) + '_' + df['LTeamID'].astype(str)
    df['Final_difference'] = df['WFinalScore'] - df['LFinalScore']
    
    df = df.sort_values(by=['DayNum', 'WTeamID', 'ElapsedSeconds'])
    
    df['points'] = df.groupby(['tmp_gameID', 'EventTeamID']).points_made.cumsum() - df.points_made
    
    del df['WCurrentScore']
    del df['LCurrentScore']
    
    df.loc[df.WTeamID == df.EventTeamID, 'WCurrentScore'] = df.points
    df.loc[df.LTeamID == df.EventTeamID, 'LCurrentScore'] = df.points

    df['WCurrentScore'] = df.groupby('tmp_gameID')['WCurrentScore'].fillna(method='ffill').fillna(0)
    df['LCurrentScore'] = df.groupby('tmp_gameID')['LCurrentScore'].fillna(method='ffill').fillna(0)
    
    df['Current_difference'] = df['WCurrentScore'] - df['LCurrentScore']
    
    del df['points']
    del df['points_made']
    del df['tmp_gameID']
    
    return df


def quarter_score(data, men=True):
    if not men:
        data = data[~((data.DayNum == 80) & (data.WTeamID == 3111) & (data.LTeamID == 3117))]  # fix for one game with odd seconds
    df = data.copy()
    
    df['period'] = 1
    df.loc[df.ElapsedSeconds >= 20 * 60, 'period'] = 2
    df.loc[df.ElapsedSeconds >= 40 * 60, 'period'] = 3
    
    df['crunch'] = 0
    df.loc[(df.ElapsedSeconds > 37 * 60) & (df.ElapsedSeconds <= 40 * 60), 'crunch'] = 1
    
    df['minutes'] = df['ElapsedSeconds'] / 60
    df['tmp_gameID'] = df['DayNum'].astype(str) + '_' + df['WTeamID'].astype(str) + '_' + df['LTeamID'].astype(str)
    
    ot = ((df.groupby('tmp_gameID').minutes.max() - 40) / 5).reset_index()
    ot['n_OT'] = np.where(ot.minutes > 0, np.ceil(ot.minutes), 0)    
    half = df[df.period==1].groupby(['tmp_gameID'], as_index=False)[['WCurrentScore', 'LCurrentScore']].max()
    half['Halftime_difference'] = half['WCurrentScore'] - half['LCurrentScore']
    half.drop(['WCurrentScore', 'LCurrentScore'], axis=1, inplace=True)
    crunchtime = df[df.crunch==0].groupby(['tmp_gameID'], as_index=False)[['WCurrentScore', 'LCurrentScore']].max()
    crunchtime['3mins_difference'] = crunchtime['WCurrentScore'] - crunchtime['LCurrentScore']
    crunchtime.drop(['WCurrentScore', 'LCurrentScore'], axis=1, inplace=True)
    
    add_ons = pd.merge(ot[['tmp_gameID', 'n_OT']], half, on='tmp_gameID')
    add_ons = pd.merge(add_ons, crunchtime, on='tmp_gameID')
    
    df = pd.merge(df, add_ons, on='tmp_gameID')
    
    del df['tmp_gameID']
    del df['minutes']
    
    if data.shape[0] != df.shape[0]:
        raise KeyError('Some merge went wrong')
    
    return df


def lead_changes(data):
    df = data.copy()
    df['tmp_gameID'] = df['DayNum'].astype(str) + '_' + df['WTeamID'].astype(str) + '_' + df['LTeamID'].astype(str)
    
    changes = df.groupby('tmp_gameID').Current_difference.apply(lambda x: len(np.where(np.diff(np.sign(x)))[0])).reset_index()
    changes.rename(columns={'Current_difference': 'game_lc'}, inplace=True)
    changes_2 = df[df.period==2].groupby('tmp_gameID').Current_difference.apply(lambda x: len(np.where(np.diff(np.sign(x)))[0])).reset_index()
    changes_2.rename(columns={'Current_difference': 'half2_lc'}, inplace=True)
    changes_3 = df[df.crunch==1].groupby('tmp_gameID').Current_difference.apply(lambda x: len(np.where(np.diff(np.sign(x)))[0])).reset_index()
    changes_3.rename(columns={'Current_difference': 'crunchtime_lc'}, inplace=True)
    
    add_ons = pd.merge(changes, changes_2, on='tmp_gameID')
    add_ons = pd.merge(add_ons, changes_3, on='tmp_gameID', how='left')
    
    df = pd.merge(df, add_ons, on='tmp_gameID', how='left').fillna(0)
    
    del df['tmp_gameID']
    
    if data.shape[0] != df.shape[0]:
        raise KeyError('Some merge went wrong')
        
    return df


def _scoreinblock(data, text):
    
    df = data.groupby('tmp_gameID', as_index=False)[['WFinalScore', 'LFinalScore', 'WCurrentScore', 'LCurrentScore']].min()
    df[f'Wpoints_made_{text}'] = df['WFinalScore'] - df['WCurrentScore']
    df[f'Lpoints_made_{text}'] = df['LFinalScore'] - df['LCurrentScore']
    
    return df[['tmp_gameID', f'Wpoints_made_{text}', f'Lpoints_made_{text}']]


def _statcount(data, stat, text):
    
    tmp = data.copy()
    tmp['is_stat'] = np.where(tmp.EventType==stat, 1, 0)
    tmp = tmp.groupby(['tmp_gameID', 'EventTeamID'], as_index=False).is_stat.sum()
    
    return tmp.rename(columns={'is_stat': text})


def event_count(data):
    df = data.copy()
    df['tmp_gameID'] = df['DayNum'].astype(str) + '_' + df['WTeamID'].astype(str) + '_' + df['LTeamID'].astype(str)
    
    # points made in each block
    half2 = _scoreinblock(df[df.period==2], 'half2')
    crunch = _scoreinblock(df[df.crunch==1], 'crunchtime')
    
    add_ons = pd.merge(half2, crunch, on='tmp_gameID')
    add_ons = pd.merge(add_ons, df[['tmp_gameID', 'WTeamID', 'LTeamID']].drop_duplicates(), on='tmp_gameID')
    
    # stats in each block
    stats = ['made1', 'made2', 'made3', 'miss1', 'miss2', 'miss3', 'reb', 'turnover', 'assist', 'steal', 'block']
    
    period = 'game'    
    for stat in stats:
        name = f'{stat}_{period}'
        to_merge = _statcount(df, stat, name)
        add_ons = pd.merge(add_ons, to_merge.rename(columns={'EventTeamID': 'WTeamID', 
                                                   name: f'W{name}'}), on=['tmp_gameID', 'WTeamID'])
        add_ons = pd.merge(add_ons, to_merge.rename(columns={'EventTeamID': 'LTeamID', 
                                                   name: f'L{name}'}), on=['tmp_gameID', 'LTeamID'])
        gc.collect()
        
    period = 'half2'
    tmp = df[df.period==2]
    for stat in stats:
        name = f'{stat}_{period}'
        to_merge = _statcount(tmp, stat, name)
        add_ons = pd.merge(add_ons, to_merge.rename(columns={'EventTeamID': 'WTeamID', 
                                                   name: f'W{name}'}), on=['tmp_gameID', 'WTeamID'])
        add_ons = pd.merge(add_ons, to_merge.rename(columns={'EventTeamID': 'LTeamID', 
                                                   name: f'L{name}'}), on=['tmp_gameID', 'LTeamID'])
        gc.collect()
        
    period = 'crunchtime'
    tmp = df[df.crunch==1]
    for stat in stats:
        name = f'{stat}_{period}'
        to_merge = _statcount(tmp, stat, name)
        add_ons = pd.merge(add_ons, to_merge.rename(columns={'EventTeamID': 'WTeamID', 
                                                   name: f'W{name}'}), on=['tmp_gameID', 'WTeamID'])
        add_ons = pd.merge(add_ons, to_merge.rename(columns={'EventTeamID': 'LTeamID', 
                                                   name: f'L{name}'}), on=['tmp_gameID', 'LTeamID'])
        gc.collect()
    
    for period in ['game', 'half2', 'crunchtime']:
        # % of scores with assists
        add_ons[f'WAst_perc_{period}'] = (add_ons[f'Wassist_{period}'] / (add_ons[f'Wmade2_{period}'] + add_ons[f'Wmade3_{period}'])).fillna(0)
        add_ons[f'LAst_perc_{period}'] = (add_ons[f'Lassist_{period}'] / (add_ons[f'Lmade2_{period}'] + add_ons[f'Lmade3_{period}'])).fillna(0)
        # % scores
        add_ons[f'WFGM_perc_{period}'] = ((add_ons[f'Wmade2_{period}'] + add_ons[f'Wmade3_{period}'])
                                          / (add_ons[f'Wmade2_{period}'] + add_ons[f'Wmade3_{period}'] + 
                                             add_ons[f'Wmiss2_{period}'] + add_ons[f'Wmiss3_{period}'])).fillna(0)
        add_ons[f'LFGM_perc_{period}'] = ((add_ons[f'Lmade2_{period}'] + add_ons[f'Lmade3_{period}'])
                                          / ((add_ons[f'Lmade2_{period}'] + add_ons[f'Lmade3_{period}']) + 
                                             add_ons[f'Lmiss2_{period}'] + add_ons[f'Lmiss3_{period}'])).fillna(0)
        add_ons[f'WFGM3_perc_{period}'] = (add_ons[f'Wmade3_{period}'] / (add_ons[f'Wmade3_{period}'] + add_ons[f'Wmiss3_{period}'])).fillna(0)
        add_ons[f'LFGM3_perc_{period}'] = (add_ons[f'Lmade3_{period}'] / (add_ons[f'Lmade3_{period}'] + add_ons[f'Lmiss3_{period}'])).fillna(0)
        add_ons[f'WFTM_perc_{period}'] = (add_ons[f'Wmade1_{period}'] / (add_ons[f'Wmade1_{period}'] + add_ons[f'Wmiss1_{period}'])).fillna(0)
        add_ons[f'LFTM_perc_{period}'] = (add_ons[f'Lmade1_{period}'] / (add_ons[f'Lmade1_{period}'] + add_ons[f'Lmiss1_{period}'])).fillna(0)
        
    
    unique_cols = ['Season', 'DayNum', 'tourney', 'tmp_gameID', 'WTeamID', 'LTeamID', 
                   'WFinalScore', 'LFinalScore', 'Final_difference', 'n_OT', 
                   'Halftime_difference', '3mins_difference', 
                   'game_lc', 'half2_lc', 'crunchtime_lc']
    
    to_drop = ['WTeamID', 'LTeamID'] + [col for col in add_ons if 'miss' in col]
    
    df = pd.merge(df[unique_cols].drop_duplicates(), add_ons.drop(to_drop, axis=1), on='tmp_gameID')
    
    del df['tmp_gameID']
    
    return df


def make_competitive(data):
    df = data.copy()
    

    fil = ((df.Final_difference < 4) | (df['3mins_difference'] < 3) | (df.n_OT > 0) | 
         (df.game_lc > 20) | (df.half2_lc > 10) | (df.crunchtime_lc > 2))
    
    df['competitive'] = np.where(fil, 1, 0)
    
    return df


