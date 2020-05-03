__author__ = 'lucabasa'
__version__ = '3.0.0'
__status__ = 'development'


import pandas as pd 
import numpy as np 


def big_wins(data, rank_loc):
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


def process_details(data, rank_loc=None):
    df = data.copy()
    
    df = big_wins(df, rank_loc)
        
    for prefix in ['W', 'L']:
        df[prefix+'FG_perc'] = df[prefix+'FGM'] / df[prefix+'FGA']
        df[prefix+'FGM2'] = df[prefix+'FGM'] - df[prefix+'FGM3']
        df[prefix+'FGA2'] = df[prefix+'FGA'] - df[prefix+'FGA3']
        df[prefix+'FG2_perc'] = df[prefix+'FGM2'] / df[prefix+'FGA2']
        df[prefix+'FG3_perc'] = df[prefix+'FGM3'] / df[prefix+'FGA3']
        df[prefix+'FT_perc'] = df[prefix+'FTM'] / df[prefix+'FTA']
        df[prefix+'Tot_Reb'] = df[prefix+'OR'] + df[prefix+'DR']
        df[prefix+'FGM_no_ast'] = df[prefix+'FGM'] - df[prefix+'Ast']
        df[prefix+'FGM_no_ast_perc'] = df[prefix+'FGM_no_ast'] / df[prefix+'FGM']
        df[prefix+'possessions'] = df[prefix+'FGA'] - df[prefix+'OR'] + df[prefix+'TO'] + 0.475*df[prefix+'FTA']
        df[prefix+'off_rating'] = df[prefix+'Score'] / df[prefix+'possessions'] * 100
        df[prefix+'shtg_opportunity'] = 1 + (df[prefix+'OR'] - df[prefix+'TO']) / df[prefix+'possessions']
        df[prefix+'TO_perposs'] = df[prefix+'TO'] / df[prefix+'possessions']
        df[prefix+'True_shooting_perc'] = 0.5 * df[prefix+'Score'] / (df[prefix+'FGA'] + 0.475 * df[prefix+'FTA'])
        df[prefix+'IE_temp'] = df[prefix+'Score'] + df[prefix+'FTM'] + df[prefix+'FGM'] + \
                                df[prefix+'DR'] + 0.5*df[prefix+'OR'] - df[prefix+'FTA'] - df[prefix+'FGA'] + \
                                df[prefix+'Ast'] + df[prefix+'Stl'] + 0.5*df[prefix+'Blk'] - df[prefix+'PF']

    df['Wdef_rating'] = df['Loff_rating']
    df['Ldef_rating'] = df['Woff_rating']
    df['Wopp_shtg_opportunity'] = df['Lshtg_opportunity']
    df['Lopp_shtg_opportunity'] = df['Wshtg_opportunity']
    df['Wopp_possessions'] = df['Lpossessions']
    df['Lopp_possessions'] = df['Wpossessions']
    df['Wopp_score'] = df['LScore']
    df['Lopp_score'] = df['WScore']
    # These will be needed for the true shooting percentage when we aggregate
    df['Wopp_FTA'] = df['LFTA']
    df['Wopp_FGA'] = df['LFGA']
    df['Lopp_FTA'] = df['WFTA']
    df['Lopp_FGA'] = df['WFGA']

    df['Wimpact'] = df['WIE_temp'] / (df['WIE_temp'] + df['LIE_temp'])
    df['Limpact'] = df['LIE_temp'] / (df['WIE_temp'] + df['LIE_temp'])

    del df['WIE_temp']
    del df['LIE_temp']

    df[[col for col in df.columns if 'perc' in col]] = df[[col for col in df.columns if 'perc' in col]].fillna(0)

#     df['WDef_effort'] = df['LFGA2']*2 + df['LFGA3']*3 - \
#                         df['LFGM2']*2 - df['LFGM3']*3 - \
#                         df['LOR']*( (df['LFGA2']/df['LFGA'])*2 + (df['LFGA3']/df['LFGA'])*3 ) + \
#                         df['LFTA']*( (df['LFGA2']/df['LFGA'])*2 + (df['LFGA3']/df['LFGA'])*3 ) - df['LFTM']
#     df['LDef_effort'] = df['WFGA2']*2 + df['WFGA3']*3 - \
#                         df['WFGM2']*2 - df['WFGM3']*3 - \
#                         df['WOR']*( (df['WFGA2']/df['WFGA'])*2 + (df['WFGA3']/df['WFGA'])*3 ) + \
#                         df['WFTA']*( (df['WFGA2']/df['WFGA'])*2 + (df['WFGA3']/df['WFGA'])*3 ) - df['WFTM']

    df['WDR_opportunity'] = df['WDR'] / (df['LFGA'] - df['LFGM'])
    df['LDR_opportunity'] = df['LDR'] / (df['WFGA'] - df['WFGM'])
    df['WOR_opportunity'] = df['WOR'] / (df['WFGA'] - df['WFGM'])
    df['LOR_opportunity'] = df['LOR'] / (df['LFGA'] - df['LFGM'])
    
    stats = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 
             'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 
             'PF', 'FGM2', 'FGA2', 'Tot_Reb', 'FGM_no_ast', 
             'DR_opportunity', 'OR_opportunity', 'possessions',
             'off_rating', 'def_rating', 'shtg_opportunity', 
             'TO_perposs', 'impact', 'True_shooting_perc'] # 'Def_effort' 
    
    for col in stats:
        df[col+'_diff'] = df['W'+col] - df['L'+col]
        df[col+'_advantage'] = (df[col+'_diff'] > 0).astype(int)
    
    return df


def perc_OT_win(data):
    df = data[['Season', 'TeamID', 'NumOT', 'OT_win']].copy()
    df['has_OT'] = np.where(df.NumOT > 0, 1, 0)
    
    df = df.groupby(['Season', 'TeamID', 'has_OT'], as_index=False).OT_win.mean()
    df = df[df.has_OT > 0].copy()
    del df['has_OT']
    
    return df.rename(columns={'OT_win': 'OT_win_perc'})


def full_stats(data):
    df = data.copy()
    
    to_select = [col for col in df.columns if col.startswith('W') 
                                             and '_perc' not in col 
                                             and 'Loc' not in col]
    to_select += [col for col in df.columns if '_diff' in col or '_advantage' in col]
    df_W = df[['Season', 'DayNum', 'NumOT'] + to_select].copy()
    df_W.columns = df_W.columns.str.replace('W','')
    df_W['N_wins'] = 1
    
    to_select = [col for col in df.columns if col.startswith('L') 
                                             and '_perc' not in col 
                                             and 'Loc' not in col]
    to_select += [col for col in df.columns if '_diff' in col or '_advantage' in col]
    df_L = df[['Season', 'DayNum', 'NumOT'] + to_select].copy()
    df_L.columns = df_L.columns.str.replace('L','')
    df_L[[col for col in df.columns if '_diff' in col]] = - df_L[[col for col in df.columns if '_diff' in col]]
    for col in [col for col in df.columns if '_advantage' in col]:
        df_L[col] = df_L[col].map({0:1, 1:0})
    df_L['N_wins'] = 0
    df_L['OT_win'] = 0
    df_L['Away'] = 0
    if 'top_team' in df_W.columns:
        df_L['top_team'] = 0
        df_L['upset'] = 0

    df = pd.concat([df_W, df_L], sort=True)
    
    del df['DayNum']
    
    OT_perc = perc_OT_win(df)
    
    not_use = ['NumOT']
    to_use = [col for col in df.columns if col not in not_use]
    
    means = df[to_use].groupby(['Season','TeamID'], as_index=False).mean()
    
    sums = df[to_use].groupby(['Season','TeamID'], as_index=False).sum()
    sums['FGM_perc'] = sums.FGM / sums.FGA
    sums['FGM2_perc'] = sums.FGM2 / sums.FGA2
    sums['FGM3_perc'] = sums.FGM3 / sums.FGA3
    sums['FT_perc'] = sums.FTM / sums.FTA
    sums['FGM_no_ast_perc'] = sums.FGM_no_ast / sums.FGM
    sums['True_shooting_perc'] = 0.5 * sums['Score'] / (sums['FGA'] + 0.475 * sums['FTA'])
    sums['Opp_True_shooting_perc'] = 0.5 * sums['opp_score'] / (sums['opp_FGA'] + 0.475 * sums['opp_FTA'])
    to_use = ['Season', 'TeamID', 'FGM_perc',
              'FGM2_perc', 'FGM3_perc', 'FT_perc', 
              'FGM_no_ast_perc', 'True_shooting_perc', 'Opp_True_shooting_perc']
    
    sums = sums[to_use].fillna(0)
    
    stats_tot = pd.merge(means, sums, on=['Season', 'TeamID'])
    stats_tot = pd.merge(stats_tot, OT_perc, on=['Season', 'TeamID'], how='left')
    stats_tot['OT_win_perc'] = stats_tot['OT_win_perc'].fillna(0)
  
    return stats_tot


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


def rolling_stats(data, season_info, window='30d'):
    df = data.copy()

    df = add_days(df, season_info)

    to_select = [col for col in df.columns if col.startswith('W') 
                                                 and '_perc' not in col 
                                                 and 'Loc' not in col]
    to_select += [col for col in df.columns if '_diff' in col or '_advantage' in col]
    df_W = df[['Season', 'GameDay', 'NumOT', 
               'game_lc', 'half2_lc', 'crunchtime_lc'] + to_select].copy()
    df_W.columns = df_W.columns.str.replace('W','')
    df_W['N_wins'] = 1

    to_select = [col for col in df.columns if col.startswith('L') 
                                             and '_perc' not in col 
                                             and 'Loc' not in col]
    to_select += [col for col in df.columns if '_diff' in col or '_advantage' in col]
    df_L = df[['Season', 'GameDay', 'NumOT', 
               'game_lc', 'half2_lc', 'crunchtime_lc'] + to_select].copy()
    df_L.columns = df_L.columns.str.replace('L','')
    df_L[[col for col in df.columns if '_diff' in col]] = - df_L[[col for col in df.columns if '_diff' in col]]
    for col in [col for col in df.columns if '_advantage' in col]:
        df_L[col] = df_L[col].map({0:1, 1:0})
    df_L['N_wins'] = 0
    df_L['OT_win'] = 0
    df_L['Away'] = 0
    if 'top_team' in df_W.columns:
        df_L['top_team'] = 0
        df_L['upset'] = 0

    df = pd.concat([df_W, df_L], sort=False)

    not_use = ['NumOT', 'Season', 'TeamID']
    to_use = [col for col in df.columns if col not in not_use]

    means = df.groupby(['Season', 'TeamID'])[to_use].rolling(window, on='GameDay', 
                                                           min_periods=1, closed='left').mean()
    means = means.dropna()
    means = means.reset_index()
    del means['level_2']

    sums = df.groupby(['Season', 'TeamID'])[to_use].rolling(window, on='GameDay', 
                                                      min_periods=1, closed='left').sum()
    sums = sums.reset_index()
    del sums['level_2']
    
    sums['FGM_perc'] = sums.FGM / sums.FGA
    sums['FGM2_perc'] = sums.FGM2 / sums.FGA2
    sums['FGM3_perc'] = sums.FGM3 / sums.FGA3
    sums['FT_perc'] = sums.FTM / sums.FTA
    sums['FGM_no_ast_perc'] = sums.FGM_no_ast / sums.FGM
    sums['True_shooting_perc'] = 0.5 * sums['Score'] / (sums['FGA'] + 0.475 * sums['FTA'])
    sums['Opp_True_shooting_perc'] = 0.5 * sums['opp_score'] / (sums['opp_FGA'] + 0.475 * sums['opp_FTA'])
    
    to_use = ['Season', 'TeamID', 'GameDay', 'FGM_perc',
              'FGM2_perc', 'FGM3_perc', 'FT_perc', 
              'FGM_no_ast_perc', 'True_shooting_perc', 'Opp_True_shooting_perc']

    sums = sums[to_use].fillna(0)

    stats_tot = pd.merge(means, sums, on=['Season', 'TeamID', 'GameDay'])

    stats_tot = add_days(stats_tot, season_info, date=False)
    del stats_tot['GameDay']
    
    return stats_tot

