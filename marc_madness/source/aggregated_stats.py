__author__ = 'lucabasa'
__version__ = '1.7.0'
__status__ = 'development'


import pandas as pd 
import numpy as np 



def process_details(data):
    df = data.copy()
        
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


def rolling_stats(data, streak=None):
    df = data.copy()
    
    to_select = [col for col in df.columns if col.startswith('W') and '_perc' not in col]
    to_select += [col for col in df.columns if '_diff' in col]
    df_W = df[['Season', 'DayNum', 'NumOT'] + to_select].copy()
    df_W.columns = df_W.columns.str.replace('W','')
    df_W['N_wins'] = 1
    
    to_select = [col for col in df.columns if col.startswith('L') and '_perc' not in col]
    to_select += [col for col in df.columns if '_diff' in col]
    df_L = df[['Season', 'DayNum', 'NumOT'] + to_select].copy()
    df_L.columns = df_L.columns.str.replace('L','')
    df_L = df_L.rename(columns={'Woc': 'Loc'})
    df_L[[col for col in df.columns if '_diff' in col]] = - df_L[[col for col in df.columns if '_diff' in col]]
    df_L['N_wins'] = 0
    
    df = pd.concat([df_W, df_L]).sort_values(by=['Season', 'DayNum'])

    df.Loc = df.Loc.map({'H': 1, 'A': -1, 'N': 0})
    
    to_use = [col for col in df.columns if col != 'NumOT']
    
    if streak:
        window = streak
        df['H_loss'] = 0
        df['H_win'] = 0
        df.loc[df.Score_diff > 15, 'H_win'] = 1
        df.loc[df.Score_diff < -15, 'H_loss'] = 1
        to_use = [col for col in df.columns]
    else:
        window = 10000

    means = df[to_use].groupby(['Season','TeamID'], 
                               as_index=False).rolling(window, on='DayNum', min_periods=1).mean()
    means = means.reset_index(drop=True)
    means.Season = means.Season.astype(int)
    means.TeamID = means.TeamID.astype(int)
    
    sums = df[to_use].groupby(['Season','TeamID']).rolling(window, on='DayNum', min_periods=1).sum()
    del sums['Season']
    del sums['TeamID']
    sums = sums.reset_index([0,1]).reset_index(drop=True)
    sums['FGM_perc'] = sums.FGM / sums.FGA
    sums['FGM2_perc'] = sums.FGM2 / sums.FGA2
    sums['FGM3_perc'] = sums.FGM3 / sums.FGA3
    sums['FT_perc'] = sums.FTM / sums.FTA
    sums['FGM_no_ast_perc'] = sums.FGM_no_ast / sums.FGM
    
    if streak:
        to_use = ['Season', 'TeamID', 'DayNum', 'Loc', 'FGM_perc', 
                 'FGM2_perc', 'FGM3_perc', 'FT_perc', 'FGM_no_ast_perc', 
                  'H_loss', 'H_win']
    else:
        to_use = ['Season', 'TeamID', 'DayNum', 'FGM_perc', 
                 'FGM2_perc', 'FGM3_perc', 'FT_perc', 'FGM_no_ast_perc']
    
    sums = sums[to_use].fillna(0)
    
    stats_tot = pd.merge(means, sums, on=['Season', 'TeamID', 'DayNum'])
    
    stats_tot = stats_tot.rename(columns={'H_loss_x': 'H_loss_mean', 'H_loss_y': 'H_loss_sum', 
                                         'H_win_x': 'H_win_mean', 'H_win_y': 'H_win_sum'})
    
    return stats_tot


def streaks(data):
    data['streak2'] = (data['Score_diff'] > 0).cumsum()
    data['cumsum'] = np.nan
    data.loc[data['Score_diff'] < 0, 'cumsum'] = data['streak2']
    data['cumsum'] = data['cumsum'].fillna(method='ffill')
    data['cumsum'] = data['cumsum'].fillna(0)
    data['w_streak'] = data['streak2'] - data['cumsum']
    data['streak2'] = (data['Score_diff'] < 0).cumsum()
    data['cumsum'] = np.nan
    data.loc[data['Score_diff'] > 0, 'cumsum'] = data['streak2']
    data['cumsum'] = data['cumsum'].fillna(method='ffill')
    data['cumsum'] = data['cumsum'].fillna(0)
    data['l_streak'] = data['streak2'] - data['cumsum']
    
    data = data.drop(['streak2', 'cumsum'], axis=1)
    
    return data


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
    df_L = df_L.rename(columns={'Woc': 'Loc'})
    df_L[[col for col in df.columns if '_diff' in col]] = - df_L[[col for col in df.columns if '_diff' in col]]
    for col in [col for col in df.columns if '_advantage' in col]:
        df_L[col] = df_L[col].map({0:1, 1:0})
    df_L['N_wins'] = 0
    
    df = pd.concat([df_W, df_L])

    #df.Loc = df.Loc.map({'H': 1, 'A': -1, 'N': 0})
    
    del df['DayNum']
    
    not_use = ['NumOT', 'opp_FTA', 'opp_FGA']
    to_use = [col for col in df.columns if col != 'NumOT']
    
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
  
    return stats_tot


def aggregated_stats(league):
    save_loc = 'processed_data/'+league + '/'

    if league == 'women':
        origins = {'reg': 'data/raw_men/Stage2DataFiles/RegularSeasonDetailedResults.csv',
                'play': 'raw_data/womens-machine-learning-competition-2019/WDataFiles/WNCAATourneyDetailedResults.csv'}
    else:
        origins = {'reg': 'raw_data/mens-machine-learning-competition-2019/DataFiles/RegularSeasonDetailedResults.csv',
                'play': 'raw_data/mens-machine-learning-competition-2019/DataFiles/NCAATourneyDetailedResults.csv'}

    for phase in ['reg', 'play']:
        print(phase)
        data = pd.read_csv(origins[phase])
        data = process_details(data)
        data.to_csv(save_loc + 'game_details_' + phase + '_extended.csv', index=False)
        stats = full_stats(data)
        stats.to_csv(save_loc + 'teams_full_' + phase + '.csv', index=False)
        for streak in [None, 5, 10]:
            print(streak)
            stats = rolling_stats(data, streak=streak)
            if streak is None:
                streak = 'all_past'
            else:
                streak = 'last_' + str(streak)
            stats.to_csv(save_loc + 'teams_byday_' + phase + '_' + streak + '.csv', index=False)



if __name__=='__main__':
    aggregated_stats('men')
    aggregated_stats('women')
