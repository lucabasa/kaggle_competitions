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

