__author__ = 'lucabasa'
__version__ = '5.1.0'
__status__ = 'development'


import pandas as pd 
import numpy as np

from source.aggregated_stats import process_details, full_stats, rolling_stats
from source.add_info import add_seed, add_rank, highlow_seed, add_stage, add_quality


def make_teams_target(data, league):
    '''
    Take the playoff compact data and double the dataframe by inverting W and L
    It also creates the ID column
    
    data: playoff compact results
    league: men or women, useful to know when to cut the data
    '''
    if league == 'men':
        limit = 2003
    else:
        limit = 2010

    df = data[data.Season >= limit].copy()

    df['Team1'] = np.where((df.WTeamID < df.LTeamID), df.WTeamID, df.LTeamID)
    df['Team2'] = np.where((df.WTeamID > df.LTeamID), df.WTeamID, df.LTeamID)
    df['target'] = np.where((df['WTeamID'] < df['LTeamID']), 1, 0)
    df['target_points'] = np.where((df['WTeamID'] < df['LTeamID']), df.WScore - df.LScore, df.LScore - df.WScore)
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


def make_training_data(details, targets):
    '''
    details: seasonal stats by team
    targets: result of make_teams_target with each playoff game present twice
    
    Add the prefix T1_ and T2_ to the seasonal stats and add it to the playoff game
    This creates the core training set where we use seasonal stats to predict the playoff games
    
    Add the delta_ statistics, given by the difference between T1_ and T2_
    '''
    tmp = details.copy()
    tmp.columns = ['Season', 'Team1'] + \
                ['T1_'+col for col in tmp.columns if col not in ['Season', 'TeamID']]
    total = pd.merge(targets, tmp, on=['Season', 'Team1'], how='left')

    tmp = details.copy()
    tmp.columns = ['Season', 'Team2'] + \
                ['T2_'+col for col in tmp.columns if col not in ['Season', 'TeamID']]
    total = pd.merge(total, tmp, on=['Season', 'Team2'], how='left')
    
    if total.isnull().any().any():
        print(total.columns[total.isnull().any()])
        raise ValueError('Something went wrong')
        
    stats = [col[3:] for col in total.columns if 'T1_' in col and 'region' not in col]

    for stat in stats:
        total['delta_'+stat] = total['T1_'+stat] - total['T2_'+stat]
        
    try:
        total['delta_off_edge'] = total['T1_off_rating'] - total['T2_def_rating']
        total['delta_def_edge'] = total['T2_off_rating'] - total['T1_def_rating']
    except KeyError:
        pass
        
    return total


def prepare_data(league):
    save_loc = 'processed_data/' + league + '/'

    if league == 'women':
        regular_season = 'data/raw_women/WDataFiles_Stage2/WRegularSeasonDetailedResults.csv'
        playoff = 'data/raw_women/WDataFiles_Stage2/WNCAATourneyDetailedResults.csv'
        playoff_compact = 'data/raw_women/WDataFiles_Stage2/WNCAATourneyCompactResults.csv'
        seed = 'data/raw_women/WDataFiles_Stage2/WNCAATourneySeeds.csv'
        rank = None
        stage2 = 'data/raw_women/WDataFiles_Stage2/WSampleSubmissionStage2.csv'
        stage2_yr = 2022
        save_loc = 'data/processed_women/'
    else:
        regular_season = 'data/raw_men/MDataFiles_Stage2/MRegularSeasonDetailedResults.csv'
        playoff = 'data/raw_men/MDataFiles_Stage2/MNCAATourneyDetailedResults.csv'
        playoff_compact = 'data/raw_men/MDataFiles_Stage2/MNCAATourneyCompactResults.csv'
        seed = 'data/raw_men/MDataFiles_Stage2/MNCAATourneySeeds.csv'
        rank = 'data/raw_men/MDataFiles_Stage2/MMasseyOrdinals_thruDay128.csv'
        stage2 = 'data/raw_men/MDataFiles_Stage2/MSampleSubmissionStage2.csv'
        stage2_yr = 2022
        save_loc = 'data/processed_men/'
    
    # Season stats
    reg = pd.read_csv(regular_season)
    reg = process_details(reg, rank)
    reg.to_csv(save_loc + 'game_details_regular_extended.csv', index=False)
    regular_stats = full_stats(reg)
    
    # Last 2 weeks stats
    last2weeks = reg[reg.DayNum >= 118].copy()
    last2weeks = full_stats(last2weeks)
    last2weeks.columns = ['L2W_' + col for col in last2weeks]
    last2weeks.rename(columns={'L2W_Season': 'Season', 'L2W_TeamID': 'TeamID'}, inplace=True)
    
    regular_stats = pd.merge(regular_stats, last2weeks, on=['Season', 'TeamID'], how='left')
    
    regular_stats = add_seed(seed, regular_stats)    
    
    # Playoff stats
    play = pd.read_csv(playoff)
    play = process_details(play)
    play.to_csv(save_loc + 'game_details_playoff_extended.csv', index=False)
    playoff_stats = full_stats(play)
    
    playoff_stats = add_seed(seed, playoff_stats)
    
    if rank:
        regular_stats = add_rank(rank, regular_stats)
        playoff_stats = add_rank(rank, playoff_stats)
    
    # Target data generation 
    target_data = pd.read_csv(playoff_compact)
    target_data = make_teams_target(target_data, league)
    
    # Add high and low seed wins perc
    regular_stats = highlow_seed(regular_stats, reg, seed)
    
    all_reg = make_training_data(regular_stats, target_data)
    all_reg = all_reg[all_reg.DayNum >= 136]  # remove pre tourney 
    all_reg = add_stage(all_reg)
    all_reg = add_quality(all_reg, reg)
    all_reg.to_csv(save_loc + 'training_data.csv', index=False)        
    
    playoff_stats.to_csv(save_loc + 'playoff_stats.csv', index=False)
    
    if stage2:
        test_data_reg = regular_stats[regular_stats.Season == stage2_yr].copy()
        sub = pd.read_csv(stage2)
        sub['Team1'] = sub['ID'].apply(lambda x: int(x[5:9]))
        sub['Team2'] = sub['ID'].apply(lambda x: int(x[10:]))
        tmp = sub.copy()
        tmp = tmp.rename(columns={'Team1': 'Team2', 'Team2': 'Team1'})
        tmp = tmp[['Team1', 'Team2', 'Pred']]
        sub = pd.concat([sub[['Team1', 'Team2', 'Pred']], tmp], ignore_index=True)
        sub['Season'] = stage2_yr
        test_data = make_training_data(test_data_reg, sub)
        test_data = add_stage(test_data)
        test_data = add_quality(test_data, reg[reg.Season == stage2_yr])
        test_data.to_csv(save_loc + f'{stage2_yr}_test_data.csv', index=False)
        return all_reg, test_data
    
    return all_reg


def prepare_competitive(league):
    if league == 'women':
        regular_season = 'data/raw_women/WDataFiles_Stage2/WRegularSeasonDetailedResults.csv'
        playoff = 'data/raw_women/WDataFiles_Stage2/WNCAATourneyDetailedResults.csv'
        rank = None
        season_info = 'data/raw_women/WDataFiles_Stage2/WSeasons.csv'
        events_data = 'data/processed_women/events.csv'
        save_loc = 'data/processed_women/'
    else:
        regular_season = 'data/raw_men/MDataFiles_Stage2/MRegularSeasonDetailedResults.csv'
        playoff = 'data/raw_men/MDataFiles_Stage2/MNCAATourneyDetailedResults.csv'
        playoff_compact = 'data/raw_men/MDataFiles_Stage2/MNCAATourneyCompactResults.csv'
        rank = 'data/raw_men/MDataFiles_Stage2/MMasseyOrdinals.csv'
        season_info = 'data/raw_men/MDataFiles_Stage2/MSeasons.csv'
        events_data = 'data/processed_men/events.csv'
        save_loc = 'data/processed_men/'
        
    reg = pd.read_csv(regular_season)
    reg = process_details(reg, rank)
    play = pd.read_csv(playoff)
    play = process_details(play)
    full = pd.concat([reg, play])
    events = pd.read_csv(events_data)
    
    to_use = [col for col in events if not col.endswith('_game') and 
              'FinalScore' not in col and 
              'n_OT' not in col and 
              '_difference' not in col]
    full = pd.merge(full, events[to_use], on=['Season', 'DayNum', 'WTeamID', 'LTeamID'])
    
    full.to_csv(save_loc + 'events_extended.csv', index=False)
    
    rolling = rolling_stats(full, season_info)
    
    rolling.to_csv(save_loc + 'rolling_stats.csv', index=False)
    
    competitive = events[['Season', 'DayNum', 'WTeamID', 'LTeamID', 
                          'tourney', 'Final_difference', 'Halftime_difference', '3mins_difference', 
                          'game_lc', 'half2_lc', 'crunchtime_lc', 'competitive']].copy()
    
    tmp = rolling.copy()
    tmp.columns = ['Season'] + \
                ['W'+col for col in tmp.columns if col not in ['Season', 'DayNum']] + ['DayNum']
    
    competitive = pd.merge(competitive, tmp, on=['Season', 'DayNum', 'WTeamID'])
    
    tmp = rolling.copy()
    tmp.columns = ['Season'] + \
                ['L'+col for col in tmp.columns if col not in ['Season', 'DayNum']] + ['DayNum']
    
    competitive = pd.merge(competitive, tmp, on=['Season', 'DayNum', 'LTeamID'])
    
    competitive.to_csv(save_loc + 'competitive.csv', index=False)
    
    return competitive


if __name__=='__main__':
    prepare_data('men')
    prepare_data('women')
    






