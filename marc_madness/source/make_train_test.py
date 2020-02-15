__author__ = 'lucabasa'
__version__ = '2.0.0'
__status__ = 'development'


import pandas as pd 
import numpy as np 

from source.aggregated_stats import process_details, full_stats


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


def add_seed(seed_location, total):
    seed_data = pd.read_csv(seed_location)
    seed_data['Seed'] = seed_data['Seed'].apply(lambda x: int(x[1:3]))
    total = pd.merge(total, seed_data, how='left', on=['TeamID', 'Season'])
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


def make_training_data(details, targets):
    tmp = details.copy()
    tmp.columns = ['Season', 'Team1'] + \
                ['T1_'+col for col in tmp.columns if col not in ['Season', 'TeamID']]
    total = pd.merge(targets, tmp, on=['Season', 'Team1'], how='left')

    tmp = details.copy()
    tmp.columns = ['Season', 'Team2'] + \
                ['T2_'+col for col in tmp.columns if col not in ['Season', 'TeamID']]
    total = pd.merge(total, tmp, on=['Season', 'Team2'], how='left')
    
    if total.isnull().any().any():
        raise ValueError('Something went wrong')
        
    stats = [col[3:] for col in total.columns if 'T1_' in col]

    for stat in stats:
        total['delta_'+stat] = total['T1_'+stat] - total['T2_'+stat]
        
    return total


def prepare_data(league):
    save_loc = 'processed_data/' + league + '/'

    if league == 'women':
        regular_season = 'data/raw_women/WDataFiles_Stage1/WRegularSeasonDetailedResults.csv'
        playoff = 'data/raw_women/WDataFiles_Stage1/WNCAATourneyDetailedResults.csv'
        playoff_compact = 'data/raw_women/WDataFiles_Stage1/WNCAATourneyCompactResults.csv'
        seed = 'data/raw_women/WDataFiles_Stage1/WNCAATourneySeeds.csv'
        save_loc = 'data/processed_women/'
    else:
        regular_season = 'data/raw_men/MDataFiles_Stage1/MRegularSeasonDetailedResults.csv'
        playoff = 'data/raw_men/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv'
        playoff_compact = 'data/raw_men/MDataFiles_Stage1/MNCAATourneyCompactResults.csv'
        seed = 'data/raw_men/MDataFiles_Stage1/MNCAATourneySeeds.csv'
        rank = 'data/raw_men/MDataFiles_Stage1/MMasseyOrdinals.csv'
        save_loc = 'data/processed_men/'
    
    # Season stats
    reg = pd.read_csv(regular_season)
    reg = process_details(reg)
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
    
    # Target data generation Todo: this is up to 2018
    target_data = pd.read_csv(playoff_compact)
    target_data = make_teams_target(target_data, league)
    
    all_reg = make_training_data(regular_stats, target_data)
    all_reg.to_csv(save_loc + 'training_data.csv', index=False)
    
    playoff_stats.to_csv(save_loc + 'playoff_stats.csv', index=False)
    
    return all_reg


if __name__=='__main__':
    prepare_data('men')
    prepare_data('women')
    






