__author__ = 'lucabasa'
__version__ = '2.4.0'
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


def add_rank(rank_location, total):
    ranks = pd.read_csv(rank_location)
    ranks = ranks[~(ranks.SystemName.isin(['AP', 'USA', 'DES', 'LYN', 'ACU', 
                                           'TRX', 'D1A', 'JNG', 'BNT']))].copy()
    ranks = ranks.groupby(['Season', 'TeamID', 'RankingDayNum'], as_index=False).OrdinalRank.mean()
    ranks = ranks[ranks.RankingDayNum == 133]
    del ranks['RankingDayNum']

    total = pd.merge(total, ranks.rename(columns={'OrdinalRank': 'Rank'}), 
                     on=['Season', 'TeamID'], how='left')

    return total


def add_seed(seed_location, total):
    seed_data = pd.read_csv(seed_location)
    seed_data['region'] = seed_data['Seed'].apply(lambda x: x[0])
    seed_data['Seed'] = seed_data['Seed'].apply(lambda x: int(x[1:3]))
    total = pd.merge(total, seed_data, how='left', on=['TeamID', 'Season'])
    return total


def add_stage(data):
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
        regular_season = 'data/raw_women/WDataFiles_Stage1/WRegularSeasonDetailedResults.csv'
        playoff = 'data/raw_women/WDataFiles_Stage1/WNCAATourneyDetailedResults.csv'
        playoff_compact = 'data/raw_women/WDataFiles_Stage1/WNCAATourneyCompactResults.csv'
        seed = 'data/raw_women/WDataFiles_Stage1/WNCAATourneySeeds.csv'
        rank = None
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
    reg = process_details(reg, rank)
    reg.to_csv(save_loc + 'game_details_regular_extended.csv', index=False)
    regular_stats = full_stats(reg)
    
    # Last 2 weeks stats
#     last2weeks = reg[reg.DayNum >= 118].copy()
#     last2weeks = full_stats(last2weeks)
#     last2weeks.columns = ['L2W_' + col for col in last2weeks]
#     last2weeks.rename(columns={'L2W_Season': 'Season', 'L2W_TeamID': 'TeamID'}, inplace=True)
    
#     regular_stats = pd.merge(regular_stats, last2weeks, on=['Season', 'TeamID'], how='left')
    
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
    
    all_reg = make_training_data(regular_stats, target_data)
    all_reg = all_reg[all_reg.DayNum >= 136]  # remove pre tourney 
    all_reg = add_stage(all_reg)
    all_reg.to_csv(save_loc + 'training_data.csv', index=False)
    
    playoff_stats.to_csv(save_loc + 'playoff_stats.csv', index=False)
    
    return all_reg


if __name__=='__main__':
    prepare_data('men')
    prepare_data('women')
    






