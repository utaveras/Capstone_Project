################################################################################
'''
FYI - Stats Collected for >= 1979-1980 season (Addition of 3 Point Line and Stats)
'''
################################################################################
################################################################################
'''
                    Data Scraping Portion of Notebook
'''
################################################################################

# Import Libraries
import gc
import sys
# sys.path.remove('/Users/utaveras/FlatironSchool/DS-042219/Mod4/Mod4_Project')
sys.path.append('/Users/utaveras/FlatironSchool/DS-042219/Mod5_Project')
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/html5lib')
import csv
import json
import time
import re
import os
import timeit as timeit
from selenium import webdriver
import chromedriver_binary
from basketballCrawler import basketballCrawler as bc
from selenium.webdriver.support.select import Select
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup as BS
from datetime import datetime
import urllib.request
import requests
from lxml import html as lh, etree
from lxml.cssselect import CSSSelector
import pandas as pd
from PandasBasketball import pandasbasketball as pb
from functools import reduce
import operator
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
%matplotlib inline
plt.style.use('seaborn-white')

# Adjust Pandas Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Check working directory and change if unnecessary
os.getcwd()
# os.chdir('/Users/utaveras/FlatironSchool/DS-042219/Mod5_Project')

########################################
'''
        Function Definitions
'''
########################################


def height_to_inches(height):
    feet, inches = height.split('-')
    return int(feet) * 12 + int(inches)


def season_start_end(season, mask):
    season_start, season_end = season.split('-')
    if mask == 'end':
        return int(season_end)
    else:
        return int(season_start)


def get_season_num(player, year):
    season_list = sorted(df_pgmg[(df_pgmg['player_id']) == player]['season_start'].value_counts().index.tolist())
    season_dict = {}
    for n, y in enumerate(season_list):
        season_dict.update({int(y): int(n+1)})
    return season_dict.get(year)

# Function to Calculate Similarity Scores
def get_sim_scores_vect(targets, comps, results):
    for target in targets.itertuples():
        # Target Player Win Share Values
        t1 = getattr(target, '_1')
        t2 = getattr(target, '_2') * .95
        t3 = getattr(target, '_3') * .9

        # Comps Win Share Vector Values
        c1 = comps.loc[1]
        c2 = comps.loc[2] * .95
        c3 = comps.loc[3] * .9

        # Penalty Calcuations
        p1 = abs(t1 - c1)
        p2 = abs(t2 - c2)
        p3 = abs(t3 - c3)

        # Sum Totals
        st = t1 + t2 + t3
        sc = c1 + c2 + c3
        sp = p1 + p2 + p3

        # Calculate Simiarlties and store results in DataFrame
        scores = 100 * (1 - (2 * sp/(sc + st)))
        df_scores = scores.to_frame().T.reset_index(drop=True)
        df_scores.set_index(getattr(target, 'Index'), inplace=True)
        results = pd.concat([results, df_scores], sort=True)
    return results

########################################
'''
        Initalize variables
'''
########################################
custom_headers = {
    'Host': 'stats.nba.com',
    'Connection': 'keep-alive',
    'Cache-Control': 'max-age=0',
    'Upgrade-Insecure-Requests': '1',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.62 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}

### Load Scraped NBA Player Data from basketballCrawler saved in players.json file | count = 2866
# players = bc.buildPlayerDictionary()
# bc.savePlayerDictionary(players, '/Users/utaveras/FlatironSchool/DS-042219/Mod5_Project/players.json')
    ## Load Players from JSON File
p_file = '/Users/utaveras/FlatironSchool/DS-042219/Mod5_Project/players.json'
players = bc.loadPlayerDictionary(p_file)

### Create DataFrame containing individual Player Level Data
with open(p_file) as f:
    data = json.load(f)

player_list =[]
for k, v in data.items():
    print(v)
    p_dict = json.loads(v)
    print(type(p_dict))
    player_list.append(p_dict)


df = pd.DataFrame(player_list)
df['player_id'] = df['overview_url'].str.split('/').str[-1].str.split('.').str[0]
player_ids = df['player_id'].tolist()
df.head()

################################################################################
'''
Collect Per Game, Per 36, Per 100, and Advanced Statics for each player
'''
################################################################################
# Per Game
counter = 0
df_pergame = pd.DataFrame()
start_time = time.time()
for player in player_ids:
    try:
        p_df = pb.get_player(player, "per_game")
        p_df['player_id'] = player
        df_pergame = pd.concat([df_pergame, p_df], sort=False)
        print(counter)
        print(df_pergame.shape)
        print(round(float(time.time() - start_time), 2))
        print('\n')
        counter += 1
    except Exception as e:
        pass
print(df_pergame.shape)
df_pergame.head()
# df_pergame.to_pickle('/Users/utaveras/FlatironSchool/DS-042219/Mod5_Project/df_pergame.pkl')
end_time = time.time()
print(round(float(end_time - start_time), 2))
# previous run shape = (18967, 31)
# print([type(c) for c in df_pergame.columns])

# Per 36
counter = 0
df_per36 = pd.DataFrame()
start_time = time.time()
for player in player_ids:
    try:
        p_df = pb.get_player(player, "per_minute")
        p_df['player_id'] = player
        df_per36 = pd.concat([df_per36, p_df], sort=False)
        print(counter)
        print(df_per36.shape)
        print(round(float(time.time() - start_time), 2))
        print('\n')
        counter += 1
    except Exception as e:
        pass
print(df_per36.shape)
df_per36.head()
# df_per36.to_pickle('/Users/utaveras/FlatironSchool/DS-042219/Mod5_Project/df_per36.pkl')
end_time = time.time()
print(round(float(end_time - start_time), 2))


# Per 100
counter = 0
df_per100 = pd.DataFrame()
start_time = time.time()
for player in player_ids:
    try:
        p_df = pb.get_player(player, "per_poss")
        p_df['player_id'] = player
        df_per100 = pd.concat([df_per100, p_df], sort=False)
        print(counter)
        print(df_per100.shape)
        print(round(float(time.time() - start_time), 2))
        print('\n')
        counter += 1
    except Exception as e:
        pass
print(df_per100.shape)
df_per100.head()
# df_per100.to_pickle('/Users/utaveras/FlatironSchool/DS-042219/Mod5_Project/df_per100.pkl')
end_time = time.time()
print(round(float(end_time - start_time), 2))


# Advanced
counter = 0
df_adv = pd.DataFrame()
start_time = time.time()
for player in player_ids:
    try:
        p_df = pb.get_player(player, "advanced")
        p_df['player_id'] = player
        df_adv = pd.concat([df_adv, p_df], sort=False)
        print(counter)
        print(df_adv.shape)
        print(round(float(time.time() - start_time), 2))
        print('\n')
        counter += 1
    except Exception as e:
        pass
print(df_adv.shape)
df_adv.head()
# df_adv.to_pickle('/Users/utaveras/FlatironSchool/DS-042219/Mod5_Project/df_adv.pkl')
end_time = time.time()
print(round(float(end_time - start_time), 2))


### Clean up dataframe datatypes and pickle
# Per Game
df_pergame = df_pergame.apply(pd.to_numeric, errors='ignore')
str_cols = df_pergame.select_dtypes(exclude='number').columns.tolist()
str_cols
df_pergame[str_cols] = df_pergame[str_cols].astype(str)
df_pergame.info()
df_pergame.columns = [str(c) for c in df_pergame.columns]
df_pergame.to_pickle('df_pergame.pkl')

# Per 36
df_per36 = df_per36.apply(pd.to_numeric, errors='ignore')
str_cols = df_per36.select_dtypes(exclude='number').columns.tolist()
str_cols
df_per36[str_cols] = df_per36[str_cols].astype(str)
df_per36.info()
df_per36.columns = [str(c) for c in df_per36.columns]
df_per36.to_pickle('df_per36.pkl')

# Per 100
df_per100 = df_per100.apply(pd.to_numeric, errors='ignore')
str_cols = df_per100.select_dtypes(exclude='number').columns.tolist()
str_cols
df_per100[str_cols] = df_per100[str_cols].astype(str)
df_per100.info()
df_per100.columns = [str(c) for c in df_per100.columns]
df_per100.to_pickle('df_per100.pkl')

# advanced
df_adv = df_adv.apply(pd.to_numeric, errors='ignore')
str_cols = df_adv.select_dtypes(exclude='number').columns.tolist()
str_cols
df_adv[str_cols] = df_adv[str_cols].astype(str)
df_adv.info()
df_adv.columns = [str(c) for c in df_adv.columns]
df_adv.to_pickle('df_adv.pkl')

### Fill in NaN values ###
# Per Game
df_pergame.update(df_pergame.select_dtypes(include=[np.number]).fillna(0))
# msno.matrix(df_pergame)

# Per36
df_per36.update(df_per36.select_dtypes(include=[np.number]).fillna(0))
# msno.matrix(df_per36)

# Per100
df_per100.update(df_per100.select_dtypes(include=[np.number]).fillna(0))
# msno.matrix(df_per100)

# Advanced
df_adv.update(df_adv.select_dtypes(include=[np.number]).fillna(0))
# msno.matrix(df_adv)
df_pergame.T.head()


# Create new base player dataframe to be used for merging all player data
df_player = df[['player_id', 'name', 'height', 'weight']]
df_player['height'] = df_player['height'].apply(lambda x: height_to_inches(x))
df_player.rename(columns = {'height': 'height_in', 'weight' : 'weight_lbs', 'name': 'player_name'}, inplace = True)

# DataFrame Column Rename
dfpg_old = df_pergame.select_dtypes(include=[np.number]).columns.tolist()
dfpg_new = ['{}_pergm'.format(x) for x in dfpg_old]
dfpg_new = [x.replace('%', 'Pct') for x in dfpg_new]
dfpg_cols = dict(zip(dfpg_old, dfpg_new))
df_pergame.rename(columns=dfpg_cols, inplace=True)
df_pergame.drop(columns=['Lg'], inplace=True)
join_cols = df_pergame.select_dtypes(exclude=[np.number]).columns.tolist()
join_cols.remove('Pos')
df_pergame.head(1)


# Merge Season, Team, and Position data
dfpg_ocols = ['Age_pergm', 'G_pergm', 'GS_pergm', 'MP_pergm']
dfpg_ncols = [x.replace('_pergm', '') for x in dfpg_ocols]
dfpg_ccols = dict(zip(dfpg_ocols, dfpg_ncols))
dfpg_ccols
df_pergame.rename(columns=dfpg_ccols, inplace=True)
merge_cols = ['Age',  'G',  'GS',  'MP',  'Pos',  'Tm', 'Season', 'player_id']
df_pergame[merge_cols].head()

###
'''
ADD CODE TO Calculate Season# to be used later
Idea: create Function
Steps:
1) Create final dataframe with all Stat values
2) Drop unnecessary columns
3) Group by [player_id, Season] and aggregate columns appropriately. ex. sum -> 'G', 'GS' | recalculate -> 'MP'
1) Create sorted list of Season
'''
###

# Sanity Checks
df_pergame.select_dtypes(exclude=[np.number]).head(1)
df_per36.select_dtypes(exclude=[np.number]).head(1)
df_per100.select_dtypes(exclude=[np.number]).head(1)
df_adv.select_dtypes(exclude=[np.number]).head(1)

# Per100 Cleanup
df100_cols = df_per100.select_dtypes(exclude=[np.number]).columns.tolist()
exclude_cols = ['Season', 'Tm', 'Lg', 'Pos', 'player_id']
df100_cols = [x for x in df100_cols if x not in exclude_cols]
df100_cols
df_per100[df100_cols] = df_per100[df100_cols].apply(pd.to_numeric, errors='coerce')
df_per100.update(df_per100.select_dtypes(include=[np.number]).fillna(0))
df_per100.isnull().sum()
df_per100.select_dtypes(exclude=[np.number]).head(1)


### Create New Dataframe for merging ###
df_pgmg = df_pergame[['player_id', 'Age', 'G', 'MP', 'Pos', 'Tm', 'Season']]
df_pgmg['season_start'] = df_pgmg['Season'].apply(lambda x: season_start_end(x, 'start'))
# Dedup Season Values by keeping only 'Tm' = 'TOT' where exists
df_pgmg.shape
df_pgmg = df_pgmg.sort_values(['player_id', 'season_start', 'G'], ascending=[True, True, False]).drop_duplicates(subset=['player_id', 'season_start'], keep='first')
df_pgmg.shape
df_pgmg.reset_index(drop=True, inplace=True)
df_pgmg.head(10)

# Distinct Count of Seasons Played
df_pergame[(df_pergame['player_id']) == 'abdelal01'].groupby('Season')['Season'].nunique().sum()

### Add season_num column to dataframe to use for comps
df_pgmg['season_num'] = df_pgmg.apply(lambda x: get_season_num(x['player_id'], x['season_start']), axis =1)
dfmp_merge = df_pgmg[['player_id', 'Age', 'Pos', 'season_start', 'season_num', 'G', 'MP']]
df_player = df_player.merge(dfmp_merge, on='player_id', how='left')
df_player.shape
df_player.head()

### Modify df_adv to add season_start and merge necessary attrs to df_player
### target_cols = ['PER', 'WS', 'WS/48', 'BPM', 'VORP']
# Add season_start
df_adv['season_start'] = df_adv['Season'].apply(lambda x: season_start_end(x, 'start'))
dfadv_merge = df_adv[['player_id', 'season_start', 'Tm','G','PER', 'WS', 'WS/48', 'BPM', 'VORP']]
dfadv_merge.shape
dfadv_merge = dfadv_merge.sort_values(['player_id', 'season_start', 'G'], ascending=[True, True, False]).drop_duplicates(subset=['player_id', 'season_start'], keep='first')
dfadv_merge.shape
dfadv_merge.drop(columns=['Tm', 'G'], inplace=True)
df_player = df_player.merge(dfadv_merge, on=['player_id', 'season_start'], how='left')
df_player.head()

# Check for Missing Advanced Stats
missing_stats = df_player[df_player.isnull().any(axis=1)]['player_id'].value_counts().index.tolist()
missing_stats
# Get Advanced Stats for missing players
counter = 0
dfadv_missing = pd.DataFrame()
start_time = time.time()
for player in missing_stats:
    try:
        p_df = pb.get_player(player, "advanced")
        p_df['player_id'] = player
        dfadv_missing = pd.concat([dfadv_missing, p_df], sort=False)
        print(counter)
        print(dfadv_missing.shape)
        print(round(float(time.time() - start_time), 2))
        print('\n')
        counter += 1
    except Exception as e:
        pass
print(dfadv_missing.shape)
dfadv_missing.head()
# dfadv_missing.to_pickle('/Users/utaveras/FlatironSchool/DS-042219/Mod5_Project/dfadv_missing.pkl')
end_time = time.time()
print(round(float(end_time - start_time), 2))
dfadv_missing.head()
pb.get_player('duncati01', "advanced")
pb.get_player(missing_stats[0], 'advanced')

# Unable to pull stats for 6 players via scraper thus will drop rows from dataframe
missing_stats
df_player.shape
df_player[df_player['player_id'].isin(missing_stats)].shape
df_player = df_player[~df_player['player_id'].isin(missing_stats)]
df_player.shape

################################################################################
'''
                        Pickle df_player dataframe
'''
################################################################################
# df_player.to_pickle('df_player.pkl')
# df_player = pd.read_pickle('df_player.pkl')

###################### Sanity Checks ######################

# # Dirk Nowitzki
# df_base[(df_base['player_id']) == 'nowitdi01']['season_num']
# df_player[(df_player['player_id']) == 'nowitdi01']['WS/48'].nlargest(seasons_length).round(3).tolist()
#
# # Larry Bird
# df_player[(df_player['player_id']) == 'birdla01']['season_num'].max()
# df_player[(df_player['player_id']) == 'birdla01']['WS/48'].nlargest(seasons_length)


################################################################################
'''
                    Data Wrangling Portion of Notebook Below
'''
################################################################################

# Import Libraries
import gc
import sys
# sys.path.remove('/Users/utaveras/FlatironSchool/DS-042219/Mod4/Mod4_Project')
sys.path.append('/Users/utaveras/FlatironSchool/DS-042219/Mod5_Project')
sys.path.append('/Library/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages/html5lib')
import csv
import json
import time
import re
import os
import timeit as timeit
from pathlib import Path
import pandas as pd
from functools import reduce
import operator
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
%matplotlib inline
plt.style.use('seaborn-white')

# Fuzzy Match Libraries
import importlib
import d6tjoin.top1
importlib.reload(d6tjoin.top1)
import d6tjoin.utils

# Adjust Pandas Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Check working directory and change if unnecessary
os.chdir('/Users/utaveras/FlatironSchool/DS-042219/Mod5_Project')
os.getcwd()

# Import Data
df1 = pd.read_csv('DataFiles/nba_salaries_1985to2018.csv')
df1.head()
df = pd.read_pickle('DataFiles/df_player.pkl')
df.shape
df.head()
df_xref = df.loc[df.season_start == 2018][['player_id', 'player_name', 'season_start']].drop_duplicates().reset_index(drop=True)
df_xref.head()
df2 = pd.read_csv('DataFiles/nba_salaries_2018to2019.csv')

# Regex to convert string salaries to int
df2['salary'] = df2[df2.columns[-1]].replace('[\$,]', '', regex=True).astype(int)
df2.drop(columns=['Rk', '2018-19'], inplace=True)
df2['league'] = 'NBA'
df2['season'] = '2018-19'
df2['season_start'] = 2018
df2['season_end'] = 2019
teams_abr = df2['Tm'].unique().tolist()
teams_abr = sorted(teams_abr)
len(teams_abr)

teams_long = df1.loc[df1.season_end == 2018]['team'].unique().tolist()
teams_long = sorted(teams_long)
len(teams_long)
teams_dict = dict(zip(teams_abr, teams_long))
teams_dict = {'ATL': 'Atlanta Hawks',
              'BOS': 'Boston Celtics',
              'BRK': 'Brooklyn Nets',
              'CHO': 'Charlotte Hornets',
              'CHI': 'Chicago Bulls',
              'CLE': 'Cleveland Cavaliers',
              'DAL': 'Dallas Mavericks',
              'DEN': 'Denver Nuggets',
              'DET': 'Detroit Pistons',
              'GSW': 'Golden State Warriors',
              'HOU': 'Houston Rockets',
              'IND': 'Indiana Pacers',
              'LAC': 'Los Angeles Clippers',
              'LAL': 'Los Angeles Lakers',
              'MEM': 'Memphis Grizzlies',
              'MIA': 'Miami Heat',
              'MIL': 'Milwaukee Bucks',
              'MIN': 'Minnesota Timberwolves',
              'NOP': 'New Orleans Pelicans',
              'NYK': 'New York Knicks',
              'OKC': 'Oklahoma City Thunder',
              'ORL': 'Orlando Magic',
              'PHI': 'Philadelphia 76ers',
              'PHO': 'Phoenix Suns',
              'POR': 'Portland Trail Blazers',
              'SAC': 'Sacramento Kings',
              'SAS': 'San Antonio Spurs',
              'TOR': 'Toronto Raptors',
              'UTA': 'Utah Jazz',
              'WAS': 'Washington Wizards'}

teams_data = {k:v for k, v in enumerate(teams_dict.items())}
teams_data
df_teams = pd.DataFrame.from_dict(teams_data, orient='index', columns=['tm','team'])
df_teams.head()
##
'''
NBA PLAYER SALARIES CODE BLOCK
'''
##
df2 = df2.merge(df_teams, how='left', left_on='Tm', right_on='tm')
df2.rename(columns={'Player':'player_name'}, inplace=True)
df2.head(1)

# Remove Special Characters
df_xref['player_name'] = df_xref['player_name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df_xref.head(1)
target_cols = df1.columns.tolist()

# Fuzzy Merge dataframes to add 'player_id' column
d6tjoin.utils.PreJoin([df2,df_xref],['player_name']).stats_prejoin(print_only=False)

result = d6tjoin.top1.MergeTop1(df2,df_xref,fuzzy_left_on=['player_name'],fuzzy_right_on=['player_name'], top_limit=[2]).merge()
result['top1']['player_name'].sort_values(by=['__top1diff__'], ascending=False).head(10)
result['top1']['player_name'].head()
result['merged'].shape
df_2018 = result['merged'][target_cols]
df_2018.shape
df1.shape
df1.shape[0] + df_2018.shape[0]
df_final = pd.concat([df1,df_2018])
df_final.shape
df_final = df_final.sort_values(by=['player_id','season_start'], ascending=True).reset_index(drop=True)

# # Export/Archive NBA Salary Data
# df_final.to_csv('DataFiles/nba_salaries_1985to2019_rev.csv')
# df_final.to_pickle('DataFiles/nba_salaries_1985to2019_rev.pkl')

##
'''
NBA YEARLY SALARY CAP CODE BLOCK
'''
##
df_sc = pd.read_csv('DataFiles/NBA_SalaryCapByYear_1984to2028.csv')
df_sc.rename(columns={'Salary Cap':'Salary_Cap'}, inplace=True)
df_sc['Salary_Cap'] = df_sc[df_sc.columns[-1]].replace('[\$,]', '', regex=True).astype(int)
df_sc.head()

# Export clean Historic Salary Cap figures to CSV
df_sc.to_csv('DataFiles/NBA_SalaryCapByYear_1984to2028_clean.csv')


# Lookup Queries
df2[df2['player_name'].str.contains('Abrines')]
df_xref[df_xref['player_name'].str.contains('Zach')]
