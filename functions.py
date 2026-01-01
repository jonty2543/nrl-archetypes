import pandas as pd
import numpy as np
import ENVIRONMENT_VARIABLES as EV
from collections import defaultdict
import re
import itertools
from supabase import create_client, Client

SUPABASE_URL = "https://glrzwxpxkckxaogpkwmn.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imdscnp3eHB4a2NreGFvZ3Brd21uIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1NjA3OTU3NiwiZXhwIjoyMDcxNjU1NTc2fQ.YOF9ryJbhBoKKHT0n4eZDMGrR9dczR8INHVs_By4vRU"
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def player_data_cleaner(df):
    
    df.columns = df.columns.str.replace('\xa0', ' ')
    
    df = df.replace('-', '0')  
    df['Kicking Metres'] = df['Kicking Metres'].str.rstrip('%')  
    df['Dummy Passes'] = df['Dummy Passes'].str.rstrip('s')  
    
    def time_to_float(time_str):
        if not time_str or time_str == '0':
            return 0.0
        try:
            minutes, seconds = map(int, time_str.split(":"))
            return minutes + seconds / 60
        except Exception as e:
            print(f"Error processing {time_str}: {e}")
            return None
    
    df['Mins Played'] = df['Mins Played'].apply(time_to_float)
    df['Stint One'] = df['Stint One'].apply(time_to_float)
    df['Stint Two'] = df['Stint Two'].apply(time_to_float)
    
    df.columns
    
    for col, col_type in EV.types_to_convert.items():
        if col_type == 'drop':
            df = df.drop(columns=[col])
        elif col_type == 'string':
            df[col] = df[col].astype('string')
        elif col_type == 'category':
            df[col] = df[col].astype('category')
        elif col_type == 'int':
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int')
        elif col_type == 'float':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.reset_index(drop=True)

    df['Date_dt'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date_dt', ascending=True)

    df['days_since_last'] = df.groupby('Player')['Date_dt'].diff().dt.days
    df['days_since_last'] = df['days_since_last'].fillna(10)

    # Cap the value at 16
    df['days_since_last'] = df['days_since_last'].clip(upper=10)
    
    df = df.drop(columns=['Date_dt'])

    return df


def clean_match_data(df):
    
    #df = df[df['Date'] > '2016-01-01']
    
    for team in ['Home', 'Away']:
            
        def time_to_float(time_str):
            if not time_str or time_str == '0':
                return 0.0
            try:
                minutes, seconds = map(int, time_str.split(":"))
                return minutes + seconds / 60
            except Exception as e:
                print(f"Error processing {time_str}: {e}")
                return None
        
        df[f'{team}TimeinPossession'] = df[f'{team}TimeinPossession'].apply(time_to_float)
        
        for stat in ['FieldGoals', 'PenaltyGoals', 'Conversions']:
            col_formatted = f'{team}{stat}'
            
            # Convert values like '2' to '2/2', handle NaNs as '0/0'
            df[col_formatted] = df[col_formatted].replace('0', '0/0').fillna('0/0')   
            
            # Handle bare integers like '2' (convert to '2/2'), and zeros like '0' to '0/0'
            df[col_formatted] = df[col_formatted].apply(
                lambda x: f'{int(x)}/{int(x)}' if str(x).isdigit() else ('0/0' if str(x) == '0' else str(x))
            )
           
            # Split into Made and Attempted
            df[[f'{team}{stat}Made', f'{team}{stat}Attempted']] = df[col_formatted].fillna('0/0').str.split('/', expand=True).fillna(0).astype(int)
        
        for stat in ['ForcedDropOuts', 'Intercepts', 'OnReports', 'SinBins', 'HeadInjuryAssessment']:
            df[f'{team}{stat}'] = df[f'{team}{stat}'].fillna(0)
    
    cols_to_drop = [
        col for col in df.columns
        if 'AwayPlayer' in col or 'HomePlayer' in col
    ]
    
    return df.drop(columns=cols_to_drop)

def create_features(df):
    df['HomeWin'] = np.where(df['HomeScore'] > df['AwayScore'], 1, 0)
    df['HomeWin'] = np.where(df['HomeScore'] == df['AwayScore'], 0.5, df['HomeWin'])
    
    df['AwayWin'] = np.where(df['AwayScore'] > df['HomeScore'], 1, 0)
    df['AwayWin'] = np.where(df['AwayScore'] == df['HomeScore'], 0.5, df['AwayWin'])
    
    df['HomeMargin'] = df['HomeScore'] - df['AwayScore']
    df['AwayMargin'] = df['AwayScore'] - df['HomeScore']
    
    def extract_round(round_str):
        if 'Round' in str(round_str):
            match = re.search(r'\d+', str(round_str))
            if match:
                return int(match.group())
        return 0  # return original string if no "Round" or no number
    
    df['RoundNumber'] = df['Round'].apply(extract_round)
    
    def get_k(round_number):
        if 1 <= round_number <= 6:
            return 135
        elif 7 <= round_number <= 12:
            return 130
        elif 13 <= round_number <= 19:
            return 105
        elif 20 <= round_number <= 26:
            return 200
        else:
            return 145  # finals
    
    
    def regress_rating_to_mean(rating, base=1500, factor=0.2):
        return base + factor * (rating - base)

    def elo_rating_system(df, p=0.0601, base_rating=1500):
        ratings = defaultdict(lambda: base_rating)
        rating_history = []
              
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
    
        df = df.sort_values('Date')  # Make sure games are in chronological order
        
        final_ratings_by_year = {}
        teams = pd.unique(df[['HomeTeam', 'AwayTeam']].values.ravel())
    
        for _, row in df.iterrows():
            year = row['Date'].year

            # Start of a new season → apply regression to previous year's ratings
            if year not in final_ratings_by_year:
                if final_ratings_by_year:  # not the first season
                    prev_year = max(final_ratings_by_year.keys())
                    for team in teams:
                        prev_rating = final_ratings_by_year[prev_year].get(team, base_rating)
                        ratings[team] = regress_rating_to_mean(prev_rating, base=base_rating, factor=0.2)
                final_ratings_by_year[year] = {}
            
            home = row['HomeTeam']
            away = row['AwayTeam']
            round_num = row['RoundNumber']
            margin = row['HomeMargin'] - row['AwayMargin']
            k = get_k(round_num)
    
            home_rating = ratings[home]
            away_rating = ratings[away]
    
            # Apply home advantage
            dr = (home_rating + 24) - (away_rating - 24)
            we_home = 1 / (1 + 10 ** (-dr / 400))
            we_away = 1 - we_home
    
            # Actual outcome scaled by margin
            w_home = 1 / (1 + np.exp(-p * margin))
            w_away = 1 - w_home
    
            # Rating updates
            new_home_rating = home_rating + k * (w_home - we_home)
            new_away_rating = away_rating + k * (w_away - we_away)
    
            ratings[home] = new_home_rating
            ratings[away] = new_away_rating
 
            rating_history.append({
                'Date': row['Date'],
                'Round': row['Round'],
                'HomeTeam': home,
                'AwayTeam': away,
                'HomeRating': home_rating,
                'AwayRating': away_rating,
                'NewHomeRating': new_home_rating,
                'NewAwayRating': new_away_rating,
                'Margin': margin,
                'K': k,
                'Year': year,
            })
    
            # Save final ratings for future preseason regression
            final_ratings_by_year[year][home] = new_home_rating
            final_ratings_by_year[year][away] = new_away_rating
        
   
        return pd.DataFrame(rating_history)
    
    elo_results = elo_rating_system(df)
    elo_results['Date'] = elo_results['Date'].dt.date
    df['Date'] = pd.to_datetime(df['Date']).dt.date

    # Step 2: Merge the ratings back into your match_data
    df = df.merge(
        elo_results[['Date', 'HomeTeam', 'AwayTeam', 'HomeRating', 'AwayRating', 'NewHomeRating', 'NewAwayRating']],
        on=['Date', 'HomeTeam', 'AwayTeam'],
        how='left'
    )
    
    df['HomePred'] = np.where(df['HomeRating'] > df['AwayRating'], 1, 0)
    
    df['Date'] = pd.to_datetime(df['Date'], yearfirst=True, errors='coerce')
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    return df

def model_dataset(df): 
    
    df = df.drop(columns=['Home Team', 'Away Team'])
        
    df = df[df['HomeWin'] != 0.5]
    df = df[df['AwayWin'] != 0.5]
    
    # Common metadata columns (keep the same for both perspectives)
    meta_cols = ['URL', 'Date', 'Round', 'RoundNumber', 'Ground Conditions']
    
    # Columns to transform for home and away
    home_cols = [col for col in df.columns if col.startswith('Home')]
    away_cols = [col for col in df.columns if col.startswith('Away')]
    all_cols = home_cols + away_cols
    
    # Strip the prefix to get unified column names
    home_renamed = {col: col.replace('Home ', '').replace('Home', '', 1).replace('Away ', 'Opponent').replace('Away', 'Opponent', 1) for col in all_cols}
    away_renamed = {col: col.replace('Away ', '').replace('Away', '', 1).replace('Home ', 'Opponent').replace('Home', 'Opponent', 1) for col in all_cols}
    #opponent_renamed = {col: col.replace('Away ', '').replace('Away', '', 1) for col in away_cols}
    
    # Create home perspective rows
    home_df = df[meta_cols + home_cols + away_cols]
    home_df = home_df.rename(columns=home_renamed)
    #home_df = home_df.rename(columns={'AwayTeam': 'Opponent', 'AwayRating': 'OpponentRating'})
    home_df['IsHome'] = 1
    
    # Create away perspective rows
    away_df = df[meta_cols + away_cols + home_cols]
    away_df = away_df.rename(columns=away_renamed)
    #away_df = away_df.rename(columns={'HomeTeam': 'Opponent', 'HomeRating':'OpponentRating'})
    away_df['IsHome'] = 0
    
    
    print(home_df.columns[home_df.columns.duplicated()])
    print(away_df.columns[away_df.columns.duplicated()])
        
    # Combine
    long_df = pd.concat([home_df, away_df], ignore_index=True)
    
    return long_df

def generate_features(df, ROLLING_WINDOWS):

    FEATURES_T = [
        "Margin", "Win", "Possession", "AverageSetDistance",
        "KickReturnMetres", "Odds"
        
    ]
    AGGREGATES  = ["min", "max", "mean", "median", "std"]

    # ------------------------ SET-UP ------------------------
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])            # ensure datetime
    df = df.set_index(["Team", "Date"]).sort_index()

    feature_cols = []

    # =======================================================
    # 1⃣  TEAM-LEVEL ROLLING FEATURES
    # =======================================================
    
    for w in ROLLING_WINDOWS:
        rolling = (
            df.reset_index(level=0)                    # keep Team as a column
              .sort_values(["Team", "Date"])
              .groupby("Team")[FEATURES_T]
              .rolling(window=w, min_periods=1)
              .agg(AGGREGATES)
              .groupby(level=0)                        # back to MultiIndex
              .shift(1)                                # no leakage
        )

        new_cols = [f"{f}_{a}_{w}" for f, a in itertools.product(FEATURES_T, AGGREGATES)]
        df[new_cols] = rolling
        feature_cols.extend(new_cols)

    
    # 2⃣ PLAYER-SPECIFIC ROLLING AVERAGES
    # -- 2a. identify the “Name” columns we want to keep
    pos_cols = [
        col for col in df.columns
        if col.endswith("Player")
    ]
    
    rating_cols = [
        col for col in df.columns
        if col.endswith("Rating_")
    ]
    
    print("Position columns:", pos_cols)
    print("Rating columns:", rating_cols)

    # ---- 2b. Reshape to long: one row per player per match ----
    long_list = []
    df_reset = df.reset_index()
    for pos_col, rating_col in zip(pos_cols, rating_cols):
        if rating_col not in df_reset.columns:
            continue
        tmp = (
            df_reset[["Team", "Date", pos_col, rating_col]]
              .dropna(subset=[pos_col])
              .rename(columns={pos_col: "Player", rating_col: "PlayerRating"})
              .assign(Position=pos_col.replace("1Rating_Player", "1").replace("6Rating_Player", "6").replace("7Rating_Player", "7").replace("9Rating_Player", "9"))
        )
        long_list.append(tmp)
    
    long_df = pd.concat(long_list, ignore_index=True)

    # ---- 2c. Rolling mean over all games per player ----
    for w in ROLLING_WINDOWS:
        col_roll = f"RollingMean_{w}"
        long_df[col_roll] = (
            long_df.sort_values(["Player", "Date"])
                   .groupby("Player")["PlayerRating"]
                   .transform(lambda x: x.shift(1).rolling(window=w, min_periods=1).mean())
        )

        # ---- 2d. Pivot back to wide: one row per Team+Date ----
        pivot = (
            long_df.pivot_table(index=["Team", "Date"], columns="Position", values=col_roll)
                   .rename(columns=lambda x: f"{x}_rating_{w}")
        )
        df = df.join(pivot, how="left")
        feature_cols.extend(pivot.columns.tolist())
    
    # Final Imputation
    df = df.apply(lambda col: col.fillna(col.median()) if col.dtype.kind in "biufc" else col)
    
    return df.reset_index(drop=False), feature_cols

def generate_modelling_dataset(dataset, feature_cols):
    '''
    This function extends the list of feature columns from the generate_rollingFeatures function by adding the features we previously created, and then keeps only the columns specified to prepare the dataset for training by the algorithm
    '''
    
    DATASET_COLUMNS_TO_KEEP = ['URL',
                        'Round',
                        'Date',
                        'Team',
                        'Odds Open',
                        'Odds',
                        'Win',
                        'IsHome',
                        'Rating',
                        'OpponentRating']
    
    # Trim the dataset
    dataset = dataset[DATASET_COLUMNS_TO_KEEP + feature_cols]

    # Extend the feature_cols list
    feature_cols.extend([
                         'Rating',
                         'OpponentRating',
                         'IsHome'
                          ])

    # Fill any missing values with 0 as a final step before training
    dataset.fillna(0,inplace=True)

    return dataset, feature_cols


def get_odds():
    
    url = f'https://www.aussportsbetting.com/historical_data/nrl.xlsx'

    try:
        # Read CSV data into a DataFrame directly from the URL
        odds = pd.read_excel(url, header=1)
        print(f"Processed: odds df")
    except Exception as e:
        print(f"Failed to fetch odds data, Error: {e}")

        
    odds['Date'] = pd.to_datetime(odds['Date'], yearfirst=True, errors='coerce')
    odds['Date'] = odds['Date'].dt.strftime('%Y-%m-%d')
    
    nrl_team_mapping = {
        "Brisbane Broncos": "Broncos",
        "Canberra Raiders": "Raiders",
        "Canterbury Bulldogs": "Bulldogs",
        "Canterbury-Bankstown Bulldogs": "Bulldogs",
        "Manly-Warringah Sea Eagles":"Sea Eagles",
        "Cronulla Sharks": "Sharks",
        "Cronulla-Sutherland Sharks": "Sharks",
        "Gold Coast Titans": "Titans",
        "Manly Sea Eagles": "Sea Eagles",
        "Melbourne Storm": "Storm",
        "Newcastle Knights": "Knights",
        "New Zealand Warriors": "Warriors",
        "North QLD Cowboys": "Cowboys",
        "North Queensland Cowboys": "Cowboys",
        "Parramatta Eels": "Eels",
        "Penrith Panthers": "Panthers",
        "South Sydney Rabbitohs": "Rabbitohs",
        "St George Illawarra Dragons": "Dragons",
        "St George Dragons": "Dragons",
        "St. George Illawarra Dragons": "Dragons",
        "Sydney Roosters": "Roosters",
        "Wests Tigers": "Wests Tigers",
        "Dolphins":"Dolphins",
        "Tigers": "Wests Tigers"
    }
    
    odds['Home Team'] = odds['Home Team'].replace(nrl_team_mapping)
    odds['Away Team'] = odds['Away Team'].replace(nrl_team_mapping)
    
    return odds


def combine_odds(df, odds):
    df = pd.merge(df, odds[['Date', 'Home Team', 'Away Team', 'Home Odds', 'Away Odds', 'Home Odds Open', 'Away Odds Open',
                            'Home Line Open', 'Away Line Open','Home Line Odds Open', 'Away Line Odds Open']], 
                            left_on=['Date', 'HomeTeam', 'AwayTeam'], right_on=['Date', 'Home Team', 'Away Team'], how='left')

    df['HomeLineDiff'] = df['HomeMargin'] + df['Home Line Open']
    df['AwayLineDiff'] = df['AwayMargin'] + df['Away Line Open']
    
    return df



def calculate_log_loss(y_true, y_pred):
    """
    Calculates the log loss.

    Args:
        y_true (array-like): True labels (0 or 1).
        y_pred (array-like): Predicted probabilities (between 0 and 1).

    Returns:
        float: Log loss value.
    """
    # Clip predictions to avoid log(0) errors
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

BACKTESTING_COLUMNS = ['URL', 'Date', 'Round', 'Team', 'Odds Open','Odds', 'Win']

def predictions(best_model,test_data, test_x, model):
    '''
    This function generates a column of probabilities for each runner, before normalising the probabilities across each game.
    '''

    # Predict probabilities using the best model
    test_data[f'win_probability_{model}'] = best_model.predict_proba(test_x)[:, 1]

    # Normalise probabilities across each race
    test_data[f'win_probability_{model}'] = test_data.groupby('URL', group_keys=False)[f'win_probability_{model}'].apply(lambda x: x / sum(x))

    # Keep only required columns
    test_data = test_data[BACKTESTING_COLUMNS + [f'win_probability_{model}']]

    # Export DataFrame to CSV
    #test_data.to_csv('catboost_gridSearch_predictions.csv', index=False)

    return test_data

def transform_draw(df, rd):
    
    df = df[['Round Number', 'Date', 'Home Team', 'Away Team']]
    df = df.rename(columns={'Home Team':'HomeTeam', 'Away Team':'AwayTeam'})
    df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y %H:%M").dt.date.astype('str')
    
    df['URL'] = df.apply(
        lambda row: f"https://www.nrl.com/draw/nrl-premiership/2025/round-{rd}/{row['HomeTeam']}-v-{row['AwayTeam']}/",
        axis=1
    )
    
    df['Round'] = f'Round {rd}'
        
    return df

def fetch_all(table, batch=1000):
    out, i = [], 0
    while True:
        data = supabase.schema("nrl").table(table).select("*").range(i, i+batch-1).execute().data
        if not data: break
        out += data
        i += batch
    return pd.DataFrame(out)

def map_position(n):
    if n == 1:
        return 'Fullback'
    elif n in {2, 5}:
        return 'Winger'
    elif n in {3, 4}:
        return 'Centre'
    elif n in {6, 7}:
        return 'Half'
    elif n == 9:
        return 'Hooker'
    elif n in {8, 10, 13}:
        return 'Middle'
    elif n in {11, 12}:
        return '2nd Row'
    elif n in {14, 15, 16, 17}:
        return 'Interchange'
    else:
        return 'Interchange'


