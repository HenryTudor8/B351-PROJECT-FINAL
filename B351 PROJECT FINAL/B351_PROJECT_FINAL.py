
import csv
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#GIT HUB TEST

# STEP 1: COMBINING ALL 39 CSV FILES FROM 1985 TO 2023 FOR ATP TOUR LEVEL SINGLES MATCHES INTO ONE DATAFRAME; DATAFRAME IS PROVIDED BY PANDAS LIBRARY
folder_path = r"C:\\Users\\sanch\\OneDrive\\Documents\\SEM 5\\CSCI-B 351\\B351 PROJECT DATA FILES"
def load_and_combine_csvs(folder_path):
    #Listing all the CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    all_dataframes = []

    # Read each file and append to the list
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        # DATAFRAME VARIABLE IS df
        df = pd.read_csv(file_path)
        all_dataframes.append(df)

    #Combining all dataframes into 1 singular dataframe
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    return combined_df


# PREPROCESSING THE NEW COMINED DATA FRAME AND ADDING FEATURES
def preprocess_data(combined_data):
    required_columns = ['winner_rank', 'loser_rank', 'winner_name', 'loser_name', 'surface'] #checks if all the columns are available or not
    missing_columns = [col for col in required_columns if col not in combined_data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


    # Dropping irrelevant columns from the combined dataframe
    columns_to_drop = ['draw_size', 'tourney_name', 'tourney_id', 'match_num', 'winner_id', 'loser_id']
    combined_data = combined_data.drop(columns=columns_to_drop, errors='ignore')

    #FILLING OR DROPPING THE MISSING VALUES
    #combined_data = combined_data.dropna(subset=['winner_rank', 'loser_rank']) #completely dropping the rows where either winner rank or loser rank is missing
    combined_data['surface'] = combined_data['surface'].fillna('Unknown') #replacing surface with the string unknown if the surface name is unknown

    #PRESERVED ORIGINAL SURFACE  COLUMN
    combined_data['original_surface'] = combined_data['surface'].copy()
    print("Columns after preserving original_surface:", combined_data.columns)
    #ADDING PLAYER COLUMNS TO THE DATAFRAME AND ADDING A TARGET VARIABLE
    combined_data['player_1'] = combined_data['winner_name']
    combined_data['player_2'] = combined_data['loser_name']
    combined_data['label'] = 1 #indicates the Player 1 (winner) won the match
 

    #FEATURES
    #FEATURE 1: SURFACE TYPE I.E GRASS, CLAY, HARD ETC. 
    combined_data = pd.get_dummies(combined_data, columns=['surface'], drop_first=True) #this function turns the type of surface into 1 or 0, surface is hard if both clay and grass values are 0
    print("Columns after applying pd.get_dummies:", combined_data.columns)
    
    # Calculate surface-specific win rates for all surfaces
    #FEATURE 2: SURFACE SPECIFIC WIN RATE
    print("Columns in combined_data right before groupby:", combined_data.columns) # to confirm whether the surface column is presnet at this point
   # print(combined_data['surface'].unique())
   # print(combined_data['surface'].isnull().sum())
    
    
    for surface in ['Grass', 'Clay', 'Hard', 'Carpet']:  # Using original surface names
        combined_data[f'player_1_{surface.lower()}_win_rate'] = (
        combined_data.groupby(['player_1', 'original_surface'])['label']
        .transform('mean')
        .where(combined_data['original_surface'] == surface)
        )
        combined_data[f'player_2_{surface.lower()}_win_rate'] = (
        combined_data.groupby(['player_2', 'original_surface'])['label']
        .transform('mean')
        .where(combined_data['original_surface'] == surface)
    )

    # Fill missing values for players with no matches on that surface
    combined_data[f'player_1_{surface.lower()}_win_rate'] = combined_data[f'player_1_{surface.lower()}_win_rate'].fillna(0.5)

    combined_data[f'player_2_{surface.lower()}_win_rate'] = combined_data[f'player_2_{surface.lower()}_win_rate'].fillna(0.5)


    #FEATURE 3: RANKING DIFFERENCE
    combined_data['rank_diff'] = combined_data['winner_rank'] - combined_data['loser_rank']

    #FEATURE 4: RECENT FORM LIKE WIN STREAKS OR LOSING STREAKS, FOR THIS FEATURE TO WORK THE AI MODEL WILL BE TOLD WHAT TIME OR THE YEAR 2 PLAYERS WOULD HAVE FACED EACH OTHER
    combined_data['player_1_recent_form'] = combined_data.groupby('player_1')['label'].rolling(10).mean().reset_index(0, drop=True) #looks at the last 10 matches of the player and calculate the mean of win percentage
    combined_data['player_2_recent_form'] = combined_data.groupby('player_2')['label'].rolling(10).mean().reset_index(0, drop=True) #same as above for player_1
    combined_data['player_1_recent_form'] = combined_data['player_1_recent_form'].fillna(0.5) # if player has played than less than 10 matches, the missing values for win percentages is filled with 0.5
    combined_data['player_2_recent_form'] = combined_data['player_2_recent_form'].fillna(0.5) # same as above


    # HEAD TO HEAD RECORD

    combined_data['date'] = pd.to_datetime(combined_data['tourney_date'], errors='coerce') # puts the data in a chronological order so H2H is meaningful
    combined_data = combined_data.sort_values(by='date')

    def calculate_head_to_head(combined_data):
        head_to_head = {} # A dictionary to store H2H stats between 2 players
        head_to_head_results = [] # list to store computed win rates for player 1

        for _, row in combined_data.iterrows():
            p1, p2 = row['player_1'], row['player_2']
            key = tuple(sorted([p1, p2]))

            if key not in head_to_head:
                head_to_head[key] = {'wins_p1' : 0, 'wins_p2': 0}

            if row['label'] == 1: # label ==1 indicates Player 1 won the match
                head_to_head[key]['wins_p1'] = head_to_head[key]['wins_p1'] + 1
            else:
                head_to_head[key]['wins_p2'] = head_to_head[key]['wins_p2'] + 1

            total_matches = head_to_head[key]['wins_p1'] + head_to_head[key]['wins_p2']
            if total_matches > 0:
                win_rate_p1 = head_to_head[key]['wins_p1']/total_matches #Only player 1 win rate is calculated because player 1 and 2 switches throughout the new dataframe based on who won the match, decreases redundancy
            else:
                win_rate_p1 = 0.5
            head_to_head_results.append(win_rate_p1)

        return head_to_head_results

    combined_data['head_to_head'] = calculate_head_to_head(combined_data)

    print("Columns after preprocessing:", combined_data.columns)
    print("Original surface column unique values:", combined_data['original_surface'].unique())


    #Normalizing features
    scaler = MinMaxScaler() #transforms the features by scaling them to a range of 0 to 1
    features_to_scale = [
    'rank_diff', 
    'player_1_recent_form', 
    'player_2_recent_form',
    'player_1_grass_win_rate', 
    'player_2_grass_win_rate',
    'player_1_clay_win_rate', 
    'player_2_clay_win_rate',
    'player_1_hard_win_rate', 
    'player_2_hard_win_rate',
    
] # H2H not scaled because those values are already between 0 and 1
    combined_data[features_to_scale] = scaler.fit_transform(combined_data[features_to_scale]) #learns the min and max values and scales accordingly

    return combined_data

# STEP 3: SPLITTING DATA INTO TRAINING AND TESTING SETS
def split_data(combined_data):
    features = ['rank_diff', 'player_1_recent_form', 'player_2_recent_form', 
                'player_1_grass_win_rate', 'player_2_grass_win_rate',
                'player_1_clay_win_rate', 'player_2_clay_win_rate',
                'player_1_hard_win_rate', 'player_2_hard_win_rate', 
     'head_to_head']
    x = combined_data[features]
    y = combined_data['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) #42 is arbitrary values to maintain consistency throughout various runs
    return x_train, x_test, y_train, y_test


# Step 4: Main Workflow
def main():
    folder_path = r"C:\\Users\\sanch\\OneDrive\\Documents\\SEM 5\\CSCI-B 351\\B351 PROJECT DATA FILES" 
    combined_data = load_and_combine_csvs(folder_path)
    print("Coulumns in combined data:", combined_data.columns)
    if 'surface' not in combined_data.columns:
        raise ValueError("The column 'surface' is missing from the dataset")
    preprocessed_data = preprocess_data(combined_data)
    x_train, x_test, y_train, y_test = split_data(preprocessed_data)

    # Save preprocessed data to a CSV file for later use
    preprocessed_data.to_csv('cleaned_tennis_data.csv', index=False)
    print("Data preprocessing complete. Training and testing sets are ready.")
    print(f"Training samples: {len(x_train)}, Testing samples: {len(x_test)}")

if __name__ == "__main__":
    main()




    