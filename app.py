#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import poisson
import os
import streamlit as st

# Set the path to the directory containing the CSV files
path = "/Users/allanabala/Desktop/Analytics and Data Science/Sports/Sports Prediction Test/data_premeir_league"

# Get the list of file names in the directory
file_names = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

# Initialize an empty list to store DataFrames
dfs = []

# Read the CSV files into a list of data frames
for file in file_names:
    try:
        df = pd.read_csv(file, encoding='latin-1')
        df.columns = df.columns.str.strip().str.lower()  # Standardize column names
        dfs.append(df)
    except Exception as e:
        print(f"Error reading file {file}: {e}")

# Merge all the DataFrames into a single DataFrame
merged_df = pd.concat(dfs, ignore_index=True)

# Write the merged DataFrame to a CSV file
merged_df.to_csv("merged_file.csv", index=False)

# Import the historical data
premier_data = pd.read_csv("merged_file.csv")

# Preprocess the data
premier_data_clean = premier_data.dropna(subset=['fthg', 'ftag']).astype({'fthg': 'float', 'ftag': 'float'})
premier_data_clean['date'] = pd.to_datetime(premier_data_clean['date'], format='%d/%m/%Y')

# Set the splitting date
split_date = pd.to_datetime("2020-08-13")

# Split the data into training and testing sets
premier_data_train = premier_data_clean[premier_data_clean['date'] < split_date].copy()
premier_data_test = premier_data_clean[premier_data_clean['date'] >= split_date].copy()

# Calculate rolling averages for team form
premier_data_train['home_form'] = (
    premier_data_train.groupby('hometeam')['fthg']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
).fillna(0)

premier_data_train['away_form'] = (
    premier_data_train.groupby('awayteam')['ftag']
    .transform(lambda x: x.rolling(window=5, min_periods=1).mean())
).fillna(0)

# Aggregate the data to get average goals scored, conceded, and form for home and away teams
data_agg_home = premier_data_train.groupby('hometeam').agg(
    avg_home_goals_scored=('fthg', 'mean'),
    avg_home_goals_conceded=('ftag', 'mean'),
    home_form=('home_form', 'mean')
).reset_index()

data_agg_away = premier_data_train.groupby('awayteam').agg(
    avg_away_goals_scored=('ftag', 'mean'),
    avg_away_goals_conceded=('fthg', 'mean'),
    away_form=('away_form', 'mean')
).reset_index()

# Merge aggregated features back into the training data
premier_data_train = premier_data_train.merge(data_agg_home, on='hometeam', how='left')
premier_data_train = premier_data_train.merge(data_agg_away, on='awayteam', how='left')

# Fit Poisson regression models
home_features = ['avg_home_goals_scored', 'avg_home_goals_conceded', 'home_form_y']
away_features = ['avg_away_goals_scored', 'avg_away_goals_conceded', 'away_form_y']

home_goals_fit = sm.GLM(premier_data_train['fthg'], sm.add_constant(premier_data_train[home_features]), family=sm.families.Poisson()).fit()
away_goals_fit = sm.GLM(premier_data_train['ftag'], sm.add_constant(premier_data_train[away_features]), family=sm.families.Poisson()).fit()

# Prepare inputs and outputs for neural network
nn_inputs = premier_data_train[home_features + away_features]
nn_output = premier_data_train[['fthg', 'ftag']]

# Scale inputs for neural network
scaler = StandardScaler()
nn_inputs_scaled = scaler.fit_transform(nn_inputs)

# Train neural network
nn_fit = MLPRegressor(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000)
nn_fit.fit(nn_inputs_scaled, nn_output)

def poisson_expected_goals_nn(home_team, away_team, home_goals_fit, away_goals_fit, data_agg):
    home_team_data = data_agg[data_agg['hometeam'] == home_team]
    away_team_data = data_agg[data_agg['awayteam'] == away_team]

    if home_team_data.empty or away_team_data.empty:
        raise ValueError(f"Team data not found for {home_team} or {away_team}")

    home_goals_avg = home_team_data['avg_home_goals_scored'].values[0]
    away_goals_avg = away_team_data['avg_away_goals_scored'].values[0]
    home_conceded_avg = home_team_data['avg_home_goals_conceded'].values[0]
    away_conceded_avg = away_team_data['avg_away_goals_conceded'].values[0]
    home_form = home_team_data['home_form_y'].values[0]
    away_form = away_team_data['away_form_y'].values[0]

    # Ensure correct number of features for Poisson model
    home_input = sm.add_constant([[home_goals_avg, home_conceded_avg, home_form]], has_constant='add')
    away_input = sm.add_constant([[away_goals_avg, away_conceded_avg, away_form]], has_constant='add')

    home_lambda_poisson = home_goals_fit.predict(home_input)[0]
    away_lambda_poisson = away_goals_fit.predict(away_input)[0]

    # Prepare inputs for neural network
    nn_inputs = pd.DataFrame({
        'avg_home_goals_scored': [home_goals_avg],
        'avg_home_goals_conceded': [home_conceded_avg],
        'home_form_y': [home_form],
        'avg_away_goals_scored': [away_goals_avg],
        'avg_away_goals_conceded': [away_conceded_avg],
        'away_form_y': [away_form]
    })

    # Scale inputs for neural network
    nn_inputs_scaled = scaler.transform(nn_inputs)

    # Predict using neural network
    nn_pred = nn_fit.predict(nn_inputs_scaled)
    home_lambda_nn = nn_pred[0][0]
    away_lambda_nn = nn_pred[0][1]

    # Combine Poisson and Neural Network predictions
    home_lambda = 0.7 * home_lambda_poisson + 0.3 * home_lambda_nn
    away_lambda = 0.7 * away_lambda_poisson + 0.3 * away_lambda_nn

    return {'home_lambda': home_lambda, 'away_lambda': away_lambda}

def poisson_probabilities(home_lambda, away_lambda):
    home_probs = poisson.pmf(range(11), home_lambda)
    away_probs = poisson.pmf(range(11), away_lambda)

    home_win_prob = np.sum(home_probs[:, None] * away_probs[None, :] * (np.arange(11)[:, None] > np.arange(11)[None, :]))
    away_win_prob = np.sum(home_probs[:, None] * away_probs[None, :] * (np.arange(11)[:, None] < np.arange(11)[None, :]))
    draw_prob = np.sum(home_probs[:, None] * away_probs[None, :] * (np.arange(11)[:, None] == np.arange(11)[None, :]))

    # Convert to percentages
    return {
        'home_win_prob': round(home_win_prob * 100, 2),
        'away_win_prob': round(away_win_prob * 100, 2),
        'draw_prob': round(draw_prob * 100, 2)
    }

# Streamlit App
st.title("Premier League Match Predictor")

# Input fields for home and away teams
home_team = st.selectbox("Select Home Team", premier_data_train['hometeam'].unique())
away_team = st.selectbox("Select Away Team", premier_data_train['awayteam'].unique())

if st.button("Predict"):
    # Calculate expected goals
    expected_goals = poisson_expected_goals_nn(home_team, away_team, home_goals_fit, away_goals_fit, premier_data_train)

    # Round expected goals
    expected_goals['home_lambda'] = round(expected_goals['home_lambda'])
    expected_goals['away_lambda'] = round(expected_goals['away_lambda'])

    # Display expected goals
    st.write(f"Expected goals for {home_team}: {expected_goals['home_lambda']}")
    st.write(f"Expected goals for {away_team}: {expected_goals['away_lambda']}")

    # Compute probabilities
    probs = poisson_probabilities(expected_goals['home_lambda'], expected_goals['away_lambda'])

    # Display probabilities
    st.write(f"Probability of {home_team} Win: {probs['home_win_prob']}%")
    st.write(f"Probability of {away_team} Win: {probs['away_win_prob']}%")
    st.write(f"Probability of Draw: {probs['draw_prob']}%")

