# Soccer Prediction
This repository contains a Streamlit-based web application for predicting match outcomes in the English Premier League. The app uses a combination of Poisson regression and neural networks to estimate expected goals and match probabilities. 

# Table of Contents
1. [Overview](#Overview)
2. [How It Works](#how_it_works)
3. [Features](#Features)
4. [Installation](#Installation)
5. [Usage](#Usage)
6. [Technologies Used](#Technologies)
7. [Contributing](#Contributing)
8. [License](#License)

# Overview
The Premier League Match Predictor is a data-driven application that predicts the expected goals and match outcomes for any two teams in the English Premier League. It uses historical match data to train a Poisson regression model and a neural network, combining their predictions to provide results (which might not be always accurate).

# How It Works
1. Data Collection and Preprocessing:
- The app reads multiple CSV files containing historical match data.
- It cleans and preprocesses the data, handling missing values and standardizing column names.
- The data is split into training and testing sets based on a specified date.

2. Feature Engineering:
- Rolling averages for team form (last 5 matches) are calculated.
- Aggregated features like average goals scored, conceded, and team form are computed for both home and away teams.

3. Model Training:
- A Poisson regression model is trained to predict goals scored by home and away teams.
- A neural network (MLPRegressor) is trained to refine the predictions further.

4. Prediction:
- The app takes user inputs (home and away teams) and calculates expected goals using the trained models.
- It combines the predictions from the Poisson model and the neural network to provide final expected goals.

5. Probability Calculation:
- Using the Poisson distribution, the app calculates the probabilities of a home win, away win, or draw.

6. User Interface:
- The app is built with Streamlit, allowing users to select teams and view predictions interactively.

# Features
1. **Dynamic User Interface**: Select home and away teams to get real-time predictions.
2. **Expected Goals**: Displays predicted goals for both teams.
3. **Match Probabilities**: Shows the likelihood of a home win, away win, or draw.

# Installation
## Clone the repository:
- git clone https://github.com/your-username/premier-league-predictor.git
- cd premier-league-predictor
## Install the required dependencies:
- pip install -r requirements.txt
## Run the Streamlit app:
- streamlit run app.py
## Open your browser and navigate to http://localhost:8501 to use the app.

# Usage
1. Select the home team and away team from the dropdown menus.
2. Click the Predict button to get:
- Expected goals for both teams.
- Probabilities of a home win, away win, or draw.

# Technologies
- Python: Primary programming language.
- Pandas: Data manipulation and analysis.
- NumPy: Numerical computations.
- Statsmodels: Poisson regression modeling.
- Scikit-learn: Neural network implementation and data scaling.
- Streamlit: Web application framework.

# Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes.
4. Submit a pull request.

