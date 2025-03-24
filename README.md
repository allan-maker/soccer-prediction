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

Key features include:

1. **Dynamic User Interface**: Select home and away teams to get real-time predictions.
2. **Expected Goals**: Displays predicted goals for both teams.
3. **Match Probabilities**: Shows the likelihood of a home win, away win, or draw.

Built with Python, Pandas, Scikit-learn, Statsmodels, and Streamlit, this project is ideal for sports analytics enthusiasts and data scientists interested in predictive modeling.

# How to use it
- Clone the repository.
- **Install dependencies**: pip install -r requirements.txt.
- Run the app: streamlit run app.py.
- Open the app in your browser and start predicting!


