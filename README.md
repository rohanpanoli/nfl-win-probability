# NFL Game Win Probability Model

A machine learning pipeline that predicts the probability a home team wins in the NFL by Week
<img width="790" height="950" alt="week4" src="https://github.com/user-attachments/assets/d130d508-1558-4929-9a67-820d07242b41" />

## Overview

- **Goal:** To predict the probability an NFL team wins a game for every upcoming game in the 2026 Season
- **Data:** Game results and schedules form [nflverse]
- **Final model:** Logistic regression with training samples weighted by recency
- **Result:** 61.70% Accuracy with 63.62% logloss, vs 57.45% baseline (home team always winning)

- ## Project Structure

- ```
├── notebooks/
│   ├── 01_data_collection.ipynb      # Pulls raw schedules/results from nflverse
│   ├── 02_feature_engineering.ipynb  # Rolling team stats + Elo ratings
│   ├── 03_model_training.ipynb       # Trains and compares models
│   └── 04_weekly_predictions.ipynb   # Predicts and plots upcoming games
├── utils.py                          # Shared feature logic (rolling stats, Elo)
├── models/                           # Saved trained model
├── requirements.txt
└── README.md
```

## Methodology

**Features (per game): **
- Each team's rolling averages over their last 5 games: points scored, points allowed, win percentage
- Pre-game Elo ratings for both teams

**Training setup:**
- Trained on team data from 2019 onward (to reflect the modern NFL era while keeping a large sample size)
- Exponential recency weighing (decay = 0.85) so that recent seasons count more

**Models compared:**

| Model | Accuracy | Log Loss |
|---|---|---|
| Home-team baseline | 57.45% | n/a |
| Logistic regression | 61.70% | 63.52% |
| XGBoost | 61.70& | 63.49% |


## How to Run

```bash
git clone https://github.com/rohanpanoli/nfl-win-probability.git
cd nfl-win-probability
pip install -r requirements.txt
```
Then run notebooks in order

## Limitations and Future Work
- Does not account for notable injuries or weather
- Features only use points scored/allowed, will implement advanced efficiency statistics in the future (DVOA, EPA)
- Will add calibration analysis
