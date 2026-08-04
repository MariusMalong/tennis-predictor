# ATP Tennis Match Predictor

A machine learning project that predicts the winner of ATP tennis matches using player rankings, points, surface history, and recent form. Built end to end: exploratory data analysis, feature engineering, model comparison, and a deployed API.

## Live Demo

- API root: https://tennis-predictor-api.onrender.com/
- Interactive docs: https://tennis-predictor-api.onrender.com/docs

Hosted on Render's free tier, which spins down after 15 minutes of inactivity. The first request after idle time can take 30-60 seconds while the service wakes back up.

## Project Overview

This project works through 68,300+ ATP matches from 2000 to 2026, sourced from Kaggle. The goal was to predict match winners using a mix of raw stats (ranking, ranking points) and engineered features (career win rate, surface-specific win rate, recent form), all built with careful attention to avoiding data leakage.

The best mdoel (XGBoost) reached 64.9% accuracy, modestly ahead of a simple "higher-ranked player wins" baseline at 64.2%. This is broadly in line with expectations for this kind of problem.

## What's in this repo

- `notebooks/01_eda.ipynb` - exploratory data analysis, including catching a disguised missing-data pattern (a -1 sentinel value instead of true nulls) and confirming which features corrolate with match outcome
- `notebooks/02_feature_engineering.ipynb` - building leakage-safe, time-aware features (rolling win rates, recent form
- `notebooks/03_modeling.ipynb` - baseline comparison, then Logistic Regression, Random Forest, and XGBoost, evaluated on a time-based train/test split
- `src/predict.py` - loads the trained model and exposes a clean prediction function
- `src/api.py` - a small FastAPI app wrapping that prediction function for public use

## Key findings

- rank_diff and points_diff both showed a real relationship with match outcome, but the engineered rolling win-rate features (especially career_win_rate_diff) turned out to be even stronger predictors, which validated the extra feature engineering work
- All three models (Logistic Regression, Random Forest, XGBoost) converged to a very similar accuracy, suggesting the relationship between these features and match outcome is fairly linear, and that most of the predictable signal in tennis outcomes lives in ranking, points, and recent form rather than anything more complex these models could exploit
- Odds data (Odd_1/Odd_2) was excluded from the main model since it's only reliably available form around 2005 onward. Including it meant either dropping a large chunk of earlier matches or mixing incomparable data

## Limitations

- No head-to-head record feature yet. This is the most promising next addition, since it captures matchup-specific effects that ranking and form can't see
- No fatigue or scheduling signal (for example, matches player in the last few days)
- Model performance likely varies by tournament tier and surface, which hasn't been broken down in detail yet

## Running locally

```bash
git clone https://github.com/MariusMalong/tennis-predictor.git
cd tennis-predictor
conda create -n datasci python-3.11
conda activate datasci
pip install -r requirements.txt

uvicorn src.api:app --reload
```

Then visit `http://127.0.0.1:8000/docs` to test predictions interactively.

Note: the raw dataset isn't included in this repo. Download it from [Kaggle](https://www.kaggle.com/datasets/dissfya/atp-tennis-2000-2023daily-pull) and place it in `data/raw/` before running the notebooks.

## Tech stack 

Python, pandas, scikit-learn, XGBoost, FastAPI, deployed on Render.