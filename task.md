# FIFA World Cup Match Prediction Platform — Task Checklist

## Phase 0 — Environment Setup
- [x] Create project folder structure
- [x] Create virtual environment (sklearn-env)
- [x] Install all dependencies (streamlit, scikit-learn, plotly, pandas, etc.)
- [x] Download Kaggle datasets

## Phase 1 — Data Layer
- [x] Process historical football data CSV
- [x] Filter to World Cup matches only
- [x] Fix public API URL and update parsing logic in `ml/live_scores.py`
- [x] Connect completed match sync to retraining pipeline in `ml/live_scores.py`
- [x] Feature engineering pipeline
- [x] 2026 fixture data setup

## Phase 2 — ML Pipeline
- [x] `ml/features.py` — feature engineering
- [x] `ml/train.py` — model training (LR → RF → XGBoost)
- [x] `ml/predict.py` — prediction generation
- [x] `ml/evaluate.py` — accuracy metrics
- [x] Train model on historical data

## Phase 3 — Streamlit UI
- [x] `app/Home.py` — landing page
- [x] `app/pages/1_Upcoming_Matches.py` — fixtures + predictions
- [x] `app/pages/2_Match_Detail.py` — match deep-dive
- [x] `app/pages/3_Accuracy_Tracker.py` — model performance
- [x] `app/pages/4_Tournament_Bracket.py` — bracket view
- [x] Beautiful dark-mode design with Plotly charts

## Phase 4 — Config & Deployment
- [x] `requirements.txt`
- [x] `.streamlit/config.toml`
- [x] `README.md`
- [x] Test locally
- [x] Streamlit Community Cloud deployment guide
