# ⚽ FIFA World Cup 2026 Match Prediction Platform

An AI-powered machine learning platform designed to predict match outcomes for the **FIFA World Cup 2026** (Round of 32 onwards) and simulate custom tournament brackets. 

Built as a lightweight, performant, and visually stunning data science showcase for portfolios and public sharing.

---

## 🌟 Key Features

1. **🏠 Dynamic Home Dashboard**: Landing page showing next match previews, model confidence scores, key performance stats, and a tournament timeline.
2. **📅 Upcoming Matches**: Chronological schedule of remaining 2026 fixtures with Win/Draw/Loss probability bars and AI-predicted winner badges.
3. **🔍 Interactive Match Simulator**: Custom matchup simulator allowing users to pitch any two teams against each other and adjust rest days or goal forms to see real-time probability changes.
4. **📊 Backtest & Accuracy Tracker**: Full backtesting evaluation of the model against the **Qatar 2022 World Cup** (64 matches), showing observed accuracy, Brier scores, and a Plotly probability calibration curve.
5. **🏆 Dynamic Bracket Simulator**: Propagates predicted winners from the Round of 32 all the way to the Final, determining the simulated 2026 World Cup Champion.

---

## 🛠️ Technology Stack

* **Frontend**: [Streamlit](https://streamlit.io/) (Clean, modern dark theme matching tournament aesthetics)
* **Visualizations**: [Plotly](https://plotly.com/) (Interactive pie charts, calibration charts, and feature importances)
* **ML Pipeline**: [scikit-learn](https://scikit-learn.org/) (Logistic Regression, Random Forests, Gradient Boosting Classifiers, CalibratedClassifierCV)
* **Data Processing**: [pandas](https://pandas.pydata.org/) & [numpy](https://numpy.org/)
* **Model Persistence**: [joblib](https://joblib.readthedocs.io/)

---

## 📁 Project Structure

```text
├── .streamlit/
│   └── config.toml          # Custom theme settings (dark mode, colors)
├── app/
│   ├── Home.py              # Main dashboard entrypoint
│   ├── utils.py             # Shared state, CSS system, and team flag assets
│   └── pages/
│       ├── 1_Upcoming_Matches.py  # Predictions list and filters
│       ├── 2_Match_Detail.py      # Match picker, H2H record, custom simulator
│       ├── 3_Accuracy_Tracker.py  # Evaluator, Qatar 2022 backtest log, calibration
│       └── 4_Tournament_Bracket.py # Symmetrical tournament bracket propagation
├── data/
│   └── results.csv          # 90+ years of historical international match results
├── ml/
│   ├── artifacts/           # Trained model and metric JSON metadata
│   ├── data_loader.py       # Kaggle dataset parsing and filtering
│   ├── evaluate.py          # Metric calculations (Accuracy, Brier, Calibration)
│   ├── features.py          # Feature engineering (rankings, rest, forms)
│   ├── predict.py           # Core inference engine (prob calibration, KO draws)
│   └── train.py             # Model training entrypoint (auto-selects best model)
├── requirements.txt         # Pinned project dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Local Installation & Setup

Follow these steps to run the project locally on your machine:

### 1. Clone the repository
```bash
git clone <your-github-repo-url>
cd "match prediction system FIFA"
```

### 2. Set up a virtual environment
```bash
# On Windows
py -3 -m venv sklearn-env
sklearn-env\Scripts\activate

# On macOS/Linux
python3 -m venv sklearn-env
source sklearn-env/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Run the model training pipeline
The repository includes a pre-trained model, but you can retrain the model on the latest dataset at any time:
```bash
python ml/train.py
```

### 5. Start the Streamlit Application
```bash
streamlit run app/Home.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🤖 Machine Learning Details

* **Dataset**: Over 49,000 international matches (1872–present) filtered down to ~10,000 FIFA World Cup matches to align on tournament-level performance dynamics.
* **Features Engineered**:
  * Home/Away team FIFA World Rankings and ranking differences.
  * Historical World Cup match win/draw rates.
  * Form variables (average goals scored and conceded in recent matches).
  * Confederation matchups and same-confederation flags.
  * Game factors (rest days, knockout stage flags, host nation advantage).
* **Calibration**: Models are calibrated using `CalibratedClassifierCV(..., method="isotonic")` to convert raw decision values into precise, reliable probability estimates.
* **Model Selection**: The training pipeline evaluates multiple model configurations:
  * **Logistic Regression**: High interpretability, strong baseline.
  * **Random Forest**: Handles non-linear feature interactions and high-dimensional inputs.
  * **Gradient Boosting**: Sequential tree ensemble.
  
  The best model by out-of-sample test accuracy is automatically saved and set as active (the current model is **Logistic Regression** with **62.9% CV accuracy**).
