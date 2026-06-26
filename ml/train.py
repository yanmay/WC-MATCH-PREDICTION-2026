"""
Model training pipeline for the FIFA World Cup Match Prediction Platform.
Trains Logistic Regression → Random Forest → XGBoost (progressively).
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.impute import SimpleImputer

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "home_ranking", "away_ranking", "ranking_diff",
    "home_win_rate", "away_win_rate", "home_draw_rate", "away_draw_rate",
    "h2h_home_wins", "h2h_away_wins", "h2h_draws",
    "h2h_total", "h2h_has_history",
    "home_goals_pg", "away_goals_pg",
    "home_conceded_pg", "away_conceded_pg",
    "home_rest_days", "away_rest_days",
    "host_nation", "round_encoded", "is_knockout", "same_confederation",
]
CATEGORICAL_FEATURES = ["confederation_matchup"]


def build_pipeline(algorithm: str = "random_forest") -> Pipeline:
    """Build a scikit-learn Pipeline for the given algorithm."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ], remainder="drop")

    if algorithm == "logistic_regression":
        base = LogisticRegression(
            solver="lbfgs",
            max_iter=1000, C=1.0, random_state=42,
        )
        classifier = CalibratedClassifierCV(base, method="isotonic", cv=3)
    elif algorithm == "gradient_boosting":
        base = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
        )
        classifier = CalibratedClassifierCV(base, method="isotonic", cv=3)
    else:  # random_forest (default)
        base = RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        classifier = CalibratedClassifierCV(base, method="isotonic", cv=3)

    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    algorithm: str = "random_forest",
    test_size: float = 0.2,
) -> Dict[str, Any]:
    """
    Train the model and return metrics + trained pipeline.
    Uses stratified split for evaluation.
    """
    if len(X) < 20:
        raise ValueError(f"Not enough training data: {len(X)} samples. Need at least 20.")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    pipeline = build_pipeline(algorithm)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cal_loss = log_loss(y_test, y_proba)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        build_pipeline(algorithm), X, y, cv=cv, scoring="accuracy", n_jobs=-1
    )

    # Feature importance (only for RF/GBM)
    try:
        cat_encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"]["onehot"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        all_feature_names = NUMERIC_FEATURES + cat_feature_names

        base_model = pipeline.named_steps["classifier"].calibrated_classifiers_[0].estimator
        if hasattr(base_model, "feature_importances_"):
            importances = base_model.feature_importances_
            feature_importance = dict(sorted(
                zip(all_feature_names, importances),
                key=lambda x: x[1], reverse=True
            )[:15])
        elif hasattr(base_model, "coef_"):
            coefs = np.abs(base_model.coef_).mean(axis=0)
            feature_importance = dict(sorted(
                zip(all_feature_names, coefs),
                key=lambda x: x[1], reverse=True
            )[:15])
        else:
            feature_importance = {}
    except Exception:
        feature_importance = {}

    metrics = {
        "algorithm": algorithm,
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "accuracy": round(accuracy, 4),
        "log_loss": round(cal_loss, 4),
        "cv_mean_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std_accuracy": round(float(cv_scores.std()), 4),
        "classes": list(pipeline.classes_),
        "feature_importance": feature_importance,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    return {"pipeline": pipeline, "metrics": metrics}


def save_model(pipeline: Pipeline, metrics: dict, version: str = "v1.0") -> str:
    """Serialize model + metadata to artifacts directory."""
    model_path = ARTIFACTS_DIR / f"model_{version}.joblib"
    meta_path = ARTIFACTS_DIR / f"model_{version}_metrics.json"

    joblib.dump(pipeline, model_path)

    meta_to_save = {k: v for k, v in metrics.items() if k != "classification_report"}
    with open(meta_path, "w") as f:
        json.dump(meta_to_save, f, indent=2, default=str)

    # Write active model pointer
    with open(ARTIFACTS_DIR / "active_model.txt", "w") as f:
        f.write(version)

    return str(model_path)


def load_model(version: str = None) -> Tuple[Pipeline, dict]:
    """Load the active (or specified) model from artifacts."""
    if version is None:
        pointer = ARTIFACTS_DIR / "active_model.txt"
        if pointer.exists():
            version = pointer.read_text().strip()
        else:
            raise FileNotFoundError("No active model found. Run train.py first.")

    model_path = ARTIFACTS_DIR / f"model_{version}.joblib"
    meta_path = ARTIFACTS_DIR / f"model_{version}_metrics.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    pipeline = joblib.load(model_path)
    metrics = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metrics = json.load(f)
    return pipeline, metrics


def auto_train_best_model(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """
    Try multiple algorithms and return the best one by test accuracy.
    """
    algorithms = ["logistic_regression", "random_forest", "gradient_boosting"]
    best_result = None
    best_accuracy = -1

    for algo in algorithms:
        try:
            result = train_model(X, y, algorithm=algo)
            if result["metrics"]["accuracy"] > best_accuracy:
                best_accuracy = result["metrics"]["accuracy"]
                best_result = result
                best_result["metrics"]["selected_algorithm"] = algo
        except Exception as e:
            print(f"[WARN] {algo} failed: {e}")
            continue

    return best_result


if __name__ == "__main__":
    import sys
    # Add root folder to sys.path if not present
    sys.path.append(str(Path(__file__).parent.parent.resolve()))
    
    from ml.data_loader import load_historical_data, get_world_cup_data
    from ml.features import build_training_features

    print("Loading historical match data...")
    df = load_historical_data()
    print(f"Total historical matches loaded: {len(df)}")
    
    wc_df = get_world_cup_data(df)
    print(f"Total World Cup matches found: {len(wc_df)}")
    
    print("Building features...")
    X, y = build_training_features(wc_df)
    print(f"Feature matrix shape: {X.shape}")
    
    if len(X) == 0:
        print("[ERROR] No features generated. Cannot train model.")
        sys.exit(1)
        
    print("Training best model...")
    result = auto_train_best_model(X, y)
    
    if result is None:
        print("[ERROR] Model training failed.")
        sys.exit(1)
        
    pipeline = result["pipeline"]
    metrics = result["metrics"]
    
    print(f"Best algorithm: {metrics.get('selected_algorithm')}")
    print(f"Accuracy: {metrics.get('accuracy')}")
    print(f"CV Mean Accuracy: {metrics.get('cv_mean_accuracy')}")
    
    saved_path = save_model(pipeline, metrics, version="v1.0")
    print(f"Model saved to: {saved_path}")

