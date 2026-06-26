"""
Model evaluation module — tracks accuracy, calibration, and per-round performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


def compute_accuracy_stats(prediction_log: List[dict]) -> Dict[str, Any]:
    """
    Compute accuracy statistics from a list of prediction outcome records.
    Each record: {round, predicted_outcome, actual_outcome, home_win_prob, draw_prob, away_win_prob}
    """
    if not prediction_log:
        return {
            "total": 0, "correct": 0, "accuracy": 0.0,
            "by_round": {}, "calibration": [],
        }

    df = pd.DataFrame(prediction_log)
    df["correct"] = df["predicted_outcome"] == df["actual_outcome"]

    total = len(df)
    correct = int(df["correct"].sum())
    accuracy = correct / total if total > 0 else 0.0

    # Per-round breakdown
    by_round = {}
    for round_name, group in df.groupby("round"):
        r_total = len(group)
        r_correct = int(group["correct"].sum())
        by_round[round_name] = {
            "total": r_total,
            "correct": r_correct,
            "accuracy": r_correct / r_total if r_total > 0 else 0.0,
        }

    # Calibration: bucket confidence into bins, check actual accuracy
    calibration = []
    for bin_low in np.arange(0.33, 1.0, 0.1):
        bin_high = bin_low + 0.1
        mask = (df["confidence"] >= bin_low) & (df["confidence"] < bin_high)
        bucket = df[mask]
        if len(bucket) > 0:
            calibration.append({
                "confidence_bin": f"{bin_low:.0%}–{bin_high:.0%}",
                "predicted_confidence": round(bucket["confidence"].mean(), 3),
                "actual_accuracy": round(bucket["correct"].mean(), 3),
                "samples": len(bucket),
            })

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "by_round": by_round,
        "calibration": calibration,
    }


def baseline_accuracy(wc_df: pd.DataFrame) -> float:
    """
    Compute naive baseline: always predict home_win.
    Used to validate our model beats the trivial baseline.
    """
    if wc_df.empty:
        return 0.0
    home_wins = (wc_df["outcome"] == "home_win").sum()
    return home_wins / len(wc_df)


def compute_brier_score(
    actual_outcome: str,
    home_win_prob: float,
    draw_prob: float,
    away_win_prob: float,
) -> float:
    """Compute Brier score for a single prediction."""
    outcome_map = {
        "home_win": [1, 0, 0],
        "draw": [0, 1, 0],
        "away_win": [0, 0, 1],
    }
    actuals = outcome_map.get(actual_outcome, [0, 0, 0])
    preds = [home_win_prob, draw_prob, away_win_prob]
    return sum((p - a) ** 2 for p, a in zip(preds, actuals)) / 3
