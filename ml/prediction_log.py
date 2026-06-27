"""
Prediction Log — FIFA World Cup 2026 AI Predictor
Persists every AI pre-match prediction to data/prediction_log.json.
When a result is received (via sync_results_from_fixtures), records whether
the AI was correct. Provides get_live_accuracy_stats() for the UI.
"""

import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

DATA_DIR = Path(__file__).parent.parent / "data"
LOG_PATH = DATA_DIR / "prediction_log.json"

# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_log() -> Dict[str, Any]:
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_log(data: Dict[str, Any]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


# ── Public API ────────────────────────────────────────────────────────────────

def log_prediction(
    match_id: int,
    home_team: str,
    away_team: str,
    round_name: str,
    predicted_outcome: str,
    confidence: float,
    home_win_prob: float,
    draw_prob: float,
    away_win_prob: float,
    match_date: str = "",
) -> None:
    """
    Store an AI prediction for a future/upcoming match.
    Only writes if this match_id hasn't been logged yet.
    """
    log = _load_log()
    key = str(match_id)
    if key in log:
        return  # Already logged — don't overwrite
    log[key] = {
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "round": round_name,
        "date": match_date,
        "predicted_outcome": predicted_outcome,
        "confidence": round(confidence, 4),
        "home_win_prob": round(home_win_prob, 4),
        "draw_prob": round(draw_prob, 4),
        "away_win_prob": round(away_win_prob, 4),
        "logged_at": datetime.datetime.utcnow().isoformat(),
        "actual_outcome": None,
        "is_correct": None,
        "result_recorded_at": None,
    }
    _save_log(log)


def record_result(match_id: int, actual_outcome: str) -> bool:
    """
    Record the actual match result and compute correctness.
    Returns True if a new result was recorded, False if already done.
    actual_outcome: 'home_win' | 'away_win' | 'draw'
    """
    log = _load_log()
    key = str(match_id)
    if key not in log:
        return False
    entry = log[key]
    if entry.get("actual_outcome") is not None:
        return False  # Already recorded
    entry["actual_outcome"] = actual_outcome
    entry["is_correct"] = (entry["predicted_outcome"] == actual_outcome)
    entry["result_recorded_at"] = datetime.datetime.utcnow().isoformat()
    log[key] = entry
    _save_log(log)
    return True


def sync_results_from_fixtures(fixtures_df) -> int:
    """
    Scan completed fixtures and record results for any logged prediction
    that doesn't have a result yet. Returns count of newly recorded results.
    """
    completed = fixtures_df[fixtures_df["status"] == "completed"].copy()
    newly_recorded = 0
    for _, row in completed.iterrows():
        hs = row.get("home_score", 0) or 0
        as_ = row.get("away_score", 0) or 0
        if hs > as_:
            actual = "home_win"
        elif as_ > hs:
            actual = "away_win"
        else:
            actual = "draw"
        if record_result(int(row["match_id"]), actual):
            newly_recorded += 1
    return newly_recorded


def get_live_accuracy_stats() -> Dict[str, Any]:
    """
    Compute live prediction accuracy from the log.
    Returns dict with: total, correct, wrong, pending, accuracy, by_round.
    """
    log = _load_log()
    entries = list(log.values())

    resolved = [e for e in entries if e.get("is_correct") is not None]
    correct = [e for e in resolved if e["is_correct"]]
    wrong = [e for e in resolved if not e["is_correct"]]
    pending = [e for e in entries if e.get("is_correct") is None]

    total_resolved = len(resolved)
    accuracy = len(correct) / total_resolved if total_resolved > 0 else None

    # Per-round breakdown
    by_round: Dict[str, Dict] = {}
    for e in resolved:
        r = e.get("round", "Unknown")
        if r not in by_round:
            by_round[r] = {"total": 0, "correct": 0}
        by_round[r]["total"] += 1
        if e["is_correct"]:
            by_round[r]["correct"] += 1
    for r in by_round:
        t = by_round[r]["total"]
        c = by_round[r]["correct"]
        by_round[r]["accuracy"] = c / t if t > 0 else None

    # Cumulative accuracy series (sorted by date)
    timeline = []
    sorted_resolved = sorted(resolved, key=lambda e: e.get("date", "") or "")
    running_correct = 0
    for i, e in enumerate(sorted_resolved, 1):
        if e["is_correct"]:
            running_correct += 1
        timeline.append({
            "match_num": i,
            "date": e.get("date", ""),
            "match": f"{e['home_team']} vs {e['away_team']}",
            "cumulative_accuracy": running_correct / i,
            "is_correct": e["is_correct"],
        })

    return {
        "total": len(entries),
        "resolved": total_resolved,
        "correct": len(correct),
        "wrong": len(wrong),
        "pending": len(pending),
        "accuracy": accuracy,
        "by_round": by_round,
        "timeline": timeline,
        "entries": entries,
    }


def get_prediction_for_match(match_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve the logged prediction entry for a specific match_id."""
    log = _load_log()
    return log.get(str(match_id))


def get_outcome_badge_html(match_id: int, home_team: str, away_team: str) -> str:
    """
    Returns an HTML badge string for a completed match:
    - Green ✅ + 'AI Correct' if prediction matched result
    - Red ❌ + 'AI Wrong' if prediction didn't match
    - Empty string if no prediction was logged
    """
    entry = get_prediction_for_match(match_id)
    if not entry or entry.get("is_correct") is None:
        return ""

    predicted = entry["predicted_outcome"]
    actual = entry.get("actual_outcome", "")
    conf = entry.get("confidence", 0)

    # Human readable labels
    def outcome_label(outcome, home, away):
        if outcome == "home_win":
            return home
        elif outcome == "away_win":
            return away
        return "Draw"

    pred_label = outcome_label(predicted, home_team, away_team)
    actual_label = outcome_label(actual, home_team, away_team)

    if entry["is_correct"]:
        return (
            f'<span class="pred-badge pred-correct" title="AI predicted: {pred_label} ({conf:.0%} confidence)">'
            f'✅ AI Correct'
            f'</span>'
        )
    else:
        return (
            f'<span class="pred-badge pred-wrong" title="AI predicted: {pred_label} ({conf:.0%} confidence) · Actual: {actual_label}">'
            f'❌ AI Wrong'
            f'</span>'
        )
