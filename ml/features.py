"""
Feature engineering for the FIFA World Cup Match Prediction Platform.
Builds feature vectors for training and prediction from historical data.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from functools import lru_cache
import hashlib


# FIFA World Rankings (approximate, 2026 pre-tournament)
FIFA_RANKINGS_2026 = {
    "Argentina": 1, "France": 2, "England": 3, "Belgium": 4, "Brazil": 5,
    "Portugal": 6, "Netherlands": 7, "Spain": 8, "Croatia": 9, "Italy": 10,
    "Morocco": 12, "USA": 13, "Mexico": 15, "Germany": 16, "Colombia": 11,
    "Uruguay": 14, "Denmark": 21, "Austria": 24, "Switzerland": 19,
    "Japan": 18, "South Korea": 23, "Australia": 25, "Canada": 47,
    "Senegal": 20, "Ecuador": 38, "Serbia": 33, "Poland": 26, "Iran": 22,
    "Ghana": 56, "Cameroon": 43, "Egypt": 32, "Wales": 29, "Tunisia": 30,
    "Ivory Coast": 41, "Algeria": 35, "Nigeria": 45, "Saudi Arabia": 56,
    "Qatar": 37, "Russia": 50, "Sweden": 28, "Chile": 36, "Peru": 27,
    "Paraguay": 60, "Bolivia": 72, "Venezuela": 58, "Panama": 66,
    "Costa Rica": 55, "Honduras": 70, "Jamaica": 52, "Trinidad and Tobago": 80,
    "South Africa": 65, "Zambia": 81, "Mozambique": 95, "Zimbabwe": 88,
    "Iceland": 68, "Slovakia": 47, "Hungary": 48, "Romania": 45,
    "Greece": 54, "Turkey": 40, "Ukraine": 24, "Czech Republic": 43,
    "North Korea": 112, "New Zealand": 104, "Iraq": 69, "Lebanon": 98,
    "Bahrain": 87, "Oman": 74, "Jordan": 77, "United Arab Emirates": 71,
    # 2026 WC additional teams
    "Norway": 31, "Scotland": 39, "Czechia": 43, "Bosnia and Herzegovina": 55,
    "Uzbekistan": 74, "Haiti": 83, "DR Congo": 62, "Curacao": 82,
    "Cabo Verde": 78, "Iraq": 67,
}

HOST_NATIONS_2026 = {"USA", "Mexico", "Canada"}

ROUND_ENCODING = {
    "group": 0, "Group Stage": 0, "Round of 32": 1, "Round of 16": 2,
    "Quarterfinal": 3, "Semifinal": 4, "3rd Place": 4, "Final": 5,
}

CONFEDERATION_MAP = {
    "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Uruguay": "CONMEBOL",
    "Colombia": "CONMEBOL", "Chile": "CONMEBOL", "Paraguay": "CONMEBOL",
    "Ecuador": "CONMEBOL", "Peru": "CONMEBOL", "Venezuela": "CONMEBOL",
    "Bolivia": "CONMEBOL",
    "France": "UEFA", "Germany": "UEFA", "England": "UEFA", "Spain": "UEFA",
    "Italy": "UEFA", "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA",
    "Croatia": "UEFA", "Denmark": "UEFA", "Sweden": "UEFA", "Switzerland": "UEFA",
    "Austria": "UEFA", "Poland": "UEFA", "Russia": "UEFA", "Serbia": "UEFA",
    "Ukraine": "UEFA", "Hungary": "UEFA", "Slovakia": "UEFA", "Czech Republic": "UEFA",
    "Romania": "UEFA", "Greece": "UEFA", "Turkey": "UEFA", "Wales": "UEFA",
    "Iceland": "UEFA", "Norway": "UEFA", "Scotland": "UEFA", "Czechia": "UEFA",
    "Bosnia and Herzegovina": "UEFA",
    "USA": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
    "Costa Rica": "CONCACAF", "Honduras": "CONCACAF", "Jamaica": "CONCACAF",
    "Panama": "CONCACAF", "Trinidad and Tobago": "CONCACAF", "Curacao": "CONCACAF",
    "Cabo Verde": "CAF", "Haiti": "CONCACAF",
    "Morocco": "CAF", "Senegal": "CAF", "Ghana": "CAF", "Cameroon": "CAF",
    "Egypt": "CAF", "Nigeria": "CAF", "Ivory Coast": "CAF",
    "Algeria": "CAF", "Tunisia": "CAF", "South Africa": "CAF", "DR Congo": "CAF",
    "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC",
    "Iran": "AFC", "Saudi Arabia": "AFC", "Iraq": "AFC",
    "Jordan": "AFC", "United Arab Emirates": "AFC", "Bahrain": "AFC",
    "Oman": "AFC", "Qatar": "AFC", "North Korea": "AFC", "Uzbekistan": "AFC",
    "New Zealand": "OFC",
}


def get_ranking(team: str) -> int:
    """Get FIFA ranking for a team, defaulting to mid-tier if unknown."""
    return FIFA_RANKINGS_2026.get(team, 60)


def get_confederation(team: str) -> str:
    """Get confederation for a team."""
    return CONFEDERATION_MAP.get(team, "OTHER")


def build_training_features(wc_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Build feature matrix and target series from historical WC data.
    Returns (X, y) for scikit-learn training.
    Optimized: uses running H2H dictionary for O(N) execution.
    """
    team_stats = _compute_team_stats(wc_df)
    # Filter valid outcomes first
    valid = wc_df[wc_df["outcome"].isin(["home_win", "away_win", "draw"])].copy()
    if len(valid) == 0:
        return pd.DataFrame(), pd.Series()

    # Sort chronologically by date to ensure running H2H is correct
    valid = valid.sort_values("date").reset_index(drop=True)

    h2h_tracker = {}
    rows = []

    for _, match in valid.iterrows():
        home_team = match["home_team"]
        away_team = match["away_team"]
        outcome = match["outcome"]

        # Determine alphabetical order for H2H keys
        teamA, teamB = (home_team, away_team) if home_team < away_team else (away_team, home_team)
        
        # Get current H2H record prior to this match
        if (teamA, teamB) in h2h_tracker:
            record = h2h_tracker[(teamA, teamB)]
            h2h_total = record["total"]
            h2h_draws = record["draws"]
            h2h_has_history = 1
            if home_team == teamA:
                h2h_home_wins = record["teamA_wins"]
                h2h_away_wins = record["teamB_wins"]
            else:
                h2h_home_wins = record["teamB_wins"]
                h2h_away_wins = record["teamA_wins"]
        else:
            h2h_home_wins = 0
            h2h_away_wins = 0
            h2h_draws = 0
            h2h_total = 0
            h2h_has_history = 0

        h_rank = get_ranking(home_team)
        a_rank = get_ranking(away_team)
        h_stats = team_stats.get(home_team, {"win_rate": 0.33, "draw_rate": 0.20, "goals_pg": 1.5, "conceded_pg": 1.0})
        a_stats = team_stats.get(away_team, {"win_rate": 0.33, "draw_rate": 0.20, "goals_pg": 1.5, "conceded_pg": 1.0})

        h_conf = get_confederation(home_team)
        a_conf = get_confederation(away_team)
        conf_matchup = f"{h_conf}_vs_{a_conf}" if h_conf <= a_conf else f"{a_conf}_vs_{h_conf}"

        round_name = match.get("round", "group")
        round_enc = ROUND_ENCODING.get(str(round_name), 0)
        is_knockout = 1 if round_enc >= 1 else 0

        row = {
            "home_ranking": h_rank,
            "away_ranking": a_rank,
            "ranking_diff": h_rank - a_rank,
            "home_win_rate": h_stats["win_rate"],
            "away_win_rate": a_stats["win_rate"],
            "home_draw_rate": h_stats["draw_rate"],
            "away_draw_rate": a_stats["draw_rate"],
            "h2h_home_wins": h2h_home_wins,
            "h2h_away_wins": h2h_away_wins,
            "h2h_draws": h2h_draws,
            "h2h_total": h2h_total,
            "h2h_has_history": h2h_has_history,
            "home_goals_pg": h_stats["goals_pg"],
            "away_goals_pg": a_stats["goals_pg"],
            "home_conceded_pg": h_stats["conceded_pg"],
            "away_conceded_pg": a_stats["conceded_pg"],
            "home_rest_days": 7,
            "away_rest_days": 7,
            "host_nation": 1 if home_team in HOST_NATIONS_2026 else 0,
            "round_encoded": round_enc,
            "is_knockout": is_knockout,
            "same_confederation": 1 if h_conf == a_conf else 0,
            "confederation_matchup": conf_matchup,
            "win_rate_diff": h_stats["win_rate"] - a_stats["win_rate"],
            "goals_pg_diff": h_stats["goals_pg"] - a_stats["goals_pg"],
            "conceded_pg_diff": h_stats["conceded_pg"] - a_stats["conceded_pg"],
            "rank_ratio": h_rank / (a_rank + 1),
            "ranking_diff_abs": abs(h_rank - a_rank),
            "outcome": outcome,
        }
        rows.append(row)

        # Update H2H tracker with the outcome of the match for future loops
        if (teamA, teamB) not in h2h_tracker:
            h2h_tracker[(teamA, teamB)] = {"teamA_wins": 0, "teamB_wins": 0, "draws": 0, "total": 0}
        
        rec = h2h_tracker[(teamA, teamB)]
        rec["total"] += 1
        if outcome == "draw":
            rec["draws"] += 1
        elif outcome == "home_win":
            if home_team == teamA:
                rec["teamA_wins"] += 1
            else:
                rec["teamB_wins"] += 1
        elif outcome == "away_win":
            if away_team == teamA:
                rec["teamA_wins"] += 1
            else:
                rec["teamB_wins"] += 1

    if not rows:
        return pd.DataFrame(), pd.Series()

    features_df = pd.DataFrame(rows)
    y = features_df.pop("outcome")
    X = features_df
    return X, y


def build_prediction_features(
    home_team: str,
    away_team: str,
    round_name: str,
    wc_df: pd.DataFrame,
    home_goals_pg: float = 1.5,
    away_goals_pg: float = 1.5,
    home_conceded_pg: float = 1.0,
    away_conceded_pg: float = 1.0,
    home_rest_days: int = 7,
    away_rest_days: int = 7,
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame for prediction.
    """
    h_rank = get_ranking(home_team)
    a_rank = get_ranking(away_team)

    # Head-to-head
    h2h_mask = (
        ((wc_df["home_team"] == home_team) & (wc_df["away_team"] == away_team)) |
        ((wc_df["home_team"] == away_team) & (wc_df["away_team"] == home_team))
    )
    h2h = wc_df[h2h_mask]
    h2h_home_wins = (
        ((h2h["home_team"] == home_team) & (h2h["outcome"] == "home_win")) |
        ((h2h["away_team"] == home_team) & (h2h["outcome"] == "away_win"))
    ).sum()
    h2h_away_wins = (
        ((h2h["home_team"] == away_team) & (h2h["outcome"] == "home_win")) |
        ((h2h["away_team"] == away_team) & (h2h["outcome"] == "away_win"))
    ).sum()
    h2h_draws = (h2h["outcome"] == "draw").sum()

    team_stats = _compute_team_stats(wc_df)
    h_stats = team_stats.get(home_team, {"win_rate": 0.33, "draw_rate": 0.20, "goals_pg": 1.5, "conceded_pg": 1.0})
    a_stats = team_stats.get(away_team, {"win_rate": 0.33, "draw_rate": 0.20, "goals_pg": 1.5, "conceded_pg": 1.0})

    h_conf = get_confederation(home_team)
    a_conf = get_confederation(away_team)
    same_conf = 1 if h_conf == a_conf else 0
    conf_matchup = f"{h_conf}_vs_{a_conf}" if h_conf <= a_conf else f"{a_conf}_vs_{h_conf}"

    is_knockout = 0 if "Group" in round_name else 1
    round_enc = ROUND_ENCODING.get(round_name, 1)

    features = {
        "home_ranking": h_rank,
        "away_ranking": a_rank,
        "ranking_diff": h_rank - a_rank,
        "home_win_rate": h_stats["win_rate"],
        "away_win_rate": a_stats["win_rate"],
        "home_draw_rate": h_stats["draw_rate"],
        "away_draw_rate": a_stats["draw_rate"],
        "h2h_home_wins": h2h_home_wins,
        "h2h_away_wins": h2h_away_wins,
        "h2h_draws": h2h_draws,
        "h2h_total": len(h2h),
        "h2h_has_history": 1 if len(h2h) > 0 else 0,
        "home_goals_pg": home_goals_pg,
        "away_goals_pg": away_goals_pg,
        "home_conceded_pg": home_conceded_pg,
        "away_conceded_pg": away_conceded_pg,
        "home_rest_days": home_rest_days,
        "away_rest_days": away_rest_days,
        "host_nation": 1 if home_team in HOST_NATIONS_2026 else 0,
        "round_encoded": round_enc,
        "is_knockout": is_knockout,
        "same_confederation": same_conf,
        "confederation_matchup": conf_matchup,
        "win_rate_diff": h_stats["win_rate"] - a_stats["win_rate"],
        "goals_pg_diff": home_goals_pg - away_goals_pg,
        "conceded_pg_diff": home_conceded_pg - away_conceded_pg,
        "rank_ratio": h_rank / (a_rank + 1),
        "ranking_diff_abs": abs(h_rank - a_rank),
    }
    return pd.DataFrame([features])


_team_stats_cache = {}
_team_stats_cache_key = None


def _compute_team_stats(wc_df: pd.DataFrame) -> dict:
    """Compute per-team win rate, draw rate, goals from WC history. Cached by dataframe shape+hash."""
    global _team_stats_cache, _team_stats_cache_key
    cache_key = (len(wc_df), wc_df["home_score"].sum() if len(wc_df) > 0 else 0)
    if cache_key == _team_stats_cache_key and _team_stats_cache:
        return _team_stats_cache

    # Vectorized approach using groupby instead of per-team loops
    home_df = wc_df[["home_team", "away_team", "home_score", "away_score", "outcome"]].copy()
    away_df = wc_df[["home_team", "away_team", "home_score", "away_score", "outcome"]].copy()

    home_df["team"] = home_df["home_team"]
    home_df["goals_for"] = home_df["home_score"].fillna(0)
    home_df["goals_against"] = home_df["away_score"].fillna(0)
    home_df["win"] = (home_df["outcome"] == "home_win").astype(int)
    home_df["draw"] = (home_df["outcome"] == "draw").astype(int)

    away_df["team"] = away_df["away_team"]
    away_df["goals_for"] = away_df["away_score"].fillna(0)
    away_df["goals_against"] = away_df["home_score"].fillna(0)
    away_df["win"] = (away_df["outcome"] == "away_win").astype(int)
    away_df["draw"] = (away_df["outcome"] == "draw").astype(int)

    combined = pd.concat([
        home_df[["team", "goals_for", "goals_against", "win", "draw"]],
        away_df[["team", "goals_for", "goals_against", "win", "draw"]]
    ], ignore_index=True)

    grouped = combined.groupby("team").agg(
        played=("win", "count"),
        wins=("win", "sum"),
        draws=("draw", "sum"),
        goals_for=("goals_for", "sum"),
        goals_against=("goals_against", "sum"),
    )

    stats = {}
    for team, row in grouped.iterrows():
        played = row["played"]
        if played == 0:
            stats[team] = {"win_rate": 0.33, "draw_rate": 0.20, "goals_pg": 1.5, "conceded_pg": 1.0}
        else:
            stats[team] = {
                "win_rate": row["wins"] / played,
                "draw_rate": row["draws"] / played,
                "goals_pg": row["goals_for"] / played,
                "conceded_pg": row["goals_against"] / played,
            }

    _team_stats_cache = stats
    _team_stats_cache_key = cache_key
    return stats


def _build_match_features(match, wc_df, team_stats) -> dict | None:
    """Build feature dict for a single historical match."""
    home_team = match["home_team"]
    away_team = match["away_team"]
    outcome = match.get("outcome")

    if pd.isna(outcome) or outcome not in ["home_win", "away_win", "draw"]:
        return None

    h_rank = get_ranking(home_team)
    a_rank = get_ranking(away_team)

    # H2H up to this match date
    prior = wc_df[wc_df["date"] < match["date"]]
    h2h_prior = prior[
        ((prior["home_team"] == home_team) & (prior["away_team"] == away_team)) |
        ((prior["home_team"] == away_team) & (prior["away_team"] == home_team))
    ]
    h2h_home_wins = (
        ((h2h_prior["home_team"] == home_team) & (h2h_prior["outcome"] == "home_win")) |
        ((h2h_prior["away_team"] == home_team) & (h2h_prior["outcome"] == "away_win"))
    ).sum()
    h2h_away_wins = (
        ((h2h_prior["home_team"] == away_team) & (h2h_prior["outcome"] == "home_win")) |
        ((h2h_prior["away_team"] == away_team) & (h2h_prior["outcome"] == "away_win"))
    ).sum()
    h2h_draws = (h2h_prior["outcome"] == "draw").sum()

    h_stats = team_stats.get(home_team, {"win_rate": 0.33, "draw_rate": 0.20, "goals_pg": 1.5, "conceded_pg": 1.0})
    a_stats = team_stats.get(away_team, {"win_rate": 0.33, "draw_rate": 0.20, "goals_pg": 1.5, "conceded_pg": 1.0})

    h_conf = get_confederation(home_team)
    a_conf = get_confederation(away_team)
    conf_matchup = f"{h_conf}_vs_{a_conf}" if h_conf <= a_conf else f"{a_conf}_vs_{h_conf}"

    round_name = match.get("round", "group")
    round_enc = ROUND_ENCODING.get(str(round_name), 0)
    is_knockout = 1 if round_enc >= 1 else 0

    year = match["date"].year if hasattr(match["date"], "year") else 2022

    return {
        "home_ranking": h_rank,
        "away_ranking": a_rank,
        "ranking_diff": h_rank - a_rank,
        "home_win_rate": h_stats["win_rate"],
        "away_win_rate": a_stats["win_rate"],
        "home_draw_rate": h_stats["draw_rate"],
        "away_draw_rate": a_stats["draw_rate"],
        "h2h_home_wins": h2h_home_wins,
        "h2h_away_wins": h2h_away_wins,
        "h2h_draws": h2h_draws,
        "h2h_total": len(h2h_prior),
        "h2h_has_history": 1 if len(h2h_prior) > 0 else 0,
        "home_goals_pg": h_stats["goals_pg"],
        "away_goals_pg": a_stats["goals_pg"],
        "home_conceded_pg": h_stats["conceded_pg"],
        "away_conceded_pg": a_stats["conceded_pg"],
        "home_rest_days": 7,
        "away_rest_days": 7,
        "host_nation": 1 if home_team in HOST_NATIONS_2026 else 0,
        "round_encoded": round_enc,
        "is_knockout": is_knockout,
        "same_confederation": 1 if h_conf == a_conf else 0,
        "confederation_matchup": conf_matchup,
        "win_rate_diff": h_stats["win_rate"] - a_stats["win_rate"],
        "goals_pg_diff": h_stats["goals_pg"] - a_stats["goals_pg"],
        "conceded_pg_diff": h_stats["conceded_pg"] - a_stats["conceded_pg"],
        "rank_ratio": h_rank / (a_rank + 1),
        "ranking_diff_abs": abs(h_rank - a_rank),
        "outcome": outcome,
    }


def postprocess_knockout_prediction(home_prob: float, draw_prob: float, away_prob: float, is_knockout: bool):
    """
    In knockout rounds, redistribute draw probability proportionally.
    Returns (home_prob, draw_prob, away_prob).
    """
    if not is_knockout:
        return home_prob, draw_prob, away_prob

    total_non_draw = home_prob + away_prob
    if total_non_draw == 0:
        return 0.5, 0.0, 0.5
    scale = 1.0 / total_non_draw
    return round(home_prob * scale, 4), 0.0, round(away_prob * scale, 4)
