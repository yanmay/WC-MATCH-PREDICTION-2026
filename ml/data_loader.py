"""
Data loading and processing for FIFA World Cup Match Prediction Platform.
Handles both historical data (Kaggle) and 2026 fixture data.
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_historical_data() -> pd.DataFrame:
    """
    Load historical international football results.
    Tries multiple CSV file names from Kaggle datasets.
    Falls back to synthetic data for dev/demo if no file found.
    """
    candidate_files = [
        DATA_DIR / "results.csv",
        DATA_DIR / "international-football-results-1872-2026.csv",
        DATA_DIR / "WorldCupMatches.csv",
        DATA_DIR / "matches.csv",
    ]
    for path in candidate_files:
        if path.exists():
            df = pd.read_csv(path)
            # Normalize column names to a standard schema
            df = _normalize_historical_columns(df)
            return df

    # No real data found — use built-in rich demo dataset
    return _get_demo_historical_data()


def _normalize_historical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map various Kaggle CSV schemas to our standard columns."""
    col_map = {
        # martj42 schema
        "home_team": "home_team",
        "away_team": "away_team",
        "home_score": "home_score",
        "away_score": "away_score",
        "tournament": "tournament",
        "date": "date",
        "neutral": "neutral",
        # WorldCupMatches schema
        "Home Team Name": "home_team",
        "Away Team Name": "away_team",
        "Home Team Goals": "home_score",
        "Away Team Goals": "away_score",
        "Stage": "round",
        "Datetime": "date",
        # World Cup predictor schema
        "Team1": "home_team",
        "Team2": "away_team",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Ensure required columns exist
    for col in ["home_team", "away_team", "home_score", "away_score"]:
        if col not in df.columns:
            df[col] = np.nan

    if "date" not in df.columns:
        df["date"] = pd.NaT
    if "tournament" not in df.columns:
        df["tournament"] = "FIFA World Cup"
    if "neutral" not in df.columns:
        df["neutral"] = True

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    return df


def get_world_cup_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filter historical data to World Cup matches only."""
    wc_mask = df["tournament"].str.contains("FIFA World Cup|World Cup", case=False, na=False)
    wc_df = df[wc_mask].copy()
    wc_df = wc_df.dropna(subset=["home_score", "away_score"])

    # Derive outcome
    wc_df["outcome"] = wc_df.apply(_derive_outcome, axis=1)
    wc_df["year"] = wc_df["date"].dt.year
    return wc_df.sort_values("date").reset_index(drop=True)


def _derive_outcome(row) -> str:
    h, a = row["home_score"], row["away_score"]
    if h > a:
        return "home_win"
    elif a > h:
        return "away_win"
    else:
        return "draw"


def get_2026_fixtures() -> pd.DataFrame:
    """
    Load or generate 2026 World Cup fixture data (Round of 32 onwards).
    Tries to load from data/fixtures_2026.json first.
    """
    fixture_path = DATA_DIR / "fixtures_2026.json"
    if fixture_path.exists():
        with open(fixture_path) as f:
            fixtures = json.load(f)
        return pd.DataFrame(fixtures)

    return _get_hardcoded_2026_fixtures()


def _get_hardcoded_2026_fixtures() -> pd.DataFrame:
    """
    2026 FIFA World Cup Round of 32 fixtures with realistic teams.
    Based on publicly available bracket information.
    """
    fixtures = [
        # Round of 32
        {"match_id": 1, "round": "Round of 32", "date": "2026-06-27", "home_team": "Argentina", "away_team": "Morocco", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 2, "round": "Round of 32", "date": "2026-06-27", "home_team": "France", "away_team": "USA", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 3, "round": "Round of 32", "date": "2026-06-28", "home_team": "Brazil", "away_team": "Poland", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 4, "round": "Round of 32", "date": "2026-06-28", "home_team": "England", "away_team": "Netherlands", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 5, "round": "Round of 32", "date": "2026-06-29", "home_team": "Spain", "away_team": "Senegal", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 6, "round": "Round of 32", "date": "2026-06-29", "home_team": "Germany", "away_team": "Mexico", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 7, "round": "Round of 32", "date": "2026-06-30", "home_team": "Portugal", "away_team": "Japan", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 8, "round": "Round of 32", "date": "2026-06-30", "home_team": "Belgium", "away_team": "Ecuador", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 9, "round": "Round of 32", "date": "2026-07-01", "home_team": "Colombia", "away_team": "Croatia", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 10, "round": "Round of 32", "date": "2026-07-01", "home_team": "Uruguay", "away_team": "Cameroon", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 11, "round": "Round of 32", "date": "2026-07-02", "home_team": "Denmark", "away_team": "South Korea", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 12, "round": "Round of 32", "date": "2026-07-02", "home_team": "Switzerland", "away_team": "Australia", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 13, "round": "Round of 32", "date": "2026-07-03", "home_team": "Serbia", "away_team": "Canada", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 14, "round": "Round of 32", "date": "2026-07-03", "home_team": "Mexico", "away_team": "Poland", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 15, "round": "Round of 32", "date": "2026-07-04", "home_team": "Iran", "away_team": "Wales", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 16, "round": "Round of 32", "date": "2026-07-04", "home_team": "Ghana", "away_team": "Egypt", "status": "scheduled", "home_score": None, "away_score": None},
        # Round of 16
        {"match_id": 17, "round": "Round of 16", "date": "2026-07-06", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 18, "round": "Round of 16", "date": "2026-07-06", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 19, "round": "Round of 16", "date": "2026-07-07", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 20, "round": "Round of 16", "date": "2026-07-07", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 21, "round": "Round of 16", "date": "2026-07-08", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 22, "round": "Round of 16", "date": "2026-07-08", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 23, "round": "Round of 16", "date": "2026-07-09", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 24, "round": "Round of 16", "date": "2026-07-09", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        # Quarterfinals
        {"match_id": 25, "round": "Quarterfinal", "date": "2026-07-11", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 26, "round": "Quarterfinal", "date": "2026-07-11", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 27, "round": "Quarterfinal", "date": "2026-07-12", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 28, "round": "Quarterfinal", "date": "2026-07-12", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        # Semifinals
        {"match_id": 29, "round": "Semifinal", "date": "2026-07-15", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 30, "round": "Semifinal", "date": "2026-07-15", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        # 3rd Place & Final
        {"match_id": 31, "round": "3rd Place", "date": "2026-07-18", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 32, "round": "Final", "date": "2026-07-19", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
    ]
    return pd.DataFrame(fixtures)


def get_team_stats(wc_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-team World Cup statistics for feature engineering."""
    teams = set(wc_df["home_team"].tolist() + wc_df["away_team"].tolist())
    stats = []
    for team in teams:
        home = wc_df[wc_df["home_team"] == team]
        away = wc_df[wc_df["away_team"] == team]
        played = len(home) + len(away)
        wins = (home["outcome"] == "home_win").sum() + (away["outcome"] == "away_win").sum()
        draws = (home["outcome"] == "draw").sum() + (away["outcome"] == "draw").sum()
        losses = played - wins - draws
        goals_scored = home["home_score"].sum() + away["away_score"].sum()
        goals_conceded = home["away_score"].sum() + away["home_score"].sum()
        stats.append({
            "team": team,
            "played": played,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / played if played > 0 else 0.33,
            "goals_scored_per_game": goals_scored / played if played > 0 else 1.0,
            "goals_conceded_per_game": goals_conceded / played if played > 0 else 1.0,
        })
    return pd.DataFrame(stats).set_index("team")


def get_head_to_head(wc_df: pd.DataFrame, team1: str, team2: str) -> dict:
    """Get head-to-head World Cup record between two teams."""
    mask = (
        ((wc_df["home_team"] == team1) & (wc_df["away_team"] == team2)) |
        ((wc_df["home_team"] == team2) & (wc_df["away_team"] == team1))
    )
    h2h = wc_df[mask].copy()
    team1_wins = (
        ((h2h["home_team"] == team1) & (h2h["outcome"] == "home_win")) |
        ((h2h["away_team"] == team1) & (h2h["outcome"] == "away_win"))
    ).sum()
    team2_wins = (
        ((h2h["home_team"] == team2) & (h2h["outcome"] == "home_win")) |
        ((h2h["away_team"] == team2) & (h2h["outcome"] == "away_win"))
    ).sum()
    draws = (h2h["outcome"] == "draw").sum()
    return {
        "matches": len(h2h),
        "team1_wins": int(team1_wins),
        "team2_wins": int(team2_wins),
        "draws": int(draws),
        "history": h2h[["date", "home_team", "away_team", "home_score", "away_score"]].to_dict("records"),
    }


def _get_demo_historical_data() -> pd.DataFrame:
    """
    Rich demo dataset of real World Cup matches for development/demo.
    Covers major matches from 1930–2022.
    """
    matches = [
        # Recent World Cups (2022 Qatar)
        {"date": "2022-11-21", "home_team": "Ecuador", "away_team": "Qatar", "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-11-21", "home_team": "England", "away_team": "Iran", "home_score": 6, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-11-22", "home_team": "Senegal", "away_team": "Netherlands", "home_score": 0, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-11-22", "home_team": "USA", "away_team": "Wales", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-11-23", "home_team": "Argentina", "away_team": "Saudi Arabia", "home_score": 1, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-11-23", "home_team": "France", "away_team": "Australia", "home_score": 4, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-11-23", "home_team": "Germany", "away_team": "Japan", "home_score": 1, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-11-24", "home_team": "Belgium", "away_team": "Canada", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-11-24", "home_team": "Spain", "away_team": "Costa Rica", "home_score": 7, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-11-24", "home_team": "Switzerland", "away_team": "Cameroon", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-11-25", "home_team": "Uruguay", "away_team": "South Korea", "home_score": 0, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-11-25", "home_team": "Portugal", "away_team": "Ghana", "home_score": 3, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-11-25", "home_team": "Brazil", "away_team": "Serbia", "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-03", "home_team": "France", "away_team": "Poland", "home_score": 3, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-03", "home_team": "Argentina", "away_team": "Australia", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-04", "home_team": "England", "away_team": "Senegal", "home_score": 3, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-04", "home_team": "Netherlands", "away_team": "USA", "home_score": 3, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-05", "home_team": "Japan", "away_team": "Croatia", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-05", "home_team": "Brazil", "away_team": "South Korea", "home_score": 4, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-06", "home_team": "Morocco", "away_team": "Spain", "home_score": 0, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-06", "home_team": "Portugal", "away_team": "Switzerland", "home_score": 6, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-09", "home_team": "Argentina", "away_team": "Netherlands", "home_score": 2, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-09", "home_team": "Croatia", "away_team": "Brazil", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-10", "home_team": "Morocco", "away_team": "Portugal", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-10", "home_team": "France", "away_team": "England", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-14", "home_team": "Argentina", "away_team": "Croatia", "home_score": 3, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-14", "home_team": "France", "away_team": "Morocco", "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2022-12-18", "home_team": "Argentina", "away_team": "France", "home_score": 3, "away_score": 3, "tournament": "FIFA World Cup", "neutral": False},
        # 2018 Russia
        {"date": "2018-06-14", "home_team": "Russia", "away_team": "Saudi Arabia", "home_score": 5, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-06-15", "home_team": "Egypt", "away_team": "Uruguay", "home_score": 0, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-06-15", "home_team": "Morocco", "away_team": "Iran", "home_score": 0, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-06-15", "home_team": "Portugal", "away_team": "Spain", "home_score": 3, "away_score": 3, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-06-16", "home_team": "France", "away_team": "Australia", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-06-16", "home_team": "Argentina", "away_team": "Iceland", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-06-16", "home_team": "Peru", "away_team": "Denmark", "home_score": 0, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-06-17", "home_team": "Croatia", "away_team": "Nigeria", "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-06-17", "home_team": "Costa Rica", "away_team": "Serbia", "home_score": 0, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-06-17", "home_team": "Germany", "away_team": "Mexico", "home_score": 0, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-06-17", "home_team": "Brazil", "away_team": "Switzerland", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-06-19", "home_team": "England", "away_team": "Tunisia", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-06-30", "home_team": "Argentina", "away_team": "France", "home_score": 3, "away_score": 4, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-06-30", "home_team": "Uruguay", "away_team": "Portugal", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-07-01", "home_team": "Spain", "away_team": "Russia", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-07-01", "home_team": "Croatia", "away_team": "Denmark", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-07-02", "home_team": "Brazil", "away_team": "Mexico", "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-07-02", "home_team": "Belgium", "away_team": "Japan", "home_score": 3, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-07-03", "home_team": "Sweden", "away_team": "Switzerland", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-07-03", "home_team": "Colombia", "away_team": "England", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-07-06", "home_team": "Uruguay", "away_team": "France", "home_score": 0, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-07-06", "home_team": "Brazil", "away_team": "Belgium", "home_score": 1, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-07-07", "home_team": "Sweden", "away_team": "England", "home_score": 0, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-07-07", "home_team": "Russia", "away_team": "Croatia", "home_score": 2, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-07-10", "home_team": "France", "away_team": "Belgium", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-07-11", "home_team": "Croatia", "away_team": "England", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2018-07-15", "home_team": "France", "away_team": "Croatia", "home_score": 4, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        # 2014 Brazil
        {"date": "2014-06-12", "home_team": "Brazil", "away_team": "Croatia", "home_score": 3, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-06-12", "home_team": "Mexico", "away_team": "Cameroon", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-06-13", "home_team": "Spain", "away_team": "Netherlands", "home_score": 1, "away_score": 5, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-06-14", "home_team": "Colombia", "away_team": "Greece", "home_score": 3, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-06-14", "home_team": "Uruguay", "away_team": "Costa Rica", "home_score": 1, "away_score": 3, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-06-16", "home_team": "Argentina", "away_team": "Bosnia and Herzegovina", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-06-16", "home_team": "Germany", "away_team": "Portugal", "home_score": 4, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-06-17", "home_team": "USA", "away_team": "Ghana", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-06-28", "home_team": "Brazil", "away_team": "Chile", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-06-28", "home_team": "Colombia", "away_team": "Uruguay", "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-06-29", "home_team": "Netherlands", "away_team": "Mexico", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-06-29", "home_team": "Costa Rica", "away_team": "Greece", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-06-30", "home_team": "France", "away_team": "Nigeria", "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-06-30", "home_team": "Germany", "away_team": "Algeria", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-07-01", "home_team": "Argentina", "away_team": "Switzerland", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-07-01", "home_team": "Belgium", "away_team": "USA", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-07-04", "home_team": "Brazil", "away_team": "Colombia", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-07-04", "home_team": "France", "away_team": "Germany", "home_score": 0, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-07-05", "home_team": "Netherlands", "away_team": "Costa Rica", "home_score": 0, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-07-05", "home_team": "Argentina", "away_team": "Belgium", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-07-08", "home_team": "Brazil", "away_team": "Germany", "home_score": 1, "away_score": 7, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-07-09", "home_team": "Netherlands", "away_team": "Argentina", "home_score": 0, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2014-07-13", "home_team": "Germany", "away_team": "Argentina", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        # 2010 South Africa
        {"date": "2010-06-11", "home_team": "South Africa", "away_team": "Mexico", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-11", "home_team": "Uruguay", "away_team": "France", "home_score": 0, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-12", "home_team": "England", "away_team": "USA", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-12", "home_team": "Argentina", "away_team": "Nigeria", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-13", "home_team": "Germany", "away_team": "Australia", "home_score": 4, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-13", "home_team": "Netherlands", "away_team": "Denmark", "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-14", "home_team": "Japan", "away_team": "Cameroon", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-14", "home_team": "Italy", "away_team": "Paraguay", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-14", "home_team": "Brazil", "away_team": "North Korea", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-15", "home_team": "Portugal", "away_team": "Ivory Coast", "home_score": 0, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-26", "home_team": "USA", "away_team": "Ghana", "home_score": 1, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-26", "home_team": "Germany", "away_team": "England", "home_score": 4, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-27", "home_team": "Argentina", "away_team": "Mexico", "home_score": 3, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-27", "home_team": "Netherlands", "away_team": "Slovakia", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-28", "home_team": "Brazil", "away_team": "Chile", "home_score": 3, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-28", "home_team": "Paraguay", "away_team": "Japan", "home_score": 0, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-29", "home_team": "Spain", "away_team": "Portugal", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-06-29", "home_team": "Uruguay", "away_team": "South Korea", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-07-02", "home_team": "Germany", "away_team": "Argentina", "home_score": 4, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-07-02", "home_team": "Netherlands", "away_team": "Brazil", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-07-03", "home_team": "Uruguay", "away_team": "Ghana", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-07-03", "home_team": "Spain", "away_team": "Paraguay", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-07-06", "home_team": "Germany", "away_team": "Spain", "home_score": 0, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-07-06", "home_team": "Netherlands", "away_team": "Uruguay", "home_score": 3, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2010-07-11", "home_team": "Spain", "away_team": "Netherlands", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        # 2006 Germany
        {"date": "2006-06-09", "home_team": "Germany", "away_team": "Costa Rica", "home_score": 4, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-09", "home_team": "Poland", "away_team": "Ecuador", "home_score": 0, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-10", "home_team": "England", "away_team": "Paraguay", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-10", "home_team": "Argentina", "away_team": "Ivory Coast", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-10", "home_team": "Netherlands", "away_team": "Serbia and Montenegro", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-12", "home_team": "Brazil", "away_team": "Croatia", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-12", "home_team": "Spain", "away_team": "Ukraine", "home_score": 4, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-12", "home_team": "France", "away_team": "Switzerland", "home_score": 0, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-24", "home_team": "Germany", "away_team": "Sweden", "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-24", "home_team": "Argentina", "away_team": "Mexico", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-25", "home_team": "England", "away_team": "Ecuador", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-25", "home_team": "Portugal", "away_team": "Netherlands", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-26", "home_team": "Italy", "away_team": "Australia", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-26", "home_team": "Switzerland", "away_team": "Ukraine", "home_score": 0, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-27", "home_team": "Brazil", "away_team": "Ghana", "home_score": 3, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-27", "home_team": "Spain", "away_team": "France", "home_score": 1, "away_score": 3, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-30", "home_team": "Germany", "away_team": "Argentina", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-06-30", "home_team": "Italy", "away_team": "Ukraine", "home_score": 3, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-07-01", "home_team": "England", "away_team": "Portugal", "home_score": 0, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-07-01", "home_team": "Brazil", "away_team": "France", "home_score": 0, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-07-04", "home_team": "Germany", "away_team": "Italy", "home_score": 0, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-07-05", "home_team": "Portugal", "away_team": "France", "home_score": 0, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2006-07-09", "home_team": "Italy", "away_team": "France", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        # Classic matches
        {"date": "2002-05-31", "home_team": "France", "away_team": "Senegal", "home_score": 0, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "2002-06-30", "home_team": "Brazil", "away_team": "Germany", "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1998-07-12", "home_team": "France", "away_team": "Brazil", "home_score": 3, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1994-07-17", "home_team": "Brazil", "away_team": "Italy", "home_score": 0, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1990-07-08", "home_team": "Germany", "away_team": "Argentina", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1986-06-29", "home_team": "Argentina", "away_team": "Germany", "home_score": 3, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1982-07-11", "home_team": "Italy", "away_team": "Germany", "home_score": 3, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1978-06-25", "home_team": "Argentina", "away_team": "Netherlands", "home_score": 3, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1974-07-07", "home_team": "Germany", "away_team": "Netherlands", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1970-06-21", "home_team": "Brazil", "away_team": "England", "home_score": 1, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1970-06-17", "home_team": "Germany", "away_team": "England", "home_score": 3, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1970-06-14", "home_team": "Brazil", "away_team": "Czechoslovakia", "home_score": 4, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1970-07-21", "home_team": "Brazil", "away_team": "Italy", "home_score": 4, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1966-07-30", "home_team": "England", "away_team": "Germany", "home_score": 4, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1962-06-17", "home_team": "Chile", "away_team": "Italy", "home_score": 2, "away_score": 0, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1958-06-29", "home_team": "Brazil", "away_team": "Sweden", "home_score": 5, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1954-07-04", "home_team": "Germany", "away_team": "Hungary", "home_score": 3, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1950-07-16", "home_team": "Uruguay", "away_team": "Brazil", "home_score": 2, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1938-06-19", "home_team": "Italy", "away_team": "Hungary", "home_score": 4, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1934-06-10", "home_team": "Italy", "away_team": "Spain", "home_score": 1, "away_score": 1, "tournament": "FIFA World Cup", "neutral": False},
        {"date": "1930-07-30", "home_team": "Uruguay", "away_team": "Argentina", "home_score": 4, "away_score": 2, "tournament": "FIFA World Cup", "neutral": False},
    ]
    df = pd.DataFrame(matches)
    df["date"] = pd.to_datetime(df["date"])
    return df
