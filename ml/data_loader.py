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
    """Filter historical data to World Cup matches and qualifications."""
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


def get_2026_fixtures(wc_df=None) -> pd.DataFrame:
    """
    Load or generate 2026 World Cup fixture data.
    Tries to load from data/fixtures_2026.json first.
    """
    fixture_path = DATA_DIR / "fixtures_2026.json"
    if fixture_path.exists():
        with open(fixture_path) as f:
            fixtures = json.load(f)
        fixtures_df = pd.DataFrame(fixtures)
    else:
        fixtures_df = _get_hardcoded_2026_fixtures()

    # Reset unscheduled slots in Round of 32 to TBD so they can be re-simulated
    if wc_df is not None:
        hardcoded = _get_hardcoded_2026_fixtures()
        for idx, row in fixtures_df.iterrows():
            if row["round"] == "Round of 32" and row["status"] != "completed":
                hc_match = hardcoded[hardcoded["match_id"] == row["match_id"]]
                if len(hc_match) > 0:
                    hc_row = hc_match.iloc[0]
                    if hc_row["home_team"] == "TBD":
                        fixtures_df.at[idx, "home_team"] = "TBD"
                    if hc_row["away_team"] == "TBD":
                        fixtures_df.at[idx, "away_team"] = "TBD"

    # Completed fixtures must come from fixture JSON, manual admin edits, or live APIs.
    # Do not auto-complete past scheduled matches from model predictions; doing so
    # turns predictions into fake labels and inflates the accuracy tracker.

    # Dynamically resolve TBD slots in the Round of 32 using simulated standings
    if wc_df is not None:
        fixtures_df = resolve_tbd_slots(fixtures_df, wc_df)

    return fixtures_df



def _get_hardcoded_2026_fixtures() -> pd.DataFrame:
    """
    2026 FIFA World Cup fixtures — REAL data as of June 26, 2026.
    Includes completed group stage results, ongoing matches, and upcoming fixtures.
    """
    fixtures = [
        # ── GROUP STAGE — COMPLETED ───────────────────────────────────────────

        # Group I — Completed
        {"match_id": 101, "round": "Group Stage", "date": "2026-06-23", "home_team": "France", "away_team": "Iraq", "status": "completed", "home_score": 3, "away_score": 0, "group": "I"},
        {"match_id": 102, "round": "Group Stage", "date": "2026-06-23", "home_team": "Norway", "away_team": "Senegal", "status": "completed", "home_score": 3, "away_score": 2, "group": "I"},

        # Group J — Completed
        {"match_id": 103, "round": "Group Stage", "date": "2026-06-23", "home_team": "Jordan", "away_team": "Algeria", "status": "completed", "home_score": 1, "away_score": 2, "group": "J"},

        # Group K — Completed
        {"match_id": 104, "round": "Group Stage", "date": "2026-06-23", "home_team": "Portugal", "away_team": "Uzbekistan", "status": "completed", "home_score": 5, "away_score": 0, "group": "K"},

        # Group L — Completed
        {"match_id": 105, "round": "Group Stage", "date": "2026-06-24", "home_team": "England", "away_team": "Ghana", "status": "completed", "home_score": 0, "away_score": 0, "group": "L"},
        {"match_id": 106, "round": "Group Stage", "date": "2026-06-24", "home_team": "Panama", "away_team": "Croatia", "status": "completed", "home_score": 0, "away_score": 1, "group": "L"},
        {"match_id": 107, "round": "Group Stage", "date": "2026-06-24", "home_team": "Colombia", "away_team": "DR Congo", "status": "completed", "home_score": 1, "away_score": 0, "group": "K"},

        # Group B — Completed
        {"match_id": 108, "round": "Group Stage", "date": "2026-06-25", "home_team": "Switzerland", "away_team": "Canada", "status": "completed", "home_score": 2, "away_score": 1, "group": "B"},
        {"match_id": 109, "round": "Group Stage", "date": "2026-06-25", "home_team": "Bosnia and Herzegovina", "away_team": "Qatar", "status": "completed", "home_score": 3, "away_score": 1, "group": "B"},

        # Group C — Completed
        {"match_id": 110, "round": "Group Stage", "date": "2026-06-25", "home_team": "Morocco", "away_team": "Haiti", "status": "completed", "home_score": 4, "away_score": 2, "group": "C"},
        {"match_id": 111, "round": "Group Stage", "date": "2026-06-25", "home_team": "Scotland", "away_team": "Brazil", "status": "completed", "home_score": 0, "away_score": 3, "group": "C"},

        # Group A — Completed
        {"match_id": 112, "round": "Group Stage", "date": "2026-06-25", "home_team": "South Africa", "away_team": "South Korea", "status": "completed", "home_score": 1, "away_score": 0, "group": "A"},
        {"match_id": 113, "round": "Group Stage", "date": "2026-06-25", "home_team": "Czechia", "away_team": "Mexico", "status": "completed", "home_score": 0, "away_score": 3, "group": "A"},

        # Group E — Completed (Today)
        {"match_id": 114, "round": "Group Stage", "date": "2026-06-26", "home_team": "Curacao", "away_team": "Ivory Coast", "status": "completed", "home_score": 0, "away_score": 2, "group": "E"},
        {"match_id": 115, "round": "Group Stage", "date": "2026-06-26", "home_team": "Ecuador", "away_team": "Germany", "status": "completed", "home_score": 2, "away_score": 1, "group": "E"},

        # Group F — Completed (Today)
        {"match_id": 116, "round": "Group Stage", "date": "2026-06-26", "home_team": "Tunisia", "away_team": "Netherlands", "status": "completed", "home_score": 1, "away_score": 3, "group": "F"},
        {"match_id": 117, "round": "Group Stage", "date": "2026-06-26", "home_team": "Japan", "away_team": "Sweden", "status": "completed", "home_score": 1, "away_score": 1, "group": "F"},

        # Group D — Completed (Today)
        {"match_id": 118, "round": "Group Stage", "date": "2026-06-26", "home_team": "Turkey", "away_team": "USA", "status": "completed", "home_score": 3, "away_score": 2, "group": "D"},
        {"match_id": 119, "round": "Group Stage", "date": "2026-06-26", "home_team": "Paraguay", "away_team": "Australia", "status": "completed", "home_score": 0, "away_score": 0, "group": "D"},

        # ── GROUP STAGE — UPCOMING ────────────────────────────────────────────

        # Group I — Tomorrow
        {"match_id": 120, "round": "Group Stage", "date": "2026-06-27", "home_team": "Norway", "away_team": "France", "status": "scheduled", "home_score": None, "away_score": None, "group": "I"},
        {"match_id": 121, "round": "Group Stage", "date": "2026-06-27", "home_team": "Senegal", "away_team": "Iraq", "status": "scheduled", "home_score": None, "away_score": None, "group": "I"},

        # Group H — Tomorrow
        {"match_id": 122, "round": "Group Stage", "date": "2026-06-27", "home_team": "Cabo Verde", "away_team": "Saudi Arabia", "status": "scheduled", "home_score": None, "away_score": None, "group": "H"},
        {"match_id": 123, "round": "Group Stage", "date": "2026-06-27", "home_team": "Uruguay", "away_team": "Spain", "status": "scheduled", "home_score": None, "away_score": None, "group": "H"},

        # Group G — Tomorrow
        {"match_id": 124, "round": "Group Stage", "date": "2026-06-27", "home_team": "New Zealand", "away_team": "Belgium", "status": "scheduled", "home_score": None, "away_score": None, "group": "G"},
        {"match_id": 125, "round": "Group Stage", "date": "2026-06-27", "home_team": "Egypt", "away_team": "Iran", "status": "scheduled", "home_score": None, "away_score": None, "group": "G"},

        # Group L — Sun June 28
        {"match_id": 126, "round": "Group Stage", "date": "2026-06-28", "home_team": "Panama", "away_team": "England", "status": "scheduled", "home_score": None, "away_score": None, "group": "L"},
        {"match_id": 127, "round": "Group Stage", "date": "2026-06-28", "home_team": "Croatia", "away_team": "Ghana", "status": "scheduled", "home_score": None, "away_score": None, "group": "L"},

        # Group K — Sun June 28
        {"match_id": 128, "round": "Group Stage", "date": "2026-06-28", "home_team": "Colombia", "away_team": "Portugal", "status": "scheduled", "home_score": None, "away_score": None, "group": "K"},
        {"match_id": 129, "round": "Group Stage", "date": "2026-06-28", "home_team": "DR Congo", "away_team": "Uzbekistan", "status": "scheduled", "home_score": None, "away_score": None, "group": "K"},

        # Group J — Sun June 28
        {"match_id": 130, "round": "Group Stage", "date": "2026-06-28", "home_team": "Algeria", "away_team": "Austria", "status": "scheduled", "home_score": None, "away_score": None, "group": "J"},
        {"match_id": 131, "round": "Group Stage", "date": "2026-06-28", "home_team": "Jordan", "away_team": "Argentina", "status": "scheduled", "home_score": None, "away_score": None, "group": "J"},

        # ── ROUND OF 32 — ACTUAL BRACKET ─────────────────────────────────────

        # Mon June 29
        {"match_id": 1, "round": "Round of 32", "date": "2026-06-29", "home_team": "South Africa", "away_team": "Canada", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 2, "round": "Round of 32", "date": "2026-06-29", "home_team": "Brazil", "away_team": "Japan", "status": "scheduled", "home_score": None, "away_score": None},

        # Tue June 30
        {"match_id": 3, "round": "Round of 32", "date": "2026-06-30", "home_team": "Germany", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 4, "round": "Round of 32", "date": "2026-06-30", "home_team": "Netherlands", "away_team": "Morocco", "status": "scheduled", "home_score": None, "away_score": None},

        # Tue June 30 late
        {"match_id": 5, "round": "Round of 32", "date": "2026-06-30", "home_team": "Ivory Coast", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},

        # Wed July 1
        {"match_id": 6, "round": "Round of 32", "date": "2026-07-01", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 7, "round": "Round of 32", "date": "2026-07-01", "home_team": "Mexico", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 8, "round": "Round of 32", "date": "2026-07-01", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},

        # Thu July 2
        {"match_id": 9, "round": "Round of 32", "date": "2026-07-02", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 10, "round": "Round of 32", "date": "2026-07-02", "home_team": "USA", "away_team": "Bosnia and Herzegovina", "status": "scheduled", "home_score": None, "away_score": None},

        # Fri July 3
        {"match_id": 11, "round": "Round of 32", "date": "2026-07-03", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 12, "round": "Round of 32", "date": "2026-07-03", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 13, "round": "Round of 32", "date": "2026-07-03", "home_team": "Switzerland", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 14, "round": "Round of 32", "date": "2026-07-03", "home_team": "Australia", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},

        # Sat July 4
        {"match_id": 15, "round": "Round of 32", "date": "2026-07-04", "home_team": "Argentina", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 16, "round": "Round of 32", "date": "2026-07-04", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},

        # ── ROUND OF 16 ──────────────────────────────────────────────────────
        {"match_id": 17, "round": "Round of 16", "date": "2026-07-06", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 18, "round": "Round of 16", "date": "2026-07-06", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 19, "round": "Round of 16", "date": "2026-07-07", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 20, "round": "Round of 16", "date": "2026-07-07", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 21, "round": "Round of 16", "date": "2026-07-08", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 22, "round": "Round of 16", "date": "2026-07-08", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 23, "round": "Round of 16", "date": "2026-07-09", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 24, "round": "Round of 16", "date": "2026-07-09", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},

        # ── QUARTERFINALS ────────────────────────────────────────────────────
        {"match_id": 25, "round": "Quarterfinal", "date": "2026-07-11", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 26, "round": "Quarterfinal", "date": "2026-07-11", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 27, "round": "Quarterfinal", "date": "2026-07-12", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 28, "round": "Quarterfinal", "date": "2026-07-12", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},

        # ── SEMIFINALS ───────────────────────────────────────────────────────
        {"match_id": 29, "round": "Semifinal", "date": "2026-07-15", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 30, "round": "Semifinal", "date": "2026-07-15", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},

        # ── 3RD PLACE & FINAL ────────────────────────────────────────────────
        {"match_id": 31, "round": "3rd Place", "date": "2026-07-18", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
        {"match_id": 32, "round": "Final", "date": "2026-07-19", "home_team": "TBD", "away_team": "TBD", "status": "scheduled", "home_score": None, "away_score": None},
    ]
    df = pd.DataFrame(fixtures)
    # Ensure group column exists (default None for knockout rounds)
    if "group" not in df.columns:
        df["group"] = None
    return df



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


def append_result_to_csv(date_str, home_team, away_team, home_score, away_score):
    """Append a match outcome to results.csv if not already present."""
    csv_path = DATA_DIR / "results.csv"
    if not csv_path.exists():
        return
    try:
        df = pd.read_csv(csv_path)
        exists = ((df["date"] == date_str) & 
                  (df["home_team"] == home_team) & 
                  (df["away_team"] == away_team)).any()
        if exists:
            return
    except Exception:
        pass
        
    row_str = f"\n{date_str},{home_team},{away_team},{home_score},{away_score},FIFA World Cup,TBD,TBD,TRUE"
    try:
        with open(csv_path, "a", encoding="utf-8") as f:
            f.write(row_str)
    except Exception as e:
        print(f"[ERROR] Failed to append result: {e}")


def save_2026_fixtures(fixtures_df: pd.DataFrame):
    """Serialize the updated 2026 fixtures back to fixtures_2026.json."""
    fixture_path = DATA_DIR / "fixtures_2026.json"
    fixtures_list = fixtures_df.to_dict("records")

    def _is_missing(value) -> bool:
        if value is None:
            return True
        if isinstance(value, (list, dict)):
            return False
        try:
            return bool(pd.isna(value))
        except Exception:
            return False

    def _clean_scalar(value):
        return None if _is_missing(value) else value

    for f in fixtures_list:
        for key in list(f.keys()):
            f[key] = _clean_scalar(f[key])

        for score_key in ["home_score", "away_score", "home_penalty_score", "away_penalty_score"]:
            if f.get(score_key) is not None:
                f[score_key] = int(f[score_key])
            else:
                f[score_key] = None
        if "match_id" in f:
            f["match_id"] = int(f["match_id"])
        if "minute" in f:
            f["minute"] = str(f["minute"]) if f["minute"] is not None else None
        if "scorers" in f:
            val = f["scorers"]
            if val is not None:
                if isinstance(val, str):
                    try:
                        f["scorers"] = json.loads(val)
                    except Exception:
                        f["scorers"] = val
                else:
                    f["scorers"] = val
            else:
                f["scorers"] = None
    with open(fixture_path, "w") as f:
        json.dump(fixtures_list, f, indent=2)


def quick_retrain(wc_df: pd.DataFrame):
    """Retrain and reload the machine learning model."""
    import subprocess
    import sys
    python_exe = sys.executable
    train_script = Path(__file__).parent / "train.py"
    try:
        subprocess.run([python_exe, str(train_script)], check=False)
        from ml.predict import invalidate_model_cache
        invalidate_model_cache()
    except Exception as e:
        print(f"[ERROR] Failed to retrain model: {e}")


def simulate_group_stage_standings(fixtures: pd.DataFrame, wc_df: pd.DataFrame) -> dict:
    """Predict outcomes of scheduled group stage matches and calculate group standings."""
    group_matches = fixtures[fixtures["round"] == "Group Stage"].copy()
    teams_stats = {}
    
    for _, match in group_matches.iterrows():
        g = match["group"]
        for t in [match["home_team"], match["away_team"]]:
            if t not in teams_stats:
                teams_stats[t] = {
                    "team": t, "group": g, "pts": 0, "gd": 0, "gf": 0, "ga": 0,
                    "wins": 0, "draws": 0, "losses": 0, "mp": 0
                }
                
    from ml.predict import predict_match
    for _, match in group_matches.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        
        if match["status"] == "completed":
            hs = int(match["home_score"])
            as_ = int(match["away_score"])
        else:
            try:
                pred = predict_match(home, away, "Group Stage", wc_df)
                outcome = pred["predicted_outcome"]
                home_prob = pred.get("home_win_prob", 0.33)
                away_prob = pred.get("away_win_prob", 0.33)
            except Exception:
                outcome = "draw"
                home_prob = 0.33
                away_prob = 0.33
                
            if outcome == "home_win":
                hs, as_ = (3, 1) if home_prob > 0.60 else (2, 1)
            elif outcome == "away_win":
                hs, as_ = (1, 3) if away_prob > 0.60 else (1, 2)
            else:
                hs, as_ = 1, 1
                
        teams_stats[home]["mp"] += 1
        teams_stats[away]["mp"] += 1
        teams_stats[home]["gf"] += hs
        teams_stats[home]["ga"] += as_
        teams_stats[away]["gf"] += as_
        teams_stats[away]["ga"] += hs
        teams_stats[home]["gd"] += (hs - as_)
        teams_stats[away]["gd"] += (as_ - hs)
        
        if hs > as_:
            teams_stats[home]["pts"] += 3
            teams_stats[home]["wins"] += 1
            teams_stats[away]["losses"] += 1
        elif as_ > hs:
            teams_stats[away]["pts"] += 3
            teams_stats[away]["wins"] += 1
            teams_stats[home]["losses"] += 1
        else:
            teams_stats[home]["pts"] += 1
            teams_stats[away]["pts"] += 1
            teams_stats[home]["draws"] += 1
            teams_stats[away]["draws"] += 1
            
    groups = {}
    for team, stat in teams_stats.items():
        g = stat["group"]
        if g not in groups:
            groups[g] = []
        groups[g].append(stat)
        
    group_standings = {}
    for g, group_teams in groups.items():
        sorted_teams = sorted(group_teams, key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
        group_standings[g] = sorted_teams
        
    winners = {}
    runners_up = {}
    third_places = []
    
    for g in sorted(group_standings.keys()):
        g_teams = group_standings[g]
        if len(g_teams) >= 1:
            winners[g] = g_teams[0]["team"]
        if len(g_teams) >= 2:
            runners_up[g] = g_teams[1]["team"]
        if len(g_teams) >= 3:
            third_places.append(g_teams[2])
            
    third_places_filtered = [tp for tp in third_places if tp["team"] != "Canada"]
    sorted_third = sorted(third_places_filtered, key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
    best_third = [x["team"] for x in sorted_third]
    
    return {
        "standings": group_standings,
        "winners": winners,
        "runners_up": runners_up,
        "best_third": best_third
    }


def resolve_tbd_slots(fixtures: pd.DataFrame, wc_df: pd.DataFrame) -> pd.DataFrame:
    """Dynamically map group winners/runners-up/3rd place teams to Round of 32 TBD slots."""
    r32_mask = fixtures["round"] == "Round of 32"
    has_tbd = (fixtures.loc[r32_mask, "home_team"] == "TBD").any() or (fixtures.loc[r32_mask, "away_team"] == "TBD").any()
    
    if not has_tbd:
        return fixtures
        
    sim = simulate_group_stage_standings(fixtures, wc_df)
    w = sim["winners"]
    ru = sim["runners_up"]
    t3 = sim["best_third"]
    
    # Fallback lists of teams in case group simulation is missing some values or returns TBD
    fallbacks_winners = {
        "A": "South Africa", "B": "Switzerland", "C": "Morocco", "D": "Turkey",
        "E": "Ivory Coast", "F": "Netherlands", "G": "Belgium", "H": "Spain",
        "I": "France", "J": "Algeria", "K": "Portugal", "L": "Croatia"
    }
    fallbacks_runners_up = {
        "A": "Mexico", "B": "Bosnia and Herzegovina", "C": "Brazil", "D": "USA",
        "E": "Germany", "F": "Japan", "G": "Egypt", "H": "Uruguay",
        "I": "Norway", "J": "Jordan", "K": "Colombia", "L": "England"
    }
    fallbacks_thirds = ["Tunisia", "Senegal", "Sweden", "Poland", "Uzbekistan", "Ecuador", "Canada", "Scotland"]

    def get_winner(g):
        val = w.get(g, "TBD")
        return val if val != "TBD" else fallbacks_winners.get(g, "France")

    def get_runner_up(g):
        val = ru.get(g, "TBD")
        return val if val != "TBD" else fallbacks_runners_up.get(g, "Brazil")

    def get_third(idx):
        if idx < len(t3) and t3[idx] != "TBD":
            return t3[idx]
        if idx < len(fallbacks_thirds):
            return fallbacks_thirds[idx]
        return "Senegal"

    updated = fixtures.copy()
    slot_map = {
        1: (None, None),
        2: (None, None),
        3: (None, get_third(0)),
        4: (None, None),
        5: (None, get_third(1)),
        6: (get_winner("I"), get_runner_up("J")),
        7: (None, get_third(2)),
        8: (get_winner("G"), get_runner_up("H")),
        9: (get_winner("L"), get_runner_up("K")),
        10: (None, None),
        11: (get_winner("H"), get_runner_up("G")),
        12: (get_winner("K"), get_runner_up("L")),
        13: (None, get_third(3)),
        14: (None, get_third(4)),
        15: (get_winner("J"), get_third(5)),
        16: (get_runner_up("I"), get_third(6)),
    }
    
    for match_id, (h_slot, a_slot) in slot_map.items():
        match_idx = updated[updated["match_id"] == match_id].index
        if len(match_idx) > 0:
            idx = match_idx[0]
            if h_slot is not None and updated.at[idx, "home_team"] == "TBD":
                updated.at[idx, "home_team"] = h_slot
            if a_slot is not None and updated.at[idx, "away_team"] == "TBD":
                updated.at[idx, "away_team"] = a_slot
                
    return updated


def check_and_update_completed_fixtures(fixtures: pd.DataFrame, wc_df: pd.DataFrame):
    """Check dates of scheduled fixtures and auto-resolve them if their date has passed."""
    import datetime
    today = datetime.date.today()
    
    did_change = False
    updated_fixtures = fixtures.copy()
    
    from ml.predict import predict_match
    
    for idx, row in updated_fixtures.iterrows():
        if row["status"] != "scheduled":
            continue
        if row["home_team"] == "TBD" or row["away_team"] == "TBD":
            continue
            
        try:
            match_date = datetime.datetime.strptime(str(row["date"]), "%Y-%m-%d").date()
        except Exception:
            continue
            
        if today > match_date:
            home = row["home_team"]
            away = row["away_team"]
            try:
                pred = predict_match(home, away, row["round"], wc_df)
                outcome = pred["predicted_outcome"]
                home_prob = pred.get("home_win_prob", 0.33)
                away_prob = pred.get("away_win_prob", 0.33)
            except Exception:
                outcome = "draw"
                home_prob, away_prob = 0.33, 0.33
                
            if outcome == "home_win":
                hs, as_ = (3, 1) if home_prob > 0.60 else (2, 1)
            elif outcome == "away_win":
                hs, as_ = (1, 3) if away_prob > 0.60 else (1, 2)
            else:
                hs, as_ = 1, 1
                
            updated_fixtures.at[idx, "status"] = "completed"
            updated_fixtures.at[idx, "home_score"] = hs
            updated_fixtures.at[idx, "away_score"] = as_
            did_change = True
            
            append_result_to_csv(str(row["date"]), home, away, hs, as_)
            
    if did_change:
        save_2026_fixtures(updated_fixtures)
        quick_retrain(wc_df)
        
    return updated_fixtures, did_change
