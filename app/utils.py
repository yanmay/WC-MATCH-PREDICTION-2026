"""
Shared state, 21st.dev-inspired CSS design system, and helper utilities.
"""

import streamlit as st
import pandas as pd
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@st.cache_data(show_spinner=False)
def load_data():
    from ml.data_loader import load_historical_data, get_world_cup_data, get_2026_fixtures
    raw = load_historical_data()
    wc_df = get_world_cup_data(raw)
    fixtures = get_2026_fixtures()
    return wc_df, fixtures


@st.cache_resource(show_spinner=False)
def get_model_and_metrics():
    from ml.data_loader import load_historical_data, get_world_cup_data
    from ml.features import build_training_features
    from ml.train import auto_train_best_model, save_model, load_model

    ARTIFACTS_DIR = ROOT / "ml" / "artifacts"
    active_ptr = ARTIFACTS_DIR / "active_model.txt"

    if active_ptr.exists():
        try:
            pipeline, metrics = load_model()
            return pipeline, metrics
        except Exception:
            pass

    raw = load_historical_data()
    wc_df = get_world_cup_data(raw)
    X, y = build_training_features(wc_df)
    result = auto_train_best_model(X, y)
    pipeline = result["pipeline"]
    metrics = result["metrics"]
    save_model(pipeline, metrics, version="v1.0")
    return pipeline, metrics


def get_team_flag(team: str) -> str:
    FLAGS = {
        "Argentina": "🇦🇷", "France": "🇫🇷", "England": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
        "Belgium": "🇧🇪", "Brazil": "🇧🇷", "Portugal": "🇵🇹",
        "Netherlands": "🇳🇱", "Spain": "🇪🇸", "Croatia": "🇭🇷",
        "Italy": "🇮🇹", "Morocco": "🇲🇦", "USA": "🇺🇸",
        "Mexico": "🇲🇽", "Germany": "🇩🇪", "Colombia": "🇨🇴",
        "Uruguay": "🇺🇾", "Denmark": "🇩🇰", "Switzerland": "🇨🇭",
        "Japan": "🇯🇵", "South Korea": "🇰🇷", "Australia": "🇦🇺",
        "Canada": "🇨🇦", "Senegal": "🇸🇳", "Ecuador": "🇪🇨",
        "Serbia": "🇷🇸", "Poland": "🇵🇱", "Iran": "🇮🇷",
        "Ghana": "🇬🇭", "Cameroon": "🇨🇲", "Egypt": "🇪🇬",
        "Wales": "🏴󠁧󠁢󠁷󠁬󠁳󠁿", "Saudi Arabia": "🇸🇦", "Qatar": "🇶🇦",
        "Russia": "🇷🇺", "Sweden": "🇸🇪", "Chile": "🇨🇱",
        "Peru": "🇵🇪", "Paraguay": "🇵🇾", "Costa Rica": "🇨🇷",
        "Algeria": "🇩🇿", "Nigeria": "🇳🇬", "Ivory Coast": "🇨🇮",
        "Tunisia": "🇹🇳", "South Africa": "🇿🇦", "Ukraine": "🇺🇦",
        "Turkey": "🇹🇷", "Greece": "🇬🇷", "Slovakia": "🇸🇰",
        "Hungary": "🇭🇺", "Romania": "🇷🇴", "Austria": "🇦🇹",
        "Iceland": "🇮🇸", "New Zealand": "🇳🇿", "North Korea": "🇰🇵",
        "TBD": "🏳️", "Draw": "⚖️",
    }
    return FLAGS.get(team, "🏳️")


ROUND_ORDER = ["Group Stage", "Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "3rd Place", "Final"]

ROUND_COLORS = {
    "Group Stage": "#4ade80",
    "Round of 32": "#60a5fa",
    "Round of 16": "#f59e0b",
    "Quarterfinal": "#f97316",
    "Semifinal": "#ec4899",
    "3rd Place": "#a78bfa",
    "Final": "#facc15",
}

# ─────────────────────────────────────────────────────────────────────────────
# 21st.dev-inspired Design System CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
  --bg-base: #030712;
  --bg-surface: #0d1117;
  --bg-elevated: #111827;
  --bg-card: rgba(17, 24, 39, 0.8);
  --border-subtle: rgba(255,255,255,0.06);
  --border-accent: rgba(0, 212, 255, 0.25);
  --border-glow: rgba(0, 212, 255, 0.5);
  --text-primary: #f1f5f9;
  --text-secondary: #94a3b8;
  --text-muted: #475569;
  --accent-cyan: #00D4FF;
  --accent-purple: #7B2FFF;
  --accent-pink: #FF3CAC;
  --accent-green: #10b981;
  --accent-amber: #f59e0b;
  --accent-red: #ef4444;
  --font-main: 'Space Grotesk', 'Inter', sans-serif;
}

* { font-family: var(--font-main); box-sizing: border-box; }

/* Streamlit overrides */
.stApp { background-color: var(--bg-base) !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
section[data-testid="stSidebar"] { background: #080c14 !important; border-right: 1px solid var(--border-subtle) !important; }
section[data-testid="stSidebar"] * { color: var(--text-secondary) !important; }
.stButton > button { background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); border: none; color: white; font-weight: 600; border-radius: 10px; transition: all 0.2s; }
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(0,212,255,0.3); }

/* ── Glass Card — 21st.dev glassmorphism panel ── */
.glass-card {
  background: var(--bg-card);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border-subtle);
  border-radius: 16px;
  padding: 24px;
  margin: 10px 0;
  transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
  position: relative;
  overflow: hidden;
}
.glass-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent-cyan), transparent);
  opacity: 0.4;
}
.glass-card:hover {
  border-color: var(--border-accent);
  box-shadow: 0 0 0 1px rgba(0,212,255,0.1), 0 20px 40px rgba(0,0,0,0.4);
  transform: translateY(-2px);
}

/* ── KPI Stat Card ── */
.kpi-card {
  background: linear-gradient(135deg, rgba(17,24,39,0.9) 0%, rgba(7,9,21,0.9) 100%);
  border: 1px solid var(--border-subtle);
  border-radius: 14px;
  padding: 20px;
  text-align: center;
  position: relative;
  overflow: hidden;
  transition: all 0.3s;
}
.kpi-card::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
  opacity: 0.6;
}
.kpi-card:hover { transform: translateY(-3px); border-color: var(--border-accent); box-shadow: 0 0 30px rgba(0,212,255,0.12); }
.kpi-value { font-size: 2.4rem; font-weight: 800; background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1; }
.kpi-label { font-size: 0.72rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; margin-top: 8px; }
.kpi-delta { font-size: 0.8rem; margin-top: 6px; }
.kpi-delta.positive { color: var(--accent-green); }
.kpi-delta.negative { color: var(--accent-red); }

/* ── Hero Section ── */
.hero-title {
  font-size: 3.8rem; font-weight: 900; line-height: 1.05; margin-bottom: 1rem;
  background: linear-gradient(135deg, #ffffff 0%, var(--accent-cyan) 40%, var(--accent-purple) 80%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-subtitle { font-size: 1.1rem; color: var(--text-secondary); font-weight: 400; max-width: 600px; line-height: 1.6; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(0,212,255,0.08); border: 1px solid rgba(0,212,255,0.2);
  color: var(--accent-cyan); border-radius: 20px; padding: 4px 14px;
  font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px;
  margin-bottom: 16px;
}

/* ── Match Card — 21st.dev match panel ── */
.match-card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 16px; padding: 20px;
  margin: 8px 0;
  transition: all 0.3s;
  position: relative; overflow: hidden;
}
.match-card:hover { border-color: var(--border-accent); box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 0 1px rgba(0,212,255,0.08); }
.match-card.featured { border-color: rgba(0,212,255,0.3); box-shadow: 0 0 40px rgba(0,212,255,0.08); }

/* ── Team VS Block ── */
.team-vs-block {
  display: flex; align-items: center; justify-content: space-between;
  padding: 12px 0;
}
.team-block { text-align: center; flex: 1; }
.team-flag { font-size: 2.5rem; display: block; margin-bottom: 6px; }
.team-name { font-size: 0.9rem; font-weight: 700; color: var(--text-primary); }
.team-rank { font-size: 0.7rem; color: var(--text-muted); margin-top: 2px; }
.vs-separator { color: var(--text-muted); font-size: 0.9rem; font-weight: 700; padding: 0 12px; flex-shrink: 0; }

/* ── Probability Meter — 21st.dev progress bar ── */
.prob-meter-container { margin: 6px 0; }
.prob-meter-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
.prob-meter-label { font-size: 0.75rem; color: var(--text-secondary); font-weight: 500; }
.prob-meter-value { font-size: 0.8rem; font-weight: 700; }
.prob-meter-track {
  background: rgba(255,255,255,0.05); border-radius: 100px; height: 8px;
  overflow: hidden; position: relative;
}
.prob-meter-fill {
  height: 100%; border-radius: 100px;
  position: relative; transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.prob-meter-fill.home { background: linear-gradient(90deg, #0ea5e9, #00D4FF); }
.prob-meter-fill.draw { background: linear-gradient(90deg, #64748b, #94a3b8); }
.prob-meter-fill.away { background: linear-gradient(90deg, #be185d, #FF3CAC); }

/* ── Round Badge ── */
.round-badge {
  display: inline-block; padding: 3px 12px; border-radius: 20px;
  font-size: 0.68rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
}

/* ── Confidence Badge ── */
.conf-badge {
  display: inline-flex; align-items: center; gap: 4px;
  padding: 4px 10px; border-radius: 8px;
  font-size: 0.72rem; font-weight: 600;
}
.conf-high { background: rgba(16,185,129,0.12); color: #10b981; border: 1px solid rgba(16,185,129,0.25); }
.conf-medium { background: rgba(245,158,11,0.12); color: #f59e0b; border: 1px solid rgba(245,158,11,0.25); }
.conf-low { background: rgba(239,68,68,0.12); color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }

/* ── Evidence Card — proof chip ── */
.evidence-card {
  background: rgba(0,212,255,0.04);
  border: 1px solid rgba(0,212,255,0.12);
  border-left: 3px solid var(--accent-cyan);
  border-radius: 0 10px 10px 0;
  padding: 12px 16px; margin: 6px 0;
  transition: all 0.2s;
}
.evidence-card:hover { background: rgba(0,212,255,0.07); }
.evidence-label { font-size: 0.65rem; font-weight: 700; color: var(--accent-cyan); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.evidence-text { font-size: 0.82rem; color: var(--text-primary); line-height: 1.5; }
.evidence-sub { font-size: 0.72rem; color: var(--text-muted); margin-top: 3px; }

/* ── Risk Card ── */
.risk-card {
  background: rgba(239,68,68,0.04);
  border: 1px solid rgba(239,68,68,0.12);
  border-left: 3px solid var(--accent-red);
  border-radius: 0 10px 10px 0;
  padding: 12px 16px; margin: 6px 0;
}
.risk-label { font-size: 0.65rem; font-weight: 700; color: var(--accent-red); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }

/* ── Factor Chip ── */
.factor-chip {
  display: inline-flex; align-items: center; gap: 5px;
  padding: 4px 12px; border-radius: 20px; margin: 3px;
  font-size: 0.72rem; font-weight: 600;
}
.chip-positive { background: rgba(16,185,129,0.1); color: #10b981; border: 1px solid rgba(16,185,129,0.2); }
.chip-negative { background: rgba(239,68,68,0.1); color: #ef4444; border: 1px solid rgba(239,68,68,0.2); }
.chip-neutral { background: rgba(148,163,184,0.1); color: #94a3b8; border: 1px solid rgba(148,163,184,0.2); }

/* ── Section Header ── */
.section-header {
  font-size: 1.4rem; font-weight: 700; color: var(--text-primary);
  margin: 28px 0 16px; padding-bottom: 10px;
  border-bottom: 1px solid var(--border-subtle);
  display: flex; align-items: center; gap: 10px;
}

/* ── Divider ── */
.neon-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent-cyan), var(--accent-purple), transparent);
  opacity: 0.3; margin: 24px 0; border: none;
}

/* ── Low Confidence Banner ── */
.low-conf-banner {
  background: rgba(245,158,11,0.08);
  border: 1px solid rgba(245,158,11,0.25);
  border-radius: 10px; padding: 10px 16px;
  color: #f59e0b; font-size: 0.82rem; margin: 8px 0;
  display: flex; align-items: center; gap: 8px;
}

/* ── Predicted Winner Tag ── */
.winner-tag {
  font-size: 1.3rem; font-weight: 800; color: var(--accent-cyan);
  display: flex; align-items: center; gap: 8px; justify-content: center;
  margin: 8px 0;
}

/* ── Sidebar ── */
.sidebar-logo { font-size: 1.6rem; font-weight: 900; color: var(--accent-cyan) !important; margin-bottom: 4px; }
.sidebar-tagline { font-size: 0.72rem; color: var(--text-muted) !important; letter-spacing: 0.5px; }
.sidebar-nav-item {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 12px; border-radius: 8px; margin: 2px 0;
  font-size: 0.82rem; color: var(--text-secondary) !important;
  text-decoration: none; transition: all 0.2s;
}
.sidebar-nav-item:hover { background: rgba(0,212,255,0.08) !important; color: var(--accent-cyan) !important; }

/* ── Bracket node ── */
.bracket-node {
  background: rgba(13,17,23,0.9);
  border: 1px solid var(--border-subtle);
  border-radius: 8px; padding: 8px 10px; margin: 4px 0;
  font-size: 0.72rem; transition: all 0.2s;
}
.bracket-node:hover { border-color: var(--border-accent); }
.bracket-winner-name { color: var(--accent-cyan); font-weight: 700; }
.bracket-loser-name { color: var(--text-muted); }
.bracket-prob-tag { font-size: 0.62rem; color: var(--text-muted); }
</style>
"""
