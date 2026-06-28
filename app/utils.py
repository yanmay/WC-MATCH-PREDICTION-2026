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


# ── Date formatting helper ───────────────────────────────────────────────────
def format_match_date(date_str: str) -> str:
    """
    Convert a YYYY-MM-DD string to a human-friendly label:
    'Today', 'Yesterday', '2 days ago', or the original string.
    """
    import datetime
    try:
        match_date = datetime.date.fromisoformat(str(date_str)[:10])
        today = datetime.date.today()
        delta = (today - match_date).days
        if delta == 0:
            return "Today"
        elif delta == 1:
            return "Yesterday"
        elif delta == 2:
            return "2 days ago"
        elif 3 <= delta <= 6:
            return f"{delta} days ago"
        elif delta < 0:
            # Future match
            if delta == -1:
                return "Tomorrow"
            return match_date.strftime("%b %d")
        else:
            return match_date.strftime("%b %d")
    except Exception:
        return str(date_str)


@st.cache_data(show_spinner=False, ttl=120)  # Refresh every 2 minutes to pick up live results
def load_data():
    from ml.data_loader import load_historical_data, get_world_cup_data, get_2026_fixtures
    raw = load_historical_data()
    wc_df = get_world_cup_data(raw)
    fixtures = get_2026_fixtures(wc_df)

    # Sync live results from worldcup26.ir API (no key required)
    try:
        from ml.live_scores import sync_live_results_to_fixtures
        fixtures, _ = sync_live_results_to_fixtures(fixtures, wc_df)
    except Exception:
        pass

    # Sync completed results into prediction log
    try:
        from ml.prediction_log import sync_results_from_fixtures
        sync_results_from_fixtures(fixtures)
    except Exception:
        pass

    return wc_df, fixtures


def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">⚽ WC 2026</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-tagline">AI MATCH PREDICTION ENGINE</div>', unsafe_allow_html=True)
        st.divider()
        
        st.markdown("**Navigation**")
        st.page_link("Home.py", label="Home", icon="🏠")
        st.page_link("pages/1_Upcoming_Matches.py", label="Upcoming Matches", icon="📅")
        st.page_link("pages/2_Match_Detail.py", label="Match Detail", icon="🔍")
        st.page_link("pages/3_Accuracy_Tracker.py", label="Accuracy Tracker", icon="📊")
        st.page_link("pages/4_Tournament_Bracket.py", label="Tournament Bracket", icon="🏆")
        st.divider()

        # ── Live 2026 Prediction Accuracy Counter ─────────────────────────
        try:
            from ml.prediction_log import get_live_accuracy_stats
            stats = get_live_accuracy_stats()
            if stats["resolved"] > 0:
                acc_2026 = stats["accuracy"]
                correct = stats["correct"]
                total = stats["resolved"]
                pct = acc_2026 * 100
                bar_color = "#34d399" if pct >= 60 else ("#fbbf24" if pct >= 45 else "#f87171")
                st.markdown(f"""
<div class="accuracy-counter">
  <div style="font-size:0.65rem;color:#9ca3af;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;">🎯 2026 Live AI Accuracy</div>
  <div style="font-size:1.6rem;font-weight:900;color:{bar_color};line-height:1;">{pct:.0f}%</div>
  <div style="font-size:0.7rem;color:#6b7280;margin-top:3px;">{correct}/{total} predictions correct</div>
  <div style="background:rgba(255,255,255,0.06);border-radius:100px;height:5px;margin-top:8px;overflow:hidden;">
    <div style="background:{bar_color};height:100%;width:{pct:.0f}%;border-radius:100px;transition:width 0.6s;"></div>
  </div>
</div>""", unsafe_allow_html=True)
                st.markdown("")
        except Exception:
            pass

        try:
            wc_df, _ = load_data()
            _, metrics = get_model_and_metrics()
            acc = metrics.get("accuracy", 0.7182)
            cv_acc = metrics.get("cv_mean_accuracy", 0.7056)
            algo = metrics.get("selected_algorithm", metrics.get("algorithm", "logistic_regression")).replace("_", " ").title()
            
            st.caption(f"📊 Training data: {len(wc_df):,} WC matches")
            st.caption(f"🤖 Model: {algo} + CalibratedCV")
            st.caption(f"🎯 Test Accuracy: {acc:.2%}")
            st.caption(f"📈 Cross-val accuracy: {cv_acc:.2%}")
        except Exception:
            st.caption("📊 Training data: 9,856 WC matches")
            st.caption("🤖 Model: Logistic Regression + CalibratedCV")
            st.caption("🎯 Test Accuracy: 71.82%")
            st.caption("📈 Cross-val accuracy: 70.56%")


def make_live_timer_html(elapsed_mins: int, elapsed_secs: int, timer_id: str = "live-timer-1") -> str:
    """
    Returns compact HTML for a live-ticking match minute counter.
    
    Uses a hidden Streamlit components.html iframe to run background JavaScript.
    The JavaScript queries the parent page DOM to find the timer element and increments
    it every second in the browser without reloading the page.
    """
    # Build initial display string
    if elapsed_mins >= 90:
        extra = elapsed_mins - 90
        init_display = f"90+{extra}&#39; {str(elapsed_secs).zfill(2)}"
    else:
        init_display = f"{elapsed_mins}&#39; {str(elapsed_secs).zfill(2)}"

    import streamlit.components.v1 as components
    components.html(
        f"""
        <script>
        (function() {{
            var parentDoc = window.parent.document;
            function tick() {{
                var el = parentDoc.getElementById('{timer_id}');
                if (!el) return;
                if (el.getAttribute('data-ticking') === 'true') return;
                el.setAttribute('data-ticking', 'true');
                var m = parseInt(el.getAttribute('data-m')) || 0;
                var s = parseInt(el.getAttribute('data-s')) || 0;
                setInterval(function() {{
                    s++;
                    if (s >= 60) {{
                        s = 0;
                        m++;
                    }}
                    var d;
                    if (m >= 90) {{
                        d = '90+' + (m - 90) + "' " + String(s).padStart(2, '0');
                    }} else {{
                        d = m + "' " + String(s).padStart(2, '0');
                    }}
                    el.textContent = d;
                    el.setAttribute('data-m', m);
                    el.setAttribute('data-s', s);
                }}, 1000);
            }}
            tick();
            // Retry locating the element in case it takes a split second to mount
            var attempts = 0;
            var interval = setInterval(function() {{
                var el = parentDoc.getElementById('{timer_id}');
                if (el) {{
                    tick();
                    clearInterval(interval);
                }}
                attempts++;
                if (attempts > 30) {{
                    clearInterval(interval);
                }}
            }}, 100);
        }})();
        </script>
        """,
        height=0,
        width=0
    )

    return (
        f'<span id="{timer_id}" data-m="{elapsed_mins}" data-s="{elapsed_secs}" '
        f'style="font-weight:700;letter-spacing:1px;">{init_display}</span>'
    )




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


FEATURE_LABELS = {
    "win_rate_diff": "Win Rate Diff",
    "home_win_rate": "Home Win Rate",
    "away_win_rate": "Away Win Rate",
    "home_ranking": "Home FIFA Rank",
    "away_ranking": "Away FIFA Rank",
    "ranking_diff": "FIFA Rank Diff",
    "ranking_diff_abs": "FIFA Rank Diff (Abs)",
    "rank_ratio": "FIFA Rank Ratio",
    "goals_pg_diff": "Goals Scored Diff",
    "conceded_pg_diff": "Goals Conceded Diff",
    "home_goals_pg": "Home Goals/Match",
    "away_goals_pg": "Away Goals/Match",
    "home_conceded_pg": "Home Conceded/Match",
    "away_conceded_pg": "Away Conceded/Match",
    "h2h_home_wins": "H2H Home Wins",
    "h2h_away_wins": "H2H Away Wins",
    "h2h_draws": "H2H Draws",
    "h2h_has_history": "Has H2H History",
    "is_knockout": "Is Knockout Stage",
    "host_nation": "Host Advantage",
    "same_confederation": "Same Confederation",
    "home_rest_days": "Home Rest Days",
    "away_rest_days": "Away Rest Days",
}


def map_feature_name(name: str) -> str:
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("confederation_matchup_"):
        parts = name.replace("confederation_matchup_", "").split("_vs_")
        if len(parts) == 2:
            conf_map = {
                "AFC": "Asia",
                "UEFA": "Europe",
                "CONMEBOL": "S. America",
                "CONCACAF": "N. America",
                "CAF": "Africa",
                "OFC": "Oceania",
                "OTHER": "Other"
            }
            c1 = conf_map.get(parts[0], parts[0])
            c2 = conf_map.get(parts[1], parts[1])
            return f"Region: {c1} vs {c2}"
    return name.replace("_", " ").title()


def infer_wc_round(match_idx: int, total_matches: int) -> str:
    if total_matches == 64:
        if match_idx < 48:
            return "Group Stage"
        elif match_idx < 56:
            return "Round of 16"
        elif match_idx < 60:
            return "Quarterfinal"
        elif match_idx < 62:
            return "Semifinal"
        elif match_idx == 62:
            return "3rd Place"
        else:
            return "Final"
    return "Group Stage"


def get_team_flag(team: str, as_emoji: bool = False) -> str:
    if not team or not isinstance(team, str):
        return "🏳️"
        
    # Clean up duplicate country name concatenation like "FranceFrance" or "IraqIraq"
    team_strip = team.strip()
    half_len = len(team_strip) // 2
    if half_len > 1 and team_strip[:half_len].lower() == team_strip[half_len:].lower():
        team_clean = team_strip[:half_len]
    else:
        team_clean = team_strip

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
        "Norway": "🇳🇴", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Czechia": "🇨🇿",
        "Bosnia and Herzegovina": "🇧🇦", "Iraq": "🇮🇶",
        "Jordan": "🇯🇴", "Uzbekistan": "🇺🇿", "Haiti": "🇭🇹",
        "DR Congo": "🇨🇩", "Curacao": "🇨🇼", "Cabo Verde": "🇨🇻",
        "Panama": "🇵🇦", "South Korea": "🇰🇷",
        "TBD": "🏳️", "Draw": "⚖️",
    }
    
    TEAM_CODES = {
        "Argentina": "ar", "France": "fr", "England": "gb-eng",
        "Belgium": "be", "Brazil": "br", "Portugal": "pt",
        "Netherlands": "nl", "Spain": "es", "Croatia": "hr",
        "Italy": "it", "Morocco": "ma", "USA": "us",
        "Mexico": "mx", "Germany": "de", "Colombia": "co",
        "Uruguay": "uy", "Denmark": "dk", "Switzerland": "ch",
        "Japan": "jp", "South Korea": "kr", "Australia": "au",
        "Canada": "ca", "Senegal": "sn", "Ecuador": "ec",
        "Serbia": "rs", "Poland": "pl", "Iran": "ir",
        "Ghana": "gh", "Cameroon": "cm", "Egypt": "eg",
        "Wales": "gb-wls", "Saudi Arabia": "sa", "Qatar": "qa",
        "Russia": "ru", "Sweden": "se", "Chile": "cl",
        "Peru": "pe", "Paraguay": "py", "Costa Rica": "cr",
        "Algeria": "dz", "Nigeria": "ng", "Ivory Coast": "ci",
        "Tunisia": "tn", "South Africa": "za", "Ukraine": "ua",
        "Turkey": "tr", "Greece": "gr", "Slovakia": "sk",
        "Hungary": "hu", "Romania": "ro", "Austria": "at",
        "Iceland": "is", "New Zealand": "nz", "North Korea": "kp",
        "Norway": "no", "Scotland": "gb-sct", "Czechia": "cz",
        "Bosnia and Herzegovina": "ba", "Iraq": "iq",
        "Jordan": "jo", "Uzbekistan": "uz", "Haiti": "ht",
        "DR Congo": "cd", "Curacao": "cw", "Cabo Verde": "cv",
        "Panama": "pa",
    }

    # Case-insensitive key matching
    matched_key = None
    for k in FLAGS:
        if k.lower() == team_clean.lower():
            matched_key = k
            break

    if not matched_key:
        return "🏳️"

    # Rule: If a country name appears in lowercase, replace it with the appropriate flag
    # If the original input string was completely lowercase (and has letters), return just the flag.
    is_all_lowercase = team.islower() and any(c.isalpha() for c in team)

    if as_emoji or matched_key in ["TBD", "Draw", "⚖️", "🏳️"] or matched_key not in TEAM_CODES or is_all_lowercase:
        return FLAGS.get(matched_key, "🏳️")

    code = TEAM_CODES[matched_key]
    return f'<img src="https://flagcdn.com/w40/{code}.png" class="flag-icon" />'


def format_team_html(team: str) -> str:
    if not team or not isinstance(team, str):
        return "🏳️"
    
    # Normalize name duplicates like "FranceFrance"
    team_strip = team.strip()
    half_len = len(team_strip) // 2
    if half_len > 1 and team_strip[:half_len].lower() == team_strip[half_len:].lower():
        team_clean = team_strip[:half_len]
    else:
        team_clean = team_strip

    # Lowercase rule
    if team.islower() and any(c.isalpha() for c in team):
        return get_team_flag(team_clean, as_emoji=False)
        
    return f"{get_team_flag(team_clean, as_emoji=False)} <span class='team-name'>{team_clean}</span>"


def format_team_emoji(team: str) -> str:
    if not team or not isinstance(team, str):
        return "🏳️"
        
    # Normalize name duplicates like "FranceFrance"
    team_strip = team.strip()
    half_len = len(team_strip) // 2
    if half_len > 1 and team_strip[:half_len].lower() == team_strip[half_len:].lower():
        team_clean = team_strip[:half_len]
    else:
        team_clean = team_strip

    # Lowercase rule
    if team.islower() and any(c.isalpha() for c in team):
        return get_team_flag(team_clean, as_emoji=True)
        
    return f"{get_team_flag(team_clean, as_emoji=True)} {team_clean}"



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
  --bg-base: #0b0f19;
  --bg-surface: #111827;
  --bg-elevated: #1f2937;
  --bg-card: rgba(17, 24, 39, 0.75);
  --border-subtle: rgba(255, 255, 255, 0.08);
  --border-accent: rgba(52, 211, 153, 0.3);
  --border-glow: rgba(52, 211, 153, 0.5);
  --text-primary: #f3f4f6;
  --text-secondary: #9ca3af;
  --text-muted: #6b7280;
  --accent-cyan: #34d399;
  --accent-purple: #065f46;
  --accent-pink: #059669;
  --accent-green: #34d399;
  --accent-amber: #fbbf24;
  --accent-red: #f87171;
  --font-main: 'Space Grotesk', 'Inter', sans-serif;
}

* { font-family: var(--font-main); box-sizing: border-box; }

/* Streamlit overrides */
.stApp { background-color: var(--bg-base) !important; color: var(--text-primary) !important; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
section[data-testid="stSidebar"] { background: #0b0f19 !important; border-right: 1px solid var(--border-subtle) !important; }
section[data-testid="stSidebar"] * { color: #e5e7eb !important; }
.stButton > button { background: linear-gradient(135deg, var(--accent-green), var(--accent-purple)); border: none; color: white; font-weight: 600; border-radius: 10px; transition: all 0.2s; }

/* ── Glass Card — 21st.dev glassmorphism panel ── */
.glass-card {
  background: var(--bg-card);
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
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent-green), transparent);
  opacity: 0.8;
}

/* ── KPI Stat Card ── */
.kpi-card {
  background: linear-gradient(135deg, rgba(17,24,39,0.95) 0%, rgba(6,78,59,0.2) 100%);
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
  background: linear-gradient(90deg, var(--accent-green), var(--accent-purple));
  opacity: 0.6;
}
.kpi-value { font-size: 2.4rem; font-weight: 800; background: linear-gradient(135deg, var(--accent-green), var(--accent-purple)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1; }
.kpi-label { font-size: 0.72rem; font-weight: 600; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; margin-top: 8px; }
.kpi-delta { font-size: 0.8rem; margin-top: 6px; }
.kpi-delta.positive { color: var(--accent-green); }
.kpi-delta.negative { color: var(--accent-red); }

/* ── Hero Section ── */
.hero-title {
  font-size: 3.8rem; font-weight: 900; line-height: 1.05; margin-bottom: 1rem;
  background: linear-gradient(135deg, #f3f4f6 0%, var(--accent-green) 50%, var(--accent-purple) 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-subtitle { font-size: 1.1rem; color: var(--text-secondary); font-weight: 400; max-width: 600px; line-height: 1.6; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 6px;
  background: rgba(52,211,153,0.08); border: 1px solid rgba(52,211,153,0.2);
  color: var(--accent-green); border-radius: 20px; padding: 4px 14px;
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
.match-card.featured { border-color: rgba(52,211,153,0.3); box-shadow: 0 0 40px rgba(52,211,153,0.08); }

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
.prob-meter-fill.home { background: linear-gradient(90deg, #34d399, #059669); }
.prob-meter-fill.draw { background: linear-gradient(90deg, #64748b, #94a3b8); }
.prob-meter-fill.away { background: linear-gradient(90deg, #f87171, #dc2626); }

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
.conf-high { background: rgba(52,211,153,0.12); color: #34d399; border: 1px solid rgba(52,211,153,0.25); }
.conf-medium { background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); }
.conf-low { background: rgba(239,68,68,0.12); color: #f87171; border: 1px solid rgba(239,68,68,0.25); }

/* ── Evidence Card — proof chip ── */
.evidence-card {
  background: rgba(52,211,153,0.04);
  border: 1px solid rgba(52,211,153,0.12);
  border-left: 3px solid var(--accent-green);
  border-radius: 0 10px 10px 0;
  padding: 12px 16px; margin: 6px 0;
  transition: all 0.2s;
}
.evidence-label { font-size: 0.65rem; font-weight: 700; color: var(--accent-green); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px; }
.evidence-text { font-size: 0.82rem; color: var(--text-primary); line-height: 1.5; }
.evidence-sub { font-size: 0.72rem; color: var(--text-muted); margin-top: 3px; }

/* ── Risk Card ── */
.risk-card {
  background: rgba(239,113,113,0.04);
  border: 1px solid rgba(239,113,113,0.12);
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
.chip-positive { background: rgba(52,211,153,0.1); color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
.chip-negative { background: rgba(239,113,113,0.1); color: #f87171; border: 1px solid rgba(239,113,113,0.2); }
.chip-neutral { background: rgba(148,163,184,0.1); color: #9ca3af; border: 1px solid rgba(148,163,184,0.2); }

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
  background: linear-gradient(90deg, transparent, var(--accent-green), var(--accent-purple), transparent);
  opacity: 0.3; margin: 24px 0; border: none;
}

/* ── Low Confidence Banner ── */
.low-conf-banner {
  background: rgba(251,191,36,0.08);
  border: 1px solid rgba(251,191,36,0.25);
  border-radius: 10px; padding: 10px 16px;
  color: #fbbf24; font-size: 0.82rem; margin: 8px 0;
  display: flex; align-items: center; gap: 8px;
}

/* ── Predicted Winner Tag ── */
.winner-tag {
  font-size: 1.3rem; font-weight: 800; color: var(--accent-green);
  display: flex; align-items: center; gap: 8px; justify-content: center;
  margin: 8px 0;
}

/* ── Sidebar ── */
.sidebar-logo { font-size: 1.6rem; font-weight: 900; color: var(--accent-green) !important; margin-bottom: 4px; }
.sidebar-tagline { font-size: 0.72rem; color: var(--text-muted) !important; letter-spacing: 0.5px; }
.sidebar-nav-item {
  display: flex; align-items: center; gap: 8px;
  padding: 8px 12px; border-radius: 8px; margin: 2px 0;
  font-size: 0.82rem; color: #f3f4f6 !important;
  text-decoration: none; transition: all 0.2s;
}

/* ── Bracket node ── */
.bracket-node {
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: 8px; padding: 8px 10px; margin: 4px 0;
  font-size: 0.72rem; transition: all 0.2s;
}
.bracket-winner-name { color: var(--accent-green) !important; font-weight: 700; }
.bracket-loser-name { color: var(--text-muted) !important; }
.bracket-prob-tag { font-size: 0.62rem; color: var(--text-muted); }

/* ── Result outcome labels (Win/Loss/Draw) ── */
.result-win {
  color: #34d399 !important;
  background: rgba(52,211,153,0.1);
  border: 1px solid rgba(52,211,153,0.25);
  border-radius: 6px;
  padding: 2px 10px;
  font-weight: 700;
  font-size: 0.78rem;
  display: inline-block;
}
.result-loss {
  color: #f87171 !important;
  background: rgba(239,113,113,0.1);
  border: 1px solid rgba(239,113,113,0.25);
  border-radius: 6px;
  padding: 2px 10px;
  font-weight: 700;
  font-size: 0.78rem;
  display: inline-block;
}
.result-draw {
  color: #fbbf24 !important;
  background: rgba(251,191,36,0.1);
  border: 1px solid rgba(251,191,36,0.25);
  border-radius: 6px;
  padding: 2px 10px;
  font-weight: 700;
  font-size: 0.78rem;
  display: inline-block;
}

/* ── Team name display ── */
.team-name, .bracket-team span, .match-team-name {
  color: var(--text-primary) !important;
  font-weight: 600;
}
.winner-team-name {
  color: #34d399 !important;
  font-weight: 800;
}
.loser-team-name {
  color: var(--text-muted) !important;
  font-weight: 400;
}

/* ── Score display ── */
.score-badge {
  background: #1f2937;
  border: 1px solid var(--border-subtle);
  border-radius: 8px;
  padding: 4px 14px;
  font-size: 1.1rem;
  font-weight: 800;
  color: var(--text-primary) !important;
  letter-spacing: 2px;
}
.score-badge.completed {
  background: rgba(52,211,153,0.08);
  border-color: rgba(52,211,153,0.2);
  color: #34d399 !important;
}

/* ── Live indicator ── */
.live-dot {
  display: inline-block;
  width: 8px; height: 8px;
  background: #f87171;
  border-radius: 50%;
  margin-right: 5px;
  animation: pulse-live 1.4s ease infinite;
  vertical-align: middle;
}
@keyframes pulse-live {
  0%, 100% { opacity: 1; transform: scale(1); }
  50% { opacity: 0.4; transform: scale(0.8); }
}
.live-badge {
  display: inline-flex; align-items: center; gap: 4px;
  background: rgba(239,113,113,0.08); border: 1px solid rgba(239,113,113,0.25);
  color: #f87171; border-radius: 12px; padding: 2px 10px;
  font-size: 0.65rem; font-weight: 700; letter-spacing: 0.5px;
}

/* ── Dataframe overrides for dark theme ── */
.stDataFrame td, .stDataFrame th { color: var(--text-primary) !important; }
.stDataFrame [data-testid="stTable"] { color: var(--text-primary) !important; }

/* ── Flag icon ── */
.flag-icon {
  vertical-align: middle;
  border-radius: 2px;
  height: 1.1em;
  display: inline-block;
  margin-right: 4px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.3);
}

/* ── Admin score entry form ── */
.admin-form {
  background: var(--bg-surface);
  border: 1px solid var(--border-accent);
  border-radius: 12px;
  padding: 16px;
  margin-top: 8px;
}

/* ── Prediction outcome badges ── */
.pred-badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 10px;
  border-radius: 8px;
  font-size: 0.72rem;
  font-weight: 700;
  letter-spacing: 0.3px;
}
.pred-correct {
  background: rgba(52, 211, 153, 0.12);
  color: #34d399;
  border: 1px solid rgba(52, 211, 153, 0.3);
}
.pred-wrong {
  background: rgba(239, 113, 113, 0.12);
  color: #f87171;
  border: 1px solid rgba(239, 113, 113, 0.3);
}
.pred-pending {
  background: rgba(251, 191, 36, 0.1);
  color: #fbbf24;
  border: 1px solid rgba(251, 191, 36, 0.25);
}

/* ── Live accuracy counter (sidebar) ── */
.accuracy-counter {
  background: rgba(52, 211, 153, 0.04);
  border: 1px solid rgba(52, 211, 153, 0.15);
  border-radius: 12px;
  padding: 14px 16px;
  margin-bottom: 12px;
}

/* ── Match status pill ── */
.match-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 2px 10px;
  border-radius: 20px;
  font-size: 0.65rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.8px;
}
.status-completed { background: rgba(52,211,153,0.1); color: #34d399; border: 1px solid rgba(52,211,153,0.2); }
.status-live      { background: rgba(239,113,113,0.1); color: #f87171; border: 1px solid rgba(239,113,113,0.2); }
.status-scheduled { background: rgba(148,163,184,0.1); color: #94a3b8; border: 1px solid rgba(148,163,184,0.2); }

/* ── Date label chip ── */
.date-chip {
  font-size: 0.65rem;
  font-weight: 700;
  color: #6b7280;
  letter-spacing: 0.5px;
}
.date-chip.today    { color: #34d399; }
.date-chip.yesterday { color: #fbbf24; }

/* ── Accuracy trend badge on KPI ── */
.kpi-accuracy-2026 {
  background: linear-gradient(135deg, rgba(17,24,39,0.95) 0%, rgba(52,211,153,0.08) 100%);
  border: 1px solid rgba(52,211,153,0.3);
  box-shadow: 0 0 20px rgba(52,211,153,0.06);
}

/* ── Hover animations — active ONLY on hover-supporting desktop devices ── */
@media (hover: hover) {
  .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(52,211,153,0.3); }
  .glass-card:hover { border-color: var(--border-accent); box-shadow: 0 0 0 1px rgba(52,211,153,0.1), 0 20px 40px rgba(0,0,0,0.3); transform: translateY(-2px); }
  .kpi-card:hover { transform: translateY(-3px); border-color: var(--border-accent); box-shadow: 0 0 30px rgba(52,211,153,0.08); }
  .match-card:hover { border-color: var(--border-accent); box-shadow: 0 8px 32px rgba(0,0,0,0.3), 0 0 0 1px rgba(52,211,153,0.08); }
  .evidence-card:hover { background: rgba(52,211,153,0.07); }
  .sidebar-nav-item:hover { background: rgba(52,211,153,0.08) !important; color: var(--accent-green) !important; }
  .bracket-node:hover { border-color: var(--border-accent); }
}

/* ── KPI responsive CSS grid ── */
.kpi-grid-container {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
  gap: 12px;
  width: 100%;
  margin-bottom: 20px;
}

/* ── Flag image styling in team vs blocks ── */
.team-flag-img {
  height: 48px;
  border-radius: 4px;
}

/* ── Touch UI & Mobile Responsive Layouts (iOS / Android) ── */
@media (max-width: 768px) {
  .hero-title { font-size: 2.2rem !important; }
  .hero-subtitle { font-size: 0.95rem !important; }
  .glass-card, .match-card, .kpi-card { padding: 16px !important; margin: 6px 0 !important; }
  .stButton > button { min-height: 48px !important; font-size: 1.0rem !important; }
  .sidebar-nav-item { min-height: 44px !important; display: flex; align-items: center; padding: 10px 14px !important; }
  .vs-separator { font-size: 0.8rem !important; }
  .winner-tag { font-size: 1.1rem !important; }
  table th, table td { padding: 10px 8px !important; } /* comfortable touch spacing for lineups */
  .team-flag-img { height: 38px !important; }
  .team-name { font-size: 0.8rem !important; }
}
</style>

"""
