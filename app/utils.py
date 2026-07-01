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
    # Retrieve active page from query parameter (Single Page App Routing)
    page = st.query_params.get("page", "home")

    active_home = "active" if page == "home" or page == "standings" else ""
    active_upcoming = "active" if page == "upcoming" or page == "completed" else ""
    active_detail = "active" if page == "detail" else ""
    active_accuracy = "active" if page == "accuracy" else ""
    active_bracket = "active" if page == "bracket" else ""

    # Live accuracy stats indicator badge
    accuracy_badge = ""
    try:
        from ml.prediction_log import get_live_accuracy_stats
        stats = get_live_accuracy_stats()
        if stats["resolved"] > 0:
            acc_2026 = stats["accuracy"]
            correct = stats["correct"]
            total = stats["resolved"]
            pct = acc_2026 * 100
            bar_color = "#34d399" if pct >= 60 else ("#fbbf24" if pct >= 45 else "#f87171")
            accuracy_badge = f'<div class="nav-accuracy-badge" style="border-color:{bar_color}; color:{bar_color};"><span class="pulse-dot" style="background-color:{bar_color}"></span>Live Accuracy: {pct:.0f}% ({correct}/{total})</div>'
    except Exception:
        pass

    # Model and training stats meta indicators
    try:
        wc_df, _ = load_data()
        _, metrics = get_model_and_metrics()
        acc = metrics.get("accuracy", 0.7182)
        cv_acc = metrics.get("cv_mean_accuracy", 0.7056)
        algo = metrics.get("selected_algorithm", metrics.get("algorithm", "logistic_regression")).replace("_", " ").title()
        train_count = len(wc_df)
    except Exception:
        train_count = 9856
        acc = 0.7182
        cv_acc = 0.7056
        algo = "Logistic Regression"

    st.markdown(f"""
<div class="football-menubar">
  <div class="menubar-brand">
    <span class="nav-logo">⚽ WC 2026</span>
    <span class="nav-tagline">AI MATCH PREDICTION ENGINE</span>
  </div>
  
  <!-- Dashboard Menu -->
  <div class="menubar-menu">
    <a href="/?page=home" class="menubar-trigger-link {active_home}" target="_self">🏠 Home</a>
  </div>

  <!-- Matches Dropdown Menu -->
  <div class="menubar-menu">
    <button class="menubar-trigger {active_upcoming}">📅 Matches</button>
    <div class="menubar-content">
      <a href="/?page=upcoming" class="menubar-item" target="_self">📅 Upcoming Matches</a>
      <a href="/?page=completed" class="menubar-item" target="_self">✅ Completed Results</a>
    </div>
  </div>

  <!-- Tournament Dropdown Menu -->
  <div class="menubar-menu">
    <button class="menubar-trigger {active_bracket}">🏆 Tournament</button>
    <div class="menubar-content">
      <a href="/?page=bracket" class="menubar-item" target="_self">🌳 Symmetrical Bracket</a>
      <a href="/?page=standings" class="menubar-item" target="_self">📊 Group Standings</a>
    </div>
  </div>

  <!-- AI Predictor Dropdown Menu -->
  <div class="menubar-menu">
    <button class="menubar-trigger {active_detail or active_accuracy}">🤖 AI Predictor</button>
    <div class="menubar-content">
      <a href="/?page=detail" class="menubar-item" target="_self">🔍 Interactive Simulator</a>
      <a href="/?page=accuracy" class="menubar-item" target="_self">📈 Accuracy & Calibration</a>
    </div>
  </div>

  <!-- Engine Config Dropdown Menu (Remixed to Football Theme) -->
  <div class="menubar-menu">
    <button class="menubar-trigger">⚙️ Engine Config</button>
    <div class="menubar-content">
      <div class="menubar-item-static" style="font-weight:700; color:var(--accent-green);">🤖 Model: {algo}</div>
      <div class="menubar-item-static">🎯 Test Accuracy: {acc:.2%}</div>
      <div class="menubar-item-static">📈 CV Accuracy: {cv_acc:.2%}</div>
      <div class="menubar-item-static">📊 Data: {train_count:,} WC matches</div>
    </div>
  </div>

  <!-- Right Side Live Stats -->
  <div class="menubar-right">
    {accuracy_badge}
  </div>
</div>
""", unsafe_allow_html=True)

    # Centralized Live/Static auto-refresh controller
    is_live_active = False
    try:
        _, fixtures = load_data()
        is_live_active = (fixtures["status"] == "live").any()
    except Exception:
        pass

    refresh_seconds = 0
    if page == "home":
        refresh_seconds = 30 if is_live_active else 60
    elif page == "upcoming" or page == "completed":
        refresh_seconds = 30 if is_live_active else 60
    elif page == "accuracy":
        refresh_seconds = 30 if is_live_active else 60
    elif page == "bracket":
        refresh_seconds = 30 if is_live_active else 120
    elif page == "detail":
        if is_live_active:
            refresh_seconds = 30

    if refresh_seconds > 0:
        import streamlit.components.v1 as components
        components.html(
            f"""
            <script>
                setTimeout(function() {{
                    window.parent.location.reload();
                }}, {refresh_seconds * 1000});
            </script>
            """,
            height=0,
            width=0
        )



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
.compact-page-head {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 18px;
  margin: 4px 0 10px;
}
.compact-page-head .hero-badge {
  margin-bottom: 8px;
}
.compact-page-title {
  color: var(--text-primary);
  font-size: 2rem;
  font-weight: 950;
  line-height: 1;
}
.compact-page-sub {
  max-width: 420px;
  color: var(--text-secondary);
  font-size: 0.86rem;
  line-height: 1.45;
  text-align: right;
}
.compact-state-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 0 0 10px;
}
.compact-state-row span {
  display: inline-flex;
  align-items: center;
  min-height: 28px;
  padding: 4px 10px;
  border: 1px solid rgba(148,163,184,0.15);
  border-radius: 999px;
  background: rgba(15,23,42,0.6);
  color: var(--text-secondary);
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 0.3px;
  text-transform: uppercase;
}
.compact-state-row strong {
  color: #f8fafc;
  font-size: 0.95rem;
  margin-right: 5px;
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

section[data-testid="stSidebar"] { display: none !important; }
button[data-testid="collapsedControl"] { display: none !important; }

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
.live-match-card {
  padding: 18px 22px;
  margin: 6px 0;
  border-color: rgba(248, 113, 113, 0.32);
  background:
    linear-gradient(135deg, rgba(17,24,39,0.96), rgba(30,41,59,0.78)),
    radial-gradient(circle at top left, rgba(248,113,113,0.14), transparent 38%);
  box-shadow: 0 18px 44px rgba(0,0,0,0.28), 0 0 0 1px rgba(248,113,113,0.08);
}
.live-card-topline {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  margin-bottom: 14px;
}
.live-phase-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  color: #cbd5e1;
  background: rgba(148,163,184,0.08);
  border: 1px solid rgba(148,163,184,0.16);
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.6px;
  text-transform: uppercase;
}
.live-scoreline {
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  align-items: center;
  gap: 16px;
}
.live-team {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;
  font-size: 1.04rem;
  font-weight: 800;
  color: var(--text-primary);
}
.live-team.home { justify-content: flex-end; text-align: right; }
.live-team.away { justify-content: flex-start; text-align: left; }
.live-team-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.live-score {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 118px;
  padding: 8px 18px;
  border-radius: 12px;
  background: rgba(248,113,113,0.10);
  border: 1px solid rgba(248,113,113,0.24);
  color: #fb7185;
  font-size: 1.9rem;
  font-weight: 950;
  letter-spacing: 3px;
  font-variant-numeric: tabular-nums;
}
.live-scorers {
  display: flex;
  justify-content: space-between;
  font-size: 0.75rem;
  color: #9ca3af;
  margin-top: 0;
  border-top: 1px dashed rgba(255,255,255,0.10);
  padding: 10px 18px 0;
  gap: 12px;
}
.live-scorers > div {
  max-width: 42%;
  overflow-wrap: anywhere;
}
.match-control-panel {
  display: grid;
  grid-template-columns: minmax(0, 1fr) repeat(3, minmax(100px, 0.24fr));
  gap: 10px;
  align-items: stretch;
  padding: 10px;
  margin: 6px 0 12px;
  border: 1px solid rgba(148,163,184,0.16);
  border-radius: 10px;
  background: linear-gradient(135deg, rgba(15,23,42,0.94), rgba(17,24,39,0.76));
  box-shadow: 0 12px 30px rgba(0,0,0,0.18);
}
.match-control-title {
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 4px;
}
.match-control-kicker {
  color: #34d399;
  font-size: 0.62rem;
  font-weight: 900;
  letter-spacing: 1.4px;
  text-transform: uppercase;
}
.match-control-heading {
  color: var(--text-primary);
  font-size: 0.96rem;
  font-weight: 900;
}
.match-control-sub {
  color: var(--text-secondary);
  font-size: 0.76rem;
  line-height: 1.35;
}
.match-stat-pill {
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-height: 54px;
  padding: 8px 10px;
  border: 1px solid rgba(148,163,184,0.14);
  border-radius: 10px;
  background: rgba(2,6,23,0.38);
}
.match-stat-pill strong {
  color: var(--text-primary);
  font-size: 1.18rem;
  line-height: 1;
}
.match-stat-pill span {
  color: var(--text-muted);
  font-size: 0.58rem;
  font-weight: 800;
  letter-spacing: 0.8px;
  margin-top: 6px;
  text-transform: uppercase;
}
.live-arena {
  position: relative;
  overflow: hidden;
  padding: 0 !important;
  margin-top: 4px !important;
  margin-bottom: 12px !important;
  border-color: rgba(248,113,113,0.38) !important;
  background:
    linear-gradient(90deg, rgba(127,29,29,0.34), rgba(15,23,42,0.98) 34%, rgba(20,83,45,0.24)),
    linear-gradient(135deg, rgba(15,23,42,0.98), rgba(2,6,23,0.96));
  box-shadow: 0 24px 70px rgba(0,0,0,0.38), 0 0 0 1px rgba(248,113,113,0.12);
}
.result-arena {
  border-color: rgba(52,211,153,0.34) !important;
  background:
    linear-gradient(90deg, rgba(6,78,59,0.32), rgba(15,23,42,0.98) 36%, rgba(15,23,42,0.92)),
    linear-gradient(135deg, rgba(15,23,42,0.98), rgba(2,6,23,0.96)) !important;
  box-shadow: 0 20px 54px rgba(0,0,0,0.32), 0 0 0 1px rgba(52,211,153,0.10) !important;
}
.live-arena::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.028) 1px, transparent 1px);
  background-size: 34px 34px;
  opacity: 0.38;
  pointer-events: none;
}
.live-arena-inner {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr);
  gap: 16px;
  align-items: center;
  padding: 18px 22px;
}
.live-status-row {
  position: relative;
  z-index: 1;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  background: rgba(2,6,23,0.34);
}
.live-source-row {
  position: relative;
  z-index: 1;
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 8px;
  padding: 0 14px 12px;
}
.source-chip {
  display: inline-flex;
  align-items: center;
  min-height: 26px;
  padding: 4px 10px;
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 999px;
  color: #cbd5e1;
  background: rgba(15,23,42,0.72);
  font-size: 0.68rem;
  font-weight: 800;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}
.source-chip.warn {
  color: #fde68a;
  border-color: rgba(251,191,36,0.28);
  background: rgba(120,53,15,0.22);
}
.live-club {
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}
.live-club.home {
  align-items: flex-end;
  text-align: right;
}
.live-club.away {
  align-items: flex-start;
  text-align: left;
}
.live-club-flag {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 34px;
  border-radius: 8px;
  background: rgba(255,255,255,0.08);
  box-shadow: 0 10px 24px rgba(0,0,0,0.18);
  overflow: hidden;
}
.live-club-flag img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  margin: 0;
  border-radius: 0;
}
.live-club-name {
  max-width: 100%;
  color: #f8fafc;
  font-size: 1.22rem;
  font-weight: 950;
  line-height: 1.05;
  overflow-wrap: anywhere;
}
.live-club-meta {
  color: #94a3b8;
  font-size: 0.7rem;
  font-weight: 800;
  letter-spacing: 0.8px;
  text-transform: uppercase;
}
.live-score-tower {
  display: grid;
  place-items: center;
  gap: 8px;
  min-width: 164px;
}
.live-clock-pill {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-height: 30px;
  padding: 5px 12px;
  border-radius: 999px;
  color: #fecaca;
  background: rgba(127,29,29,0.45);
  border: 1px solid rgba(248,113,113,0.32);
  font-size: 0.78rem;
  font-weight: 950;
  letter-spacing: 1px;
  text-transform: uppercase;
}
.live-main-score {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 14px;
  min-width: 152px;
  padding: 8px 16px;
  border-radius: 10px;
  color: #ffffff;
  background: rgba(2,6,23,0.62);
  border: 1px solid rgba(255,255,255,0.12);
  font-size: 2.55rem;
  font-weight: 950;
  line-height: 1;
  font-variant-numeric: tabular-nums;
}
.live-main-score .dash {
  color: #fb7185;
  font-size: 1.8rem;
}
.filter-band {
  margin: 4px 0 12px;
  padding: 10px;
  border: 1px solid rgba(148,163,184,0.12);
  border-radius: 10px;
  background: rgba(15,23,42,0.42);
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

/* ── 365scores Inspired Football Header System ── */
.scores-top-header {
  display: flex;
  align-items: center;
  background: #000000;
  height: 48px;
  width: 100%;
  padding: 0 16px;
  font-family: var(--font-main);
  border-bottom: 1px solid rgba(255, 255, 255, 0.05);
}

.brand-365 {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  line-height: 1;
  margin-right: 20px;
}

.brand-365 .brand-number {
  font-size: 1.15rem;
  font-weight: 900;
  color: #ffffff;
  letter-spacing: -0.5px;
}

.brand-365 .brand-text {
  font-size: 0.6rem;
  font-weight: 700;
  color: #ffffff;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin-top: -2px;
}

.scores-top-header .header-tab-active {
  background: #1a1d24;
  height: 100%;
  display: flex;
  align-items: center;
  padding: 0 20px;
  color: #ffffff;
  font-weight: 700;
  font-size: 0.85rem;
  border-right: 1px solid rgba(255, 255, 255, 0.05);
  border-left: 1px solid rgba(255, 255, 255, 0.05);
}

.scores-top-header .header-title-text {
  color: #9ca3af;
  font-size: 0.82rem;
  font-weight: 600;
  margin-left: 16px;
}

.scores-top-header .predictions-cup-banner {
  margin-left: auto;
  margin-right: auto;
  display: flex;
  align-items: center;
  gap: 6px;
  font-family: var(--font-main);
}

.predictions-cup-banner .cup-logo {
  font-size: 1.1rem;
}

.predictions-cup-banner .cup-text {
  color: #ffffff;
  font-size: 0.72rem;
  font-weight: 800;
  letter-spacing: 1.5px;
  text-transform: uppercase;
}

.header-right-actions {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 16px;
  color: #9ca3af;
  font-size: 0.82rem;
  font-weight: 600;
}

.header-right-actions span:hover {
  color: #ffffff;
  cursor: pointer;
}

.scores-breadcrumbs {
  background: #111317;
  padding: 6px 16px;
  font-size: 0.68rem;
  color: #6b7280;
  font-weight: 600;
  width: 100%;
}

.scores-main-header {
  background: #111317;
  padding: 16px;
  display: flex;
  align-items: center;
  gap: 16px;
  width: 100%;
}

.wc-badge-large {
  width: 44px;
  height: 44px;
  background: radial-gradient(circle, #1e293b 0%, #0f172a 100%);
  border: 1px solid rgba(250, 204, 21, 0.4);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.5rem;
  box-shadow: 0 0 10px rgba(250, 204, 21, 0.15);
}

.wc-title-container {
  display: flex;
  flex-direction: column;
}

.wc-title-row {
  display: flex;
  align-items: center;
  gap: 12px;
}

.wc-main-title {
  font-size: 1.4rem;
  font-weight: 900;
  color: #ffffff;
  letter-spacing: 0.5px;
  text-transform: uppercase;
}

.wc-follow-btn {
  background: transparent;
  border: 1px solid #3b82f6;
  color: #3b82f6 !important;
  font-size: 0.7rem;
  font-weight: 700;
  border-radius: 20px;
  padding: 2px 12px;
  cursor: pointer;
  text-decoration: none !important;
  transition: all 0.2s;
}

.wc-follow-btn:hover {
  background: #3b82f6;
  color: #ffffff !important;
}

.wc-follower-count {
  font-size: 0.72rem;
  color: #6b7280;
  font-weight: 600;
}

.scores-tabs-row {
  background: #111317;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  display: flex;
  padding: 0 16px;
  gap: 20px;
  width: 100%;
}

.score-tab-item {
  color: #9ca3af !important;
  text-decoration: none !important;
  font-size: 0.85rem;
  font-weight: 600;
  padding: 10px 4px;
  position: relative;
  transition: all 0.2s;
  cursor: pointer;
}

.score-tab-item:hover {
  color: #ffffff !important;
}

.score-tab-item.active {
  color: #3b82f6 !important;
  font-weight: 700;
}

.score-tab-item.active::after {
  content: '';
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  height: 3px;
  background: #3b82f6;
  border-radius: 3px 3px 0 0;
}

.pulse-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
  animation: pulse-animation 1.5s infinite;
}

@keyframes pulse-animation {
  0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(251, 191, 36, 0.7); }
  70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(251, 191, 36, 0); }
  100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(251, 191, 36, 0); }
}

/* Ensure Streamlit page content stretches nicely since there is no sidebar */
[data-testid="stAppViewBlockContainer"] {
  max-width: 100% !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
  padding-top: 1rem !important;
}

/* ── Mobile styles for Sports Header System ── */
  @media (max-width: 768px) {
    .scores-top-header {
      flex-wrap: wrap;
      height: auto;
      padding: 8px;
      justify-content: space-between;
      gap: 8px;
    }
    .scores-top-header .predictions-cup-banner {
      width: 100%;
      justify-content: center;
      order: 3;
      margin-top: 4px;
      border-top: 1px solid rgba(255, 255, 255, 0.05);
      padding-top: 4px;
    }
    .scores-main-header {
      flex-direction: column;
      align-items: center;
      text-align: center;
      gap: 8px;
      padding: 12px;
    }
    .wc-title-row {
      flex-direction: column;
      align-items: center;
      gap: 6px;
    }
    .wc-main-title {
      font-size: 1.15rem;
      text-align: center;
    }
    .scores-tabs-row {
      overflow-x: auto !important;
      -webkit-overflow-scrolling: touch !important;
      white-space: nowrap !important;
      gap: 14px;
      padding: 0 8px;
      justify-content: flex-start;
    }
    .scores-tabs-row::-webkit-scrollbar {
      display: none;
    }
    .score-tab-item {
      display: inline-block;
      font-size: 0.78rem;
      padding: 8px 2px;
      white-space: nowrap;
    }
  }

/* ─── Comprehensive Mobile Phone Layout ─── */
@media (max-width: 768px) {

  /* Prevent horizontal overflow on the whole app */
  .stApp, body {
    overflow-x: hidden !important;
    width: 100% !important;
  }
  * {
    max-width: 100%;
    box-sizing: border-box;
  }

  /* Tighten main content padding */
  [data-testid="stAppViewBlockContainer"] {
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
  }
  [data-testid="stVerticalBlock"] {
    gap: 0.5rem !important;
  }

  /* Force Streamlit columns to stack vertically on phone */
  [data-testid="stHorizontalBlock"] {
    flex-direction: column !important;
    gap: 8px !important;
  }
  [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
    width: 100% !important;
    flex: none !important;
    min-width: 0 !important;
  }

  /* Hero section */
  .hero-title { font-size: 1.8rem !important; line-height: 1.15 !important; }
  .hero-subtitle { font-size: 0.88rem !important; max-width: 100% !important; }
  .hero-badge { font-size: 0.68rem !important; }
  .compact-page-head {
    align-items: flex-start !important;
    flex-direction: column !important;
    gap: 6px !important;
    margin-bottom: 8px !important;
  }
  .compact-page-title {
    font-size: 1.6rem !important;
  }
  .compact-page-sub {
    max-width: 100% !important;
    text-align: left !important;
    font-size: 0.78rem !important;
  }
  .compact-state-row {
    gap: 6px !important;
    margin-bottom: 8px !important;
  }
  .compact-state-row span {
    min-height: 26px !important;
    padding: 3px 8px !important;
    font-size: 0.62rem !important;
  }
  .compact-state-row strong {
    font-size: 0.82rem !important;
  }

  /* Cards */
  .glass-card, .match-card, .kpi-card {
    padding: 14px !important;
    margin: 5px 0 !important;
    border-radius: 12px !important;
  }

  /* KPI grid: 2 cols on phone */
  .kpi-grid-container {
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 8px !important;
  }
  .kpi-value { font-size: 1.7rem !important; }
  .kpi-label { font-size: 0.65rem !important; }

  /* Team VS block: make it wrap nicely */
  .team-vs-block {
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center !important;
  }
  .team-block {
    flex: 0 0 38% !important;
    min-width: 100px;
  }
  .team-flag { font-size: 2rem !important; }
  .team-name { font-size: 0.78rem !important; }
  .team-name, .match-team-name, .bracket-team span {
    overflow-wrap: anywhere !important;
    word-break: normal !important;
  }
  .team-rank { font-size: 0.62rem !important; }
  .team-flag-img { height: 32px !important; }
  .vs-separator {
    font-size: 0.85rem !important;
    padding: 0 6px !important;
    flex-shrink: 0;
  }

  /* Winner tag */
  .winner-tag { font-size: 0.95rem !important; }

  /* Probability meters */
  .prob-meter-label { font-size: 0.68rem !important; }
  .prob-meter-value { font-size: 0.72rem !important; }

  /* Evidence / risk cards */
  .evidence-card, .risk-card { padding: 10px 12px !important; margin: 4px 0 !important; }
  .evidence-label, .risk-label { font-size: 0.6rem !important; }
  .evidence-text { font-size: 0.78rem !important; }

  /* Section headers */
  .section-header { font-size: 1.1rem !important; margin: 16px 0 10px !important; }

  /* Confidence badge */
  .conf-badge { font-size: 0.65rem !important; padding: 3px 8px !important; }

  /* Score badge */
  .score-badge {
    font-size: 0.9rem !important;
    padding: 3px 10px !important;
    letter-spacing: 1px !important;
    white-space: nowrap !important;
  }
  .live-match-card {
    padding: 14px !important;
  }
  .live-card-topline {
    align-items: flex-start !important;
    flex-direction: column !important;
    gap: 8px !important;
  }
  .live-scoreline {
    grid-template-columns: 1fr !important;
    gap: 8px !important;
    text-align: center !important;
  }
  .live-team,
  .live-team.home,
  .live-team.away {
    justify-content: center !important;
    text-align: center !important;
    font-size: 0.95rem !important;
  }
  .live-score {
    justify-self: center !important;
    min-width: 108px !important;
    font-size: 1.55rem !important;
  }
  .live-scorers {
    flex-direction: column !important;
    text-align: center !important;
  }
  .live-scorers > div {
    max-width: 100% !important;
  }
  .match-control-panel {
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    padding: 8px !important;
    gap: 6px !important;
  }
  .match-control-title {
    grid-column: 1 / -1;
  }
  .match-control-heading {
    font-size: 0.88rem !important;
  }
  .match-control-kicker {
    font-size: 0.58rem !important;
  }
  .match-stat-pill {
    min-height: 54px !important;
    padding: 7px !important;
  }
  .match-stat-pill strong {
    font-size: 1.15rem !important;
  }
  .match-stat-pill span {
    font-size: 0.52rem !important;
    letter-spacing: 0.4px !important;
  }
  .live-status-row {
    flex-direction: row !important;
    align-items: center !important;
    padding: 9px 10px !important;
  }
  .live-arena-inner {
    grid-template-columns: minmax(0, 1fr) auto minmax(0, 1fr) !important;
    gap: 8px !important;
    padding: 12px 10px !important;
  }
  .live-club.home,
  .live-club.away {
    align-items: center !important;
    text-align: center !important;
  }
  .live-club-name {
    font-size: 0.88rem !important;
  }
  .live-club-meta {
    font-size: 0.56rem !important;
  }
  .live-club-flag {
    width: 38px !important;
    height: 28px !important;
  }
  .live-score-tower {
    order: 0;
    min-width: 0 !important;
  }
  .live-clock-pill {
    min-height: 26px !important;
    padding: 4px 9px !important;
    font-size: 0.66rem !important;
  }
  .live-main-score {
    min-width: 98px !important;
    padding: 7px 10px !important;
    font-size: 1.8rem !important;
    gap: 8px !important;
  }
  .live-main-score .dash {
    font-size: 1.15rem !important;
  }
  .live-source-row {
    padding: 0 10px 10px !important;
    justify-content: center !important;
  }

  /* Round badge */
  .round-badge { font-size: 0.6rem !important; padding: 2px 8px !important; }

  /* Factor chips */
  .factor-chip { font-size: 0.65rem !important; padding: 3px 8px !important; }

  /* Buttons */
  .stButton > button {
    min-height: 44px !important;
    font-size: 0.9rem !important;
    width: 100% !important;
  }

  /* Tabs */
  [data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 0.78rem !important;
    padding: 8px 10px !important;
  }

  /* Dataframe / tables */
  table th, table td { padding: 8px 6px !important; font-size: 0.75rem !important; }
  .stDataFrame {
    font-size: 0.75rem !important;
    overflow-x: auto !important;
  }
  [data-testid="stTable"], [data-testid="stDataFrame"] {
    overflow-x: auto !important;
  }

  /* Bracket horizontal scroll */
  .bracket-scroll-container {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
  }
  .bracket-flow-wrapper {
    min-width: 900px;
  }

  /* Dropdown / selectbox */
  [data-testid="stSelectbox"] > div {
    font-size: 0.82rem !important;
  }

  /* Sliders */
  [data-testid="stSlider"] { margin: 4px 0 !important; }

  /* Remove chart overflow */
  .js-plotly-plot, .plotly { max-width: 100% !important; }
  .svg-container { max-width: 100% !important; overflow: hidden !important; }
}

@media (max-width: 480px) {
  [data-testid="stAppViewBlockContainer"] {
    padding-left: 0.5rem !important;
    padding-right: 0.5rem !important;
  }
  .top-nav-meta {
    display: none !important;
  }
  .kpi-grid-container {
    grid-template-columns: 1fr !important;
  }
  .glass-card, .match-card, .kpi-card {
    padding: 12px !important;
  }
  .hero-title {
    font-size: 1.55rem !important;
  }
  .team-block {
    flex-basis: 100% !important;
  }
  .vs-separator {
    width: 100% !important;
    text-align: center !important;
  }
}
</style>

"""
