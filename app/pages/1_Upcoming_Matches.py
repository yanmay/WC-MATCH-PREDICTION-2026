"""
Upcoming Matches — FIFA World Cup 2026 AI Predictor | Dark-mode 21st.dev design language
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import load_data, get_model_and_metrics, get_team_flag, format_team_html, format_team_emoji, format_match_date, CSS, ROUND_COLORS, ROUND_ORDER, render_sidebar
from ml.predict import predict_match_with_evidence, predict_match, get_winner_label, get_confidence_label


def make_live_timer_html(elapsed_mins: int, elapsed_secs: int, timer_id: str = "live-timer-1") -> str:
    """JavaScript-powered live-ticking match minute counter."""
    if elapsed_mins >= 90:
        extra = elapsed_mins - 90
        init_display = f"90+{extra}' {str(elapsed_secs).zfill(2)}"
    else:
        init_display = f"{elapsed_mins}' {str(elapsed_secs).zfill(2)}"

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


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_predict(home_team, away_team, round_name, _wc_df_hash):
    """Cached wrapper — predictions are expensive, cache for 1 hour."""
    from app.utils import load_data
    wc_df, _ = load_data()
    return predict_match_with_evidence(
        home_team=home_team,
        away_team=away_team,
        round_name=round_name,
        wc_df=wc_df,
    )


def _get_flag_code(team: str) -> str:
    """Return the ISO 2-letter country code for flagcdn.com, or empty string."""
    TEAM_CODES = {
        "Argentina": "ar", "France": "fr", "England": "gb-eng", "Belgium": "be",
        "Brazil": "br", "Portugal": "pt", "Netherlands": "nl", "Spain": "es",
        "Croatia": "hr", "Italy": "it", "Morocco": "ma", "USA": "us",
        "Mexico": "mx", "Germany": "de", "Colombia": "co", "Uruguay": "uy",
        "Denmark": "dk", "Switzerland": "ch", "Japan": "jp", "South Korea": "kr",
        "Australia": "au", "Canada": "ca", "Senegal": "sn", "Ecuador": "ec",
        "Serbia": "rs", "Poland": "pl", "Iran": "ir", "Ghana": "gh",
        "Cameroon": "cm", "Egypt": "eg", "Wales": "gb-wls", "Saudi Arabia": "sa",
        "Qatar": "qa", "Russia": "ru", "Sweden": "se", "Chile": "cl",
        "Peru": "pe", "Paraguay": "py", "Costa Rica": "cr", "Algeria": "dz",
        "Nigeria": "ng", "Ivory Coast": "ci", "Tunisia": "tn", "South Africa": "za",
        "Ukraine": "ua", "Turkey": "tr", "Greece": "gr", "Slovakia": "sk",
        "Hungary": "hu", "Romania": "ro", "Austria": "at", "Iceland": "is",
        "New Zealand": "nz", "North Korea": "kp", "Norway": "no",
        "Scotland": "gb-sct", "Czechia": "cz", "Bosnia and Herzegovina": "ba",
        "Iraq": "iq", "Jordan": "jo", "Uzbekistan": "uz", "Haiti": "ht",
        "DR Congo": "cd", "Curacao": "cw", "Cabo Verde": "cv", "Panama": "pa",
    }
    return TEAM_CODES.get(team, "")

st.set_page_config(
    page_title="Upcoming Matches | FIFA WC 2026 Predictor",
    page_icon="📅",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)

# Auto-refresh every 30 seconds for live scores
st.markdown("""
<meta http-equiv="refresh" content="30">
""", unsafe_allow_html=True)

render_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-badge">📅 FIXTURES &amp; LIVE PREDICTIONS</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title" style="font-size:2.5rem; line-height:1.2;">Upcoming Matches</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Explore upcoming matches with deep AI evidence, historical facts, and probability calibrations. Data refreshes every 30 seconds from live sources.</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Analyzing match statistics and generating predictions..."):
    wc_df, fixtures = load_data()
    pipeline, model_metrics = get_model_and_metrics()

# ── Filters ───────────────────────────────────────────────────────────────────
col_f1, col_f2 = st.columns([2, 3])
with col_f1:
    available_rounds = sorted(
        fixtures["round"].unique().tolist(),
        key=lambda r: ["Group Stage", "Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "3rd Place", "Final"].index(r)
        if r in ["Group Stage", "Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "3rd Place", "Final"] else 99
    )
    default_rounds = [r for r in ["Group Stage", "Round of 32"] if r in available_rounds]
    selected_rounds = st.multiselect(
        "Filter by Round",
        options=available_rounds,
        default=default_rounds,
        help="Select which tournament rounds to show"
    )
with col_f2:
    all_teams = sorted(set(
        fixtures["home_team"].tolist() + fixtures["away_team"].tolist()
    ) - {"TBD"})
    selected_team = st.selectbox("Filter by Team (optional)", ["All Teams"] + all_teams)

st.markdown("<br>", unsafe_allow_html=True)

# ── LIVE Matches Section ──────────────────────────────────────────────────────
live_matches = fixtures[fixtures["status"] == "live"].copy()
if not live_matches.empty:
    st.markdown("""
    <div style="display:inline-flex; align-items:center; gap:8px; margin-bottom:8px;">
        <span class="live-badge"><span class="live-dot"></span>LIVE MATCHES</span>
    </div>""", unsafe_allow_html=True)
    for _, lm in live_matches.iterrows():
        hs = int(lm.get("home_score", 0) or 0)
        as_ = int(lm.get("away_score", 0) or 0)
        hcode = _get_flag_code(lm["home_team"])
        acode = _get_flag_code(lm["away_team"])
        hflag = f'<img src="https://flagcdn.com/w40/{hcode}.png" class="flag-icon">' if hcode else "&#127987;"
        aflag = f'<img src="https://flagcdn.com/w40/{acode}.png" class="flag-icon">' if acode else "&#127987;"
        elapsed_mins = int(lm.get("elapsed_mins", 0) or 0)
        elapsed_secs = int(lm.get("elapsed_secs", 0) or 0)
        # Create unique ID per match (use home+away team names)
        timer_id = f"timer-{lm['home_team'].replace(' ','-')}-{lm['away_team'].replace(' ','-')}"
        timer_html = make_live_timer_html(elapsed_mins, elapsed_secs, timer_id)
        
        # Scorers — built as a compact single-line string to avoid Markdown code-block interpretation
        scorers_html = ""
        try:
            import json
            scorers_dict = json.loads(lm["scorers"]) if isinstance(lm.get("scorers"), str) else lm.get("scorers", {})
            if scorers_dict and isinstance(scorers_dict, dict):
                h_s = ", ".join(scorers_dict.get(lm["home_team"], []))
                a_s = ", ".join(scorers_dict.get(lm["away_team"], []))
                if h_s or a_s:
                    scorers_html = (
                        f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;'
                        f'color:#9ca3af;margin-top:10px;border-top:1px dashed rgba(255,255,255,0.1);'
                        f'padding-top:8px;gap:8px;">'
                        f'<div style="text-align:left;max-width:40%;font-weight:600;color:#d1fae5;">{h_s}</div>'
                        f'<div style="text-align:center;color:#34d399;font-weight:800;">&#9917;</div>'
                        f'<div style="text-align:right;max-width:40%;font-weight:600;color:#fca5a5;">{a_s}</div>'
                        f'</div>'
                    )
        except Exception:
            pass

        # Card — all on one line to prevent Markdown treating indented HTML as code blocks
        st.markdown(
            f'<div class="glass-card" style="padding:16px 20px;margin:4px 0;border-color:rgba(239,113,113,0.35);box-shadow:0 0 24px rgba(239,113,113,0.12);">'
            f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">'
            f'<span class="live-badge"><span class="live-dot"></span>LIVE &middot; {timer_html}</span>'
            f'<span style="color:#9ca3af;font-size:0.72rem;">&#127942; {lm.get("round","")}</span>'
            f'</div>'
            f'<div style="display:flex;align-items:center;justify-content:space-between;gap:8px;">'
            f'<div style="flex:1;text-align:right;"><span style="font-size:1rem;font-weight:700;color:#f3f4f6;">{hflag} {lm["home_team"]}</span></div>'
            f'<div style="text-align:center;min-width:100px;"><span style="font-size:1.6rem;font-weight:900;color:#f87171;letter-spacing:4px;">{hs} &mdash; {as_}</span></div>'
            f'<div style="flex:1;text-align:left;"><span style="font-size:1rem;font-weight:700;color:#f3f4f6;">{aflag} {lm["away_team"]}</span></div>'
            f'</div>'
            f'{scorers_html}'
            f'</div>',
            unsafe_allow_html=True
        )
    st.markdown("<br>", unsafe_allow_html=True)

# ── Completed Matches Section ─────────────────────────────────────────────────
completed_all = fixtures[fixtures["status"] == "completed"].copy()
if "round" in completed_all.columns:
    completed_filtered = completed_all[
        completed_all["round"].isin(selected_rounds if selected_rounds else available_rounds)
    ]
else:
    completed_filtered = completed_all

if not completed_filtered.empty:
    st.markdown("""
    <div style="display:inline-flex; align-items:center; gap:8px; margin-bottom:8px;">
        <span style="font-size:1rem; font-weight:700; color:#f3f4f6;">&#x2705; Completed Results</span>
        <span class="live-badge"><span class="live-dot"></span>AUTO-REFRESH</span>
    </div>""", unsafe_allow_html=True)
    for _, cmatch in completed_filtered.iterrows():
        hs = cmatch.get("home_score", 0) or 0
        as_ = cmatch.get("away_score", 0) or 0
        if hs > as_:
            h_cls = "winner-team-name"; a_cls = "loser-team-name"; outcome_badge = '<span class="result-win">WIN</span>'
        elif as_ > hs:
            h_cls = "loser-team-name"; a_cls = "winner-team-name"; outcome_badge = '<span class="result-loss">LOSS</span>'
        else:
            h_cls = "team-name"; a_cls = "team-name"; outcome_badge = '<span class="result-draw">DRAW</span>'
        hcode = _get_flag_code(cmatch["home_team"])
        acode = _get_flag_code(cmatch["away_team"])
        hflag = f'<img src="https://flagcdn.com/w40/{hcode}.png" class="flag-icon">' if hcode else "🏳️"
        aflag = f'<img src="https://flagcdn.com/w40/{acode}.png" class="flag-icon">' if acode else "🏳️"

        # AI prediction badge
        try:
            from ml.prediction_log import log_prediction, get_outcome_badge_html
            pred_c = predict_match(
                home_team=cmatch["home_team"],
                away_team=cmatch["away_team"],
                round_name=cmatch.get("round", "Group Stage"),
                wc_df=wc_df,
            )
            log_prediction(
                match_id=int(cmatch["match_id"]),
                home_team=cmatch["home_team"],
                away_team=cmatch["away_team"],
                round_name=cmatch.get("round", "Group Stage"),
                predicted_outcome=pred_c["predicted_outcome"],
                confidence=pred_c["confidence"] or 0,
                home_win_prob=pred_c["home_win_prob"] or 0,
                draw_prob=pred_c["draw_prob"] or 0,
                away_win_prob=pred_c["away_win_prob"] or 0,
                match_date=str(cmatch.get("date", "")),
            )
            ai_badge = get_outcome_badge_html(int(cmatch["match_id"]), cmatch["home_team"], cmatch["away_team"])
        except Exception:
            ai_badge = ""

        date_label = format_match_date(str(cmatch.get("date", "")))
        date_cls = "today" if date_label == "Today" else ("yesterday" if date_label == "Yesterday" else "")

        st.markdown(f"""
        <div class="glass-card" style="padding:12px 18px; margin:4px 0;">
            <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
                <div style="display:flex; align-items:center; gap:10px; flex:1; justify-content:flex-end;">
                    <span class="{h_cls}" style="font-size:0.95rem;">{hflag} {cmatch['home_team']}</span>
                </div>
                <div style="text-align:center; min-width:90px;">
                    <span class="score-badge completed">{int(hs)} — {int(as_)}</span>
                    <div style="margin-top:4px; font-size:0.65rem;">
                        <span class="date-chip {date_cls}">{date_label}</span>
                    </div>
                    <div style="margin-top:2px;">{outcome_badge}</div>
                </div>
                <div style="display:flex; align-items:center; gap:10px; flex:1;">
                    <span class="{a_cls}" style="font-size:0.95rem;">{aflag} {cmatch['away_team']}</span>
                </div>
                <div style="min-width:88px; text-align:right;">{ai_badge}</div>
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ── Filter Upcoming Fixtures ──────────────────────────────────────────────────
upcoming = fixtures[
    (fixtures["status"] == "scheduled") &
    (fixtures["round"].isin(selected_rounds if selected_rounds else available_rounds))
].copy()

if selected_team != "All Teams":
    upcoming = upcoming[
        (upcoming["home_team"] == selected_team) | (upcoming["away_team"] == selected_team)
    ]

if upcoming.empty:
    st.info("No upcoming matches found for the selected filters.")
    st.stop()

# Group by round and display
for round_name in ROUND_ORDER:
    round_matches = upcoming[upcoming["round"] == round_name]
    if round_matches.empty:
        continue

    round_color = ROUND_COLORS.get(round_name, "#60a5fa")
    st.markdown(f"""
    <div style="
        display:inline-block;
        background:rgba(0,212,255,0.05);
        border-left: 4px solid {round_color};
        padding: 6px 16px;
        border-radius: 0 8px 8px 0;
        margin: 28px 0 16px;
        font-size:1rem; font-weight:700; color:{round_color};
        letter-spacing:1.5px; text-transform:uppercase;
    ">
        {round_name} · {len(round_matches)} matches
    </div>""", unsafe_allow_html=True)

    for _, match in round_matches.iterrows():
        is_tbd = match["home_team"] == "TBD" or match["away_team"] == "TBD"

        if not is_tbd:
            # Run prediction with complete evidence chain (cached)
            _wc_hash = len(wc_df)
            pred = _cached_predict(
                home_team=match["home_team"],
                away_team=match["away_team"],
                round_name=match["round"],
                _wc_df_hash=_wc_hash,
            )
            winner = get_winner_label(pred["predicted_outcome"], match["home_team"], match["away_team"])
            conf_label, conf_emoji = get_confidence_label(pred["confidence"])
            conf_cls = "conf-high" if conf_label == "High" else ("conf-medium" if conf_label == "Medium" else "conf-low")

            # Log this prediction so we can track accuracy when result arrives
            try:
                from ml.prediction_log import log_prediction
                log_prediction(
                    match_id=int(match["match_id"]),
                    home_team=match["home_team"],
                    away_team=match["away_team"],
                    round_name=match["round"],
                    predicted_outcome=pred["predicted_outcome"],
                    confidence=pred["confidence"] or 0,
                    home_win_prob=pred["home_win_prob"] or 0,
                    draw_prob=pred["draw_prob"] or 0,
                    away_win_prob=pred["away_win_prob"] or 0,
                    match_date=str(match.get("date", "")),
                )
            except Exception:
                pass
        else:
            pred = {
                "home_win_prob": None, "draw_prob": None, "away_win_prob": None,
                "predicted_outcome": "TBD", "confidence": None,
                "is_low_confidence": True,
                "confidence_note": "Teams not yet determined.",
                "is_knockout": True,
                "evidence": [], "risks": [], "verdict": "Match pending team qualification."
            }
            winner = "TBD"
            conf_label, conf_emoji = "Unknown", "❓"
            conf_cls = "conf-low"

        # ── Match Card ───────────────────────────────────────────────────────
        date_label = format_match_date(str(match.get("date", "")))
        date_cls = "today" if date_label == "Today" else ("yesterday" if date_label == "Yesterday" else "")
        st.markdown(f"""
        <div class="glass-card" style="padding:16px 20px 12px;">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:14px;">
                <span style="color:#9ca3af; font-size:0.75rem; font-weight:600;">
                    <span class="date-chip {date_cls}">&#128197; {date_label}</span>
                </span>
                <span style="font-size:0.72rem; color:#6b7280; font-weight:600;">ID: #{match['match_id']}</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # Team vs block
        home_code = _get_flag_code(match['home_team'])
        away_code = _get_flag_code(match['away_team'])
        col_home, col_vs, col_away = st.columns([3, 1, 3])
        with col_home:
            if home_code:
                st.markdown(f'<div style="text-align:center;padding:8px 0;"><img src="https://flagcdn.com/w80/{home_code}.png" style="height:48px;border-radius:4px;margin-bottom:6px;"><br><b style="font-size:1.1rem;color:#f3f4f6;">{match["home_team"]}</b></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align:center;padding:8px 0;font-size:2.5rem;">🏳️<br><b style="font-size:1.1rem;color:#f3f4f6;">{match["home_team"]}</b></div>', unsafe_allow_html=True)
        with col_vs:
            st.markdown('<div style="text-align:center;padding:20px 0;font-size:1rem;font-weight:700;color:#6b7280;">VS</div>', unsafe_allow_html=True)
        with col_away:
            if away_code:
                st.markdown(f'<div style="text-align:center;padding:8px 0;"><img src="https://flagcdn.com/w80/{away_code}.png" style="height:48px;border-radius:4px;margin-bottom:6px;"><br><b style="font-size:1.1rem;color:#f3f4f6;">{match["away_team"]}</b></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div style="text-align:center;padding:8px 0;font-size:2.5rem;">🏳️<br><b style="font-size:1.1rem;color:#f3f4f6;">{match["away_team"]}</b></div>', unsafe_allow_html=True)


        if not is_tbd:
            # Probability meter bars
            c_left, c_right = st.columns([3, 2], gap="large")
            with c_left:
                st.markdown('<div style="font-size:0.78rem; font-weight:700; color:#34d399; letter-spacing:1px; margin-bottom:8px;">PROBABILITY MODEL OUTCOMES</div>', unsafe_allow_html=True)
                hp = pred["home_win_prob"]
                dp = pred["draw_prob"] if not pred["is_knockout"] else 0
                ap = pred["away_win_prob"]

                # Home Prob
                st.markdown(f"""
                <div class="prob-meter-container">
                    <div class="prob-meter-header">
                        <span class="prob-meter-label">{format_team_html(match['home_team'])} Win</span>
                        <span class="prob-meter-value" style="color:#34d399;">{hp:.1%}</span>
                    </div>
                    <div class="prob-meter-track">
                        <div class="prob-meter-fill home" style="width:{hp*100:.1f}%"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

                # Draw Prob
                if not pred["is_knockout"]:
                    st.markdown(f"""
                    <div class="prob-meter-container">
                        <div class="prob-meter-header">
                            <span class="prob-meter-label">⚖️ Draw Probability</span>
                            <span class="prob-meter-value" style="color:#9ca3af;">{dp:.1%}</span>
                        </div>
                        <div class="prob-meter-track">
                            <div class="prob-meter-fill draw" style="width:{dp*100:.1f}%"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Away Prob
                st.markdown(f"""
                <div class="prob-meter-container">
                    <div class="prob-meter-header">
                        <span class="prob-meter-label">{format_team_html(match['away_team'])} Win</span>
                        <span class="prob-meter-value" style="color:#f87171;">{ap:.1%}</span>
                    </div>
                    <div class="prob-meter-track">
                        <div class="prob-meter-fill away" style="width:{ap*100:.1f}%"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            with c_right:
                st.markdown(f"""
                <div style="text-align:center; padding:12px; background:rgba(52,211,153,0.04);
                    border-radius:12px; border:1px solid rgba(52,211,153,0.15);
                    height:100%; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-size:0.65rem; color:#6b7280; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:4px;">AI PREDICTED WINNER</div>
                    <div class="winner-tag" style="margin:4px 0; font-size:1.4rem;">{format_team_html(winner)}</div>
                    <div style="margin-top:6px;">
                        <span class="conf-badge {conf_cls}">{conf_emoji} {conf_label} Confidence ({pred['confidence']:.1%})</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Deep Analysis Evidence Block
            with st.expander("🔬 View AI Evidence Breakdown & Historical Proofs"):
                # Natural language verdict
                st.markdown(f"""
                <div class="evidence-card" style="margin-bottom:14px; background:rgba(16,185,129,0.06); border-color:rgba(16,185,129,0.25);">
                    <div class="evidence-label" style="font-size:0.75rem;">🤖 AI FORECAST VERDICT</div>
                    <div class="evidence-text" style="font-size:0.85rem; font-weight:500;">{pred['verdict']}</div>
                </div>""", unsafe_allow_html=True)

                col_ev, col_risk = st.columns(2, gap="medium")
                with col_ev:
                    st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#34d399; letter-spacing:1px; margin-bottom:6px;">📈 SUPPORTING EVIDENCE CHAIN</div>', unsafe_allow_html=True)
                    for item in pred["evidence"]:
                        st.markdown(f"""
                        <div class="evidence-card">
                            <div class="evidence-label">{item['label']}</div>
                            <div class="evidence-text" style="font-size:0.78rem;">{item['text']}</div>
                            <div class="evidence-sub">{item['sub']}</div>
                        </div>""", unsafe_allow_html=True)

                with col_risk:
                    st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#f87171; letter-spacing:1px; margin-bottom:6px;">⚠️ POTENTIAL RISK FACTORS &amp; UPSET TRIGGERS</div>', unsafe_allow_html=True)
                    for item in pred["risks"]:
                        st.markdown(f"""
                        <div class="risk-card">
                            <div class="risk-label">{item['label']}</div>
                            <div class="evidence-text" style="font-size:0.78rem;">{item['text']}</div>
                        </div>""", unsafe_allow_html=True)

                    # Form indicator sparkline summary
                    st.markdown('<div style="margin-top:12px; font-size:0.75rem; font-weight:700; color:#9ca3af;">FORM TRAJECTORY (LAST 5 WC MATCHES)</div>', unsafe_allow_html=True)
                    
                    def render_mini_form(form_list):
                        html = '<div style="display:flex; gap:6px; margin-top:4px;">'
                        for game in form_list:
                            color = "#10b981" if game["result"] == "W" else "#ef4444" if game["result"] == "L" else "#6b7280"
                            html += f"""
                            <span style="
                                display:inline-flex; align-items:center; justify-content:center;
                                width:24px; height:24px; border-radius:6px;
                                font-size:0.75rem; font-weight:700; color:white;
                                background:{color};
                            " title="{game['year']} vs {game['opponent']} ({game['score']})">
                                {game['result']}
                            </span>"""
                        html += "</div>"
                        return html

                    st.markdown(f"""
                    <div style="background:rgba(31,41,55,0.95); padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,0.08); margin-top:6px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-size:0.72rem; color:#9ca3af; font-weight:600;">{format_team_html(match['home_team'])} Form:</span>
                            {render_mini_form(pred['home_form'])}
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:0.72rem; color:#9ca3af; font-weight:600;">{format_team_html(match['away_team'])} Form:</span>
                            {render_mini_form(pred['away_form'])}
                        </div>
                    </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="padding:16px 0; text-align:center; background:rgba(17,24,39,0.8); border-radius:12px; border:1px solid rgba(255,255,255,0.08);">
                <div style="color:#9ca3af; font-size:0.85rem; font-weight:600;">🔒 Match Pending Team Qualification</div>
                <div style="color:#6b7280; font-size:0.75rem; margin-top:4px;">Group stage must complete before predictions unlock for this bracket slot.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr style="border:none; border-top:1px solid rgba(255,255,255,0.06); margin:20px 0;">', unsafe_allow_html=True)

# ── Summary Stats ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Tournament Predictions Overview</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
non_tbd = upcoming[(upcoming["home_team"] != "TBD") & (upcoming["away_team"] != "TBD")]
with c1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{len(non_tbd)}</div>
        <div class="kpi-label">Scheduled Predictions Generated</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{len(selected_rounds or available_rounds)}</div>
        <div class="kpi-label">Rounds Checked</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{model_metrics.get('accuracy', 0):.1%}</div>
        <div class="kpi-label">Model Pipeline Accuracy</div>
    </div>""", unsafe_allow_html=True)
