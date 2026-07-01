"""
Home — FIFA World Cup 2026 AI Predictor | 21st.dev glass design system
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import load_data, get_model_and_metrics, get_team_flag, format_team_html, format_team_emoji, format_match_date, CSS, ROUND_COLORS, ROUND_ORDER, render_sidebar, map_feature_name
from ml.features import get_ranking
import textwrap


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
def _cached_home_prediction(home_team, away_team, round_name, _wc_len):
    """Cached featured match prediction for home page."""
    from app.utils import load_data
    from ml.predict import predict_match_with_evidence
    wc_df, _ = load_data()
    return predict_match_with_evidence(
        home_team=home_team,
        away_team=away_team,
        round_name=round_name,
        wc_df=wc_df,
    )

st.set_page_config(
    page_title="FIFA WC 2026 AI Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
render_sidebar()

# ── Load ───────────────────────────────────────────────────────────────────────
with st.spinner("Initialising AI engine..."):
    try:
        wc_df, fixtures = load_data()
        pipeline, model_metrics = get_model_and_metrics()
        model_ready = True
    except Exception as e:
        model_ready = False
        st.error(f"Error: {e}")

# ── Hero ───────────────────────────────────────────────────────────────────────
col_hero, col_match = st.columns([3, 2], gap="large")

with col_hero:
    st.markdown('<div class="hero-badge">🔴 LIVE — FIFA WORLD CUP 2026</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hero-title">
        AI-Powered<br>Match Predictions
    </div>
    <div class="hero-subtitle">
        Every prediction is backed by 90+ years of World Cup data, 
        feature-level evidence, and a calibrated ML pipeline — not guesswork.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row
    if model_ready:
        total_wc = len(wc_df)
        upcoming_count = len(fixtures[fixtures["status"] == "scheduled"])
        acc = model_metrics.get("accuracy", 0)
        cv_acc = model_metrics.get("cv_mean_accuracy", 0)
        algo = model_metrics.get("algorithm", "random_forest").replace("_", " ").title()

        # Live 2026 prediction accuracy calculations
        try:
            from ml.prediction_log import get_live_accuracy_stats
            pstats = get_live_accuracy_stats()
            if pstats["resolved"] > 0:
                pct = pstats["accuracy"] * 100
                kpi_val = f"{pct:.0f}%"
                kpi_sub = f"{pstats['correct']}/{pstats['resolved']} correct"
                delta_col = "positive" if pct >= 60 else "negative"
                delta_txt = "↑ Live WC 2026" if pct >= 60 else "↓ Live WC 2026"
            else:
                kpi_val = "—"
                kpi_sub = "No results yet"
                delta_col = ""
                delta_txt = ""
        except Exception:
            kpi_val = "—"
            kpi_sub = "Tracking..."
            delta_col = ""
            delta_txt = ""

        kpi_html = f"""
        <div class="kpi-grid-container">
            <div class="kpi-card" style="margin: 0 !important;">
                <div class="kpi-value">{total_wc:,}</div>
                <div class="kpi-label">WC Matches Trained</div>
            </div>
            <div class="kpi-card" style="margin: 0 !important;">
                <div class="kpi-value">{upcoming_count}</div>
                <div class="kpi-label">Games to Predict</div>
            </div>
            <div class="kpi-card" style="margin: 0 !important;">
                <div class="kpi-value">{cv_acc:.1%}</div>
                <div class="kpi-label">CV Accuracy</div>
                <div class="kpi-delta positive">↑ vs 50.6% baseline</div>
            </div>
            <div class="kpi-card" style="margin: 0 !important;">
                <div class="kpi-value">48</div>
                <div class="kpi-label">Teams · 2026</div>
            </div>
            <div class="kpi-card kpi-accuracy-2026" style="margin: 0 !important;">
                <div class="kpi-value">{kpi_val}</div>
                <div class="kpi-label">🎯 2026 AI Accuracy</div>
                {f'<div class="kpi-delta {delta_col}">{delta_txt}</div>' if delta_txt else ''}
                <div style="font-size:0.65rem;color:#6b7280;margin-top:2px;">{kpi_sub}</div>
            </div>
        </div>
        """
        st.markdown(kpi_html, unsafe_allow_html=True)

# ── Next Match Preview ─────────────────────────────────────────────────────────
with col_match:
    if model_ready:
        live_df = fixtures[fixtures["status"] == "live"]
        upcoming_df = fixtures[
            (fixtures["status"] == "scheduled") &
            (fixtures["home_team"] != "TBD") &
            (fixtures["away_team"] != "TBD")
        ]
        if not live_df.empty:
            next_m = live_df.iloc[0]
            is_live_match = True
        elif not upcoming_df.empty:
            next_m = upcoming_df.iloc[0]
            is_live_match = False
        else:
            next_m = None
            is_live_match = False

        if next_m is not None:
            from ml.predict import get_winner_label, get_confidence_label
            pred = _cached_home_prediction(
                home_team=next_m["home_team"],
                away_team=next_m["away_team"],
                round_name=next_m["round"],
                _wc_len=len(wc_df),
            )

            winner = get_winner_label(pred["predicted_outcome"], next_m["home_team"], next_m["away_team"])
            conf_label, conf_emoji = get_confidence_label(pred["confidence"])
            conf_cls = "conf-high" if conf_label == "High" else ("conf-medium" if conf_label == "Medium" else "conf-low")
            round_color = ROUND_COLORS.get(next_m["round"], "#60a5fa")

            if is_live_match:
                elapsed_mins = int(next_m.get("elapsed_mins", 0) or 0)
                elapsed_secs = int(next_m.get("elapsed_secs", 0) or 0)
                timer_html = make_live_timer_html(elapsed_mins, elapsed_secs, "home-live-timer")
                badge_html = f'<span class="round-badge" style="background:rgba(239,113,113,0.1);color:#f87171;border:1px solid rgba(239,113,113,0.25);"><span class="live-dot"></span> LIVE &middot; {timer_html}</span>'
                vs_html = f'<div style="text-align:center;min-width:90px;"><span class="score-badge completed" style="background:rgba(239,113,113,0.08);border-color:rgba(239,113,113,0.2);color:#f87171 !important;">{int(next_m["home_score"])} — {int(next_m["away_score"])}</span></div>'
                pass
            else:
                badge_html = f'<span class="round-badge" style="background:rgba(52,211,153,0.1);color:#34d399;border:1px solid rgba(52,211,153,0.25);">&#128281; {next_m["round"].upper()}</span>'
                vs_html = '<div class="vs-separator">VS</div>'

            # Parse scorers
            scorers_html = ""
            if is_live_match and "scorers" in next_m and next_m["scorers"]:
                try:
                    import json
                    scorers_dict = json.loads(next_m["scorers"]) if isinstance(next_m["scorers"], str) else next_m["scorers"]
                    if scorers_dict:
                        h_scorers = ", ".join(scorers_dict.get(next_m["home_team"], []))
                        a_scorers = ", ".join(scorers_dict.get(next_m["away_team"], []))
                        scorers_html = f'<div style="display:flex;justify-content:space-between;font-size:0.75rem;color:var(--text-secondary);margin-top:12px;border-top:1px dashed var(--border-subtle);padding-top:8px;gap:8px;"><div style="text-align:left;max-width:45%;font-weight:600;">{h_scorers}</div><div style="text-align:center;font-weight:700;">&#9917;</div><div style="text-align:right;max-width:45%;font-weight:600;">{a_scorers}</div></div>'
                except Exception:
                    pass

            home_name = format_team_html(next_m['home_team'])
            away_name = format_team_html(next_m['away_team'])
            home_rank = get_ranking(next_m['home_team'])
            away_rank = get_ranking(next_m['away_team'])
            winner_html = format_team_html(winner)
            st.markdown(
                f'<div class="glass-card featured">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">'
                f'{badge_html}'
                f'<span style="color:var(--text-secondary);font-size:0.72rem;">&#128197; {next_m["date"]}</span>'
                f'</div>'
                f'<div class="team-vs-block">'
                f'<div class="team-block"><div class="team-name">{home_name}</div><div class="team-rank">Rank #{home_rank}</div></div>'
                f'{vs_html}'
                f'<div class="team-block"><div class="team-name">{away_name}</div><div class="team-rank">Rank #{away_rank}</div></div>'
                f'</div>'
                f'{scorers_html}'
                f'<div style="text-align:center;padding:12px 0;border-top:1px solid var(--border-subtle);margin-top:8px;">'
                f'<div style="font-size:0.65rem;color:var(--text-secondary);text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">AI PREDICTED WINNER</div>'
                f'<div class="winner-tag">{winner_html}</div>'
                f'<span class="conf-badge {conf_cls}">{conf_emoji} {conf_label} Confidence &middot; {pred["confidence"]:.1%}</span>'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True)

            # Probability bars
            probs = [(next_m["home_team"], pred["home_win_prob"], "home"),
                     ("Draw", pred["draw_prob"] if not pred["is_knockout"] else 0, "draw"),
                     (next_m["away_team"], pred["away_win_prob"], "away")]

            for team, prob, cls in probs:
                if prob and prob > 0:
                    st.markdown(textwrap.dedent(f"""
                    <div class="prob-meter-container">
                        <div class="prob-meter-header">
                            <span class="prob-meter-label">{format_team_html(team)}</span>
                            <span class="prob-meter-value" style="color:{'#34d399' if cls=='home' else '#64748b' if cls=='draw' else '#f87171'}">{prob:.1%}</span>
                        </div>
                        <div class="prob-meter-track">
                            <div class="prob-meter-fill {cls}" style="width:{prob*100:.1f}%"></div>
                        </div>
                    </div>"""), unsafe_allow_html=True)

            # One-line verdict
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(textwrap.dedent(f"""
            <div class="evidence-card">
                <div class="evidence-label">🧠 AI VERDICT</div>
                <div class="evidence-text" style="font-size:0.78rem;">{pred.get('verdict', '')}</div>
            </div>"""), unsafe_allow_html=True)

# ── Tournament Roadmap ─────────────────────────────────────────────────────────
st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🏆 Tournament Roadmap</div>', unsafe_allow_html=True)

rounds = [
    ("Group Stage", "Jun 11–26", "DONE", "#4ade80"),
    ("Round of 32", "Jun 27–Jul 4", "LIVE", "#10b981"),
    ("Round of 16", "Jul 6–9", "UPCOMING", "#f59e0b"),
    ("Quarterfinals", "Jul 11–12", "UPCOMING", "#f97316"),
    ("Semifinals", "Jul 15–16", "UPCOMING", "#ec4899"),
    ("3rd Place", "Jul 18", "UPCOMING", "#a78bfa"),
    ("Final", "Jul 19", "UPCOMING", "#facc15"),
]

cols = st.columns(len(rounds))
for col, (name, dates, status, color) in zip(cols, rounds):
    is_live = status == "LIVE"
    is_done = status == "DONE"
    opacity = "1" if status in ["DONE", "LIVE"] else "0.45"
    glow = f"box-shadow:0 0 24px {color}33;" if is_live else ""
    border = f"2px solid {color}" if is_live else f"1px solid rgba(0,0,0,0.07)"
    with col:
        st.markdown(textwrap.dedent(f"""
        <div style="
            background:linear-gradient(135deg, rgba(17,24,39,0.9), rgba(31,41,55,0.8));
            border:{border}; border-radius:12px; padding:12px 8px;
            text-align:center; opacity:{opacity}; {glow} transition:all 0.3s;
        ">
            <div style="font-size:1.3rem;margin-bottom:4px;">
                {'✅' if is_done else '🔴' if is_live else '⏳'}
            </div>
            <div style="font-size:0.75rem;font-weight:700;color:{color};">{name}</div>
            <div style="font-size:0.65rem;color:#6b7280;margin-top:3px;">{dates}</div>
            <div style="font-size:0.6rem;font-weight:700;color:{color if is_live or is_done else '#4b5563'};
                letter-spacing:1px;margin-top:6px;">{status}</div>
        </div>"""), unsafe_allow_html=True)

# ── Completed Results Scoreboard ───────────────────────────────────────────────
if model_ready:
    completed_home = fixtures[fixtures["status"] == "completed"].copy()
    if not completed_home.empty:
        st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:inline-flex; align-items:center; gap:10px; margin-bottom:16px;">
            <div class="section-header" style="margin:0;">📋 Match Results</div>
            <span class="live-badge"><span class="live-dot"></span>AUTO-REFRESH</span>
        </div>""", unsafe_allow_html=True)

        def _flag_img(t):
            """Return an <img> flag for team t using the TEAM_CODES already in get_team_flag."""
            flag = get_team_flag(t, as_emoji=False)
            return flag

        recent = completed_home.tail(10).iloc[::-1]
        for _, cm in recent.iterrows():
            hs = int(cm.get("home_score") or 0)
            as_ = int(cm.get("away_score") or 0)
            
            hp = cm.get("home_penalty_score")
            ap = cm.get("away_penalty_score")
            import pandas as pd
            has_pens = pd.notna(hp) and pd.notna(ap)
            
            if has_pens and int(hp) != int(ap):
                h_w = int(hp) > int(ap)
                a_w = int(ap) > int(hp)
            else:
                h_w = hs > as_
                a_w = as_ > hs
                
            h_style = "color:#34d399; font-weight:800;" if h_w else ("color:#4b5563;" if a_w else "color:#f3f4f6;")
            a_style = "color:#34d399; font-weight:800;" if a_w else ("color:#4b5563;" if h_w else "color:#f3f4f6;")
            
            if has_pens and int(hp) != int(ap):
                h_result = '<span class="result-win">WIN (P)</span>' if h_w else '<span class="result-loss">LOSS (P)</span>'
                score_display = f"{hs} — {as_}<br><span style='font-size:0.65rem; font-weight:normal; opacity:0.85;'>({int(hp)}-{int(ap)} pens)</span>"
            else:
                h_result = '<span class="result-win">WIN</span>' if h_w else ('<span class="result-loss">LOSS</span>' if a_w else '<span class="result-draw">DRAW</span>')
                score_display = f"{hs} — {as_}"

            # AI prediction badge for this completed match
            try:
                from ml.prediction_log import get_outcome_badge_html, log_prediction
                from ml.predict import predict_match
                wc_df_ref, _ = load_data()
                pred_c = predict_match(
                    home_team=cm["home_team"],
                    away_team=cm["away_team"],
                    round_name=cm.get("round", "Group Stage"),
                    wc_df=wc_df_ref,
                )
                log_prediction(
                    match_id=int(cm["match_id"]),
                    home_team=cm["home_team"],
                    away_team=cm["away_team"],
                    round_name=cm.get("round", "Group Stage"),
                    predicted_outcome=pred_c["predicted_outcome"],
                    confidence=pred_c["confidence"] or 0,
                    home_win_prob=pred_c["home_win_prob"] or 0,
                    draw_prob=pred_c["draw_prob"] or 0,
                    away_win_prob=pred_c["away_win_prob"] or 0,
                    match_date=str(cm.get("date", "")),
                )
                ai_badge_html = get_outcome_badge_html(int(cm["match_id"]), cm["home_team"], cm["away_team"])
            except Exception:
                ai_badge_html = ""

            date_label = format_match_date(str(cm.get("date", "")))
            date_cls = "today" if date_label == "Today" else ("yesterday" if date_label == "Yesterday" else "")

            card_html = textwrap.dedent(f"""
            <div class="completed-match-card">
                <div class="completed-match-header">
                    <div class="completed-match-header-left">
                        <span class="date-chip {date_cls}">{date_label}</span>
                        <span style="opacity:0.3; margin:0 4px;">·</span>
                        <b style="color:#10b981;">{cm.get('round', 'Round of 32') if cm.get('round') else cm.get('group', 'GS')}</b>
                    </div>
                    <div class="completed-match-header-right">
                        {ai_badge_html}
                    </div>
                </div>
                <div class="completed-match-body">
                    <div class="completed-match-team home">
                        <span style="{h_style}">{cm['home_team']}</span>
                        {_flag_img(cm['home_team'])}
                    </div>
                    <div class="completed-match-score-sec">
                        <span class="score-badge completed">{score_display}</span>
                        <div style="margin-top:2px;">{h_result}</div>
                    </div>
                    <div class="completed-match-team away">
                        {_flag_img(cm['away_team'])}
                        <span style="{a_style}">{cm['away_team']}</span>
                    </div>
                </div>
            </div>""").replace('\n', ' ')
            st.markdown(card_html, unsafe_allow_html=True)

# ── Feature Importance ─────────────────────────────────────────────────────────
if model_ready:
    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🤖 What Drives the Model?</div>', unsafe_allow_html=True)

    # Feature category mapping
    CATEGORIES = {
        "FIFA Rankings": ["FIFA Rank Diff", "FIFA Rank Ratio", "Home FIFA Rank", "Away FIFA Rank", "FIFA Rank Diff (Abs)"],
        "Team Form & Rest": ["Win Rate Diff", "Home Win Rate", "Away Win Rate", "Home Rest Days", "Away Rest Days"],
        "Attack & Defense": ["Goals Scored Diff", "Goals Conceded Diff", "Home Goals/Match", "Away Goals/Match", "Home Conceded/Match", "Away Conceded/Match"],
        "Head-to-Head": ["H2H Home Wins", "H2H Away Wins", "H2H Draws", "Has H2H History"],
        "Match Context": ["Is Knockout Stage", "Host Advantage", "Same Confederation"],
    }

    def get_feature_category(label: str) -> str:
        if label.startswith("Region:"):
            return "Regional Matchup Bias"
        for cat, labels in CATEGORIES.items():
            if label in labels:
                return cat
        return "Other Factors"

    c_imp, c_meta = st.columns([3, 2], gap="large")

    with c_imp:
        st.markdown("""
        <div class="glass-card" style="margin-bottom:16px; border-left:4px solid #10b981; background:rgba(16,185,129,0.01); padding:16px; border-radius:8px;">
            <div style="font-weight:700; color:#047857; font-size:0.95rem; margin-bottom:6px;">📊 How Team Stats Affect Match Predictions</div>
            <div style="font-size:0.8rem; color:#475569; line-height:1.5;">
                This chart displays the <b>relative predictive weight</b> that the AI model assigns to different performance metrics. 
                Larger bars represent factors that have a greater mathematical impact on the calculated win, draw, and loss probabilities. 
                Regional Matchup Bias terms represent baseline historical advantages between different confederations in World Cup history.
            </div>
        </div>
        """, unsafe_allow_html=True)

        fi = model_metrics.get("feature_importance", {})
        if fi:
            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"])
            fi_df["Label"] = fi_df["Feature"].apply(map_feature_name)
            fi_df["Category"] = fi_df["Label"].apply(get_feature_category)
            
            # Select top 10 and sort
            fi_df = fi_df.head(10).sort_values("Importance")
            
            import plotly.express as px
            color_map = {
                "FIFA Rankings": "#10b981",        # Emerald
                "Team Form & Rest": "#3b82f6",      # Blue
                "Attack & Defense": "#f59e0b",      # Amber
                "Head-to-Head": "#8b5cf6",          # Purple
                "Regional Matchup Bias": "#ec4899",  # Pink
                "Match Context": "#06b6d4",         # Cyan
                "Other Factors": "#64748b"          # Slate
            }
            
            fig = px.bar(
                fi_df,
                x="Importance",
                y="Label",
                color="Category",
                orientation="h",
                color_discrete_map=color_map,
                title="Top 10 Factors Influencing Predictions",
                labels={"Importance": "AI Weight (Relative Impact)", "Label": "Performance Metric"},
            )
            
            fig.update_traces(
                hovertemplate="<b>%{y}</b><br>Category: %{customdata[0]}<br>Relative Impact: %{x:.4f}<extra></extra>"
            )
            
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f3f4f6", size=11, family="Inter, sans-serif"),
                title_font=dict(color="#f3f4f6", size=14, weight="bold"),
                height=360,
                margin=dict(l=10, r=20, t=50, b=10),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#9ca3af"), zeroline=False),
                yaxis=dict(showgrid=False, tickfont=dict(color="#f3f4f6")),
                legend=dict(
                    title=dict(text="Stat Category", font=dict(size=10, weight="bold")),
                    font=dict(size=9, color="#f3f4f6"),
                    yanchor="bottom",
                    y=0.01,
                    xanchor="right",
                    x=0.99,
                    bgcolor="rgba(17,24,39,0.9)",
                    bordercolor="rgba(255,255,255,0.08)",
                    borderwidth=1
                )
            )
            st.plotly_chart(fig, use_container_width=True)

    with c_meta:
        st.markdown(textwrap.dedent(f"""
        <div class="glass-card">
            <div style="font-size:0.8rem;font-weight:700;color:#34d399;margin-bottom:16px;letter-spacing:1px;">
                MODEL PERFORMANCE SUMMARY
            </div>
            <div style="display:grid;gap:14px;">
                <div>
                    <div style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px;">Algorithm</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#f3f4f6;margin-top:3px;">
                        {model_metrics.get('algorithm','').replace('_',' ').title()}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px;">Test Accuracy</div>
                    <div style="font-size:1.4rem;font-weight:800;color:#34d399;margin-top:3px;">
                        {model_metrics.get('accuracy',0):.1%}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px;">5-Fold CV Accuracy</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#f3f4f6;margin-top:3px;">
                        {model_metrics.get('cv_mean_accuracy',0):.1%} ± {model_metrics.get('cv_std_accuracy',0):.1%}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px;">Log Loss</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#f3f4f6;margin-top:3px;">
                        {model_metrics.get('log_loss',0):.4f}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.65rem;color:#6b7280;text-transform:uppercase;letter-spacing:1px;">Training Samples</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#f3f4f6;margin-top:3px;">
                        {model_metrics.get('training_samples',0):,} WC matches
                    </div>
                </div>
            </div>
        </div>"""), unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
st.markdown("""<div style="text-align:center;color:#6b7280;font-size:0.72rem;padding:12px;">
    ⚽ FIFA World Cup 2026 AI Predictor &nbsp;|&nbsp; Built with Python · scikit-learn · Streamlit · Plotly
    <br>Not affiliated with FIFA. For educational and portfolio purposes only.
</div>""", unsafe_allow_html=True)
