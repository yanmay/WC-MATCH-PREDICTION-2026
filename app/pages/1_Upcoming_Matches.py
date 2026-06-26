"""
Upcoming Matches — FIFA World Cup 2026 AI Predictor | 21st.dev design language
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import load_data, get_model_and_metrics, get_team_flag, CSS, ROUND_COLORS, ROUND_ORDER
from ml.predict import predict_match_with_evidence, get_winner_label, get_confidence_label

st.set_page_config(
    page_title="Upcoming Matches | FIFA WC 2026 Predictor",
    page_icon="📅",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-badge">📅 FIXTURES & LIVE PREDICTIONS</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title" style="font-size:2.5rem; line-height:1.2;">Upcoming Matches</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Explore upcoming match ups with deep AI evidence, historical facts, and probability calibrations.</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Analyzing match statistics and generating predictions..."):
    wc_df, fixtures = load_data()
    pipeline, model_metrics = get_model_and_metrics()

# ── Filters ───────────────────────────────────────────────────────────────────
col_f1, col_f2 = st.columns([2, 3])
with col_f1:
    available_rounds = fixtures["round"].unique().tolist()
    selected_rounds = st.multiselect(
        "Filter by Round",
        options=available_rounds,
        default=available_rounds[:2],
        help="Select which tournament rounds to show"
    )
with col_f2:
    all_teams = sorted(set(
        fixtures["home_team"].tolist() + fixtures["away_team"].tolist()
    ) - {"TBD"})
    selected_team = st.selectbox("Filter by Team (optional)", ["All Teams"] + all_teams)

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
    st.info("No matches found matching the selected filters.")
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
            # Run prediction with complete evidence chain
            pred = predict_match_with_evidence(
                home_team=match["home_team"],
                away_team=match["away_team"],
                round_name=match["round"],
                wc_df=wc_df,
            )
            winner = get_winner_label(pred["predicted_outcome"], match["home_team"], match["away_team"])
            conf_label, conf_emoji = get_confidence_label(pred["confidence"])
            conf_cls = "conf-high" if conf_label == "High" else ("conf-medium" if conf_label == "Medium" else "conf-low")
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

        # Glass match card container
        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                <span style="color:#475569; font-size:0.75rem; font-weight:600;">📅 Match Date: {match['date']}</span>
                <span style="font-size:0.72rem; color:#94a3b8; font-weight:600;">ID: #{match['match_id']}</span>
            </div>
            
            <div class="team-vs-block" style="margin-bottom:16px;">
                <div class="team-block">
                    <span class="team-flag" style="font-size:3rem;">{get_team_flag(match['home_team'])}</span>
                    <div class="team-name" style="font-size:1.15rem; font-weight:800;">{match['home_team']}</div>
                </div>
                <div class="vs-separator" style="font-size:1.1rem; color:#475569;">VS</div>
                <div class="team-block">
                    <span class="team-flag" style="font-size:3rem;">{get_team_flag(match['away_team'])}</span>
                    <div class="team-name" style="font-size:1.15rem; font-weight:800;">{match['away_team']}</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        if not is_tbd:
            # Probability meter bars
            c_left, c_right = st.columns([3, 2], gap="large")
            with c_left:
                st.markdown('<div style="font-size:0.78rem; font-weight:700; color:#00D4FF; letter-spacing:1px; margin-bottom:8px;">PROBABILITY MODEL OUTCOMES</div>', unsafe_allow_html=True)
                hp = pred["home_win_prob"]
                dp = pred["draw_prob"] if not pred["is_knockout"] else 0
                ap = pred["away_win_prob"]

                # Home Prob
                st.markdown(f"""
                <div class="prob-meter-container">
                    <div class="prob-meter-header">
                        <span class="prob-meter-label">{get_team_flag(match['home_team'])} {match['home_team']} Win</span>
                        <span class="prob-meter-value" style="color:#00D4FF;">{hp:.1%}</span>
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
                            <span class="prob-meter-value" style="color:#94a3b8;">{dp:.1%}</span>
                        </div>
                        <div class="prob-meter-track">
                            <div class="prob-meter-fill draw" style="width:{dp*100:.1f}%"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                # Away Prob
                st.markdown(f"""
                <div class="prob-meter-container">
                    <div class="prob-meter-header">
                        <span class="prob-meter-label">{get_team_flag(match['away_team'])} {match['away_team']} Win</span>
                        <span class="prob-meter-value" style="color:#FF3CAC;">{ap:.1%}</span>
                    </div>
                    <div class="prob-meter-track">
                        <div class="prob-meter-fill away" style="width:{ap*100:.1f}%"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            with c_right:
                st.markdown(f"""
                <div style="text-align:center; padding:12px; background:rgba(0,0,0,0.2); border-radius:12px; border:1px solid rgba(255,255,255,0.03); height:100%; display:flex; flex-direction:column; justify-content:center;">
                    <div style="font-size:0.65rem; color:#475569; text-transform:uppercase; letter-spacing:1.5px; margin-bottom:4px;">AI PREDICTED WINNER</div>
                    <div class="winner-tag" style="margin:4px 0; font-size:1.4rem;">{get_team_flag(winner)} {winner}</div>
                    <div style="margin-top:6px;">
                        <span class="conf-badge {conf_cls}">{conf_emoji} {conf_label} Confidence ({pred['confidence']:.1%})</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Deep Analysis Evidence Block
            with st.expander("🔬 View AI Evidence Breakdown & Historical Proofs"):
                # Natural language verdict
                st.markdown(f"""
                <div class="evidence-card" style="margin-bottom:14px; background:rgba(0,212,255,0.06); border-color:rgba(0,212,255,0.25);">
                    <div class="evidence-label" style="font-size:0.75rem;">🤖 AI FORECAST VERDICT</div>
                    <div class="evidence-text" style="font-size:0.85rem; font-weight:500;">{pred['verdict']}</div>
                </div>""", unsafe_allow_html=True)

                col_ev, col_risk = st.columns(2, gap="medium")
                with col_ev:
                    st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#00D4FF; letter-spacing:1px; margin-bottom:6px;">📈 SUPPORTING EVIDENCE CHAIN</div>', unsafe_allow_html=True)
                    for item in pred["evidence"]:
                        st.markdown(f"""
                        <div class="evidence-card">
                            <div class="evidence-label">{item['label']}</div>
                            <div class="evidence-text" style="font-size:0.78rem;">{item['text']}</div>
                            <div class="evidence-sub">{item['sub']}</div>
                        </div>""", unsafe_allow_html=True)

                with col_risk:
                    st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#ef4444; letter-spacing:1px; margin-bottom:6px;">⚠️ POTENTIAL RISK FACTORS & UPSET TRIGGERS</div>', unsafe_allow_html=True)
                    for item in pred["risks"]:
                        st.markdown(f"""
                        <div class="risk-card">
                            <div class="risk-label">{item['label']}</div>
                            <div class="evidence-text" style="font-size:0.78rem;">{item['text']}</div>
                        </div>""", unsafe_allow_html=True)

                    # Form indicator sparkline summary
                    st.markdown('<div style="margin-top:12px; font-size:0.75rem; font-weight:700; color:#94a3b8;">FORM TRAJECTORY (LAST 5 WC MATCHES)</div>', unsafe_allow_html=True)
                    
                    def render_mini_form(form_list):
                        html = '<div style="display:flex; gap:6px; margin-top:4px;">'
                        for game in form_list:
                            color = "#10b981" if game["result"] == "W" else "#ef4444" if game["result"] == "L" else "#94a3b8"
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
                    <div style="background:rgba(255,255,255,0.02); padding:10px; border-radius:8px; border:1px solid rgba(255,255,255,0.03); margin-top:6px;">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                            <span style="font-size:0.72rem; color:#94a3b8;">{get_team_flag(match['home_team'])} {match['home_team']} Form:</span>
                            {render_mini_form(pred['home_form'])}
                        </div>
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <span style="font-size:0.72rem; color:#94a3b8;">{get_team_flag(match['away_team'])} {match['away_team']} Form:</span>
                            {render_mini_form(pred['away_form'])}
                        </div>
                    </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="padding:16px 0; text-align:center; background:rgba(255,255,255,0.02); border-radius:12px;">
                <div style="color:#6B7280; font-size:0.85rem;">🔒 Match Pending Team Qualification</div>
                <div style="color:#4B5563; font-size:0.75rem; margin-top:4px;">Deep predictions and probability distribution will unlock once teams are determined.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr style="border:none; border-top:1px solid rgba(255,255,255,0.05); margin:20px 0;">', unsafe_allow_html=True)

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
