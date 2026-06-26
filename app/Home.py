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

from app.utils import load_data, get_model_and_metrics, get_team_flag, CSS, ROUND_COLORS

st.set_page_config(
    page_title="FIFA WC 2026 AI Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(CSS, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">⚽ WC 2026</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">AI MATCH PREDICTION ENGINE</div>', unsafe_allow_html=True)
    st.divider()
    st.markdown("**Navigation**")
    for page, icon in [
        ("Home", "🏠"), ("Upcoming Matches", "📅"), ("Match Detail", "🔍"),
        ("Accuracy Tracker", "📊"), ("Tournament Bracket", "🏆")
    ]:
        st.markdown(f'<div class="sidebar-nav-item">{icon} {page}</div>', unsafe_allow_html=True)
    st.divider()
    st.caption("📊 Training data: 49,477 matches (1872–2026)")
    st.caption("🤖 Model: Logistic Regression + CalibratedCV")
    st.caption("🎯 Cross-val accuracy: 62.91%")

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

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{total_wc:,}</div>
                <div class="kpi-label">WC Matches Trained</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{upcoming_count}</div>
                <div class="kpi-label">Games to Predict</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{cv_acc:.1%}</div>
                <div class="kpi-label">CV Accuracy</div>
                <div class="kpi-delta positive">↑ vs 50.6% baseline</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">48</div>
                <div class="kpi-label">Teams · 2026</div>
            </div>""", unsafe_allow_html=True)

# ── Next Match Preview ─────────────────────────────────────────────────────────
with col_match:
    if model_ready:
        upcoming_df = fixtures[
            (fixtures["status"] == "scheduled") &
            (fixtures["home_team"] != "TBD") &
            (fixtures["away_team"] != "TBD")
        ]
        if not upcoming_df.empty:
            from ml.predict import predict_match_with_evidence, get_winner_label, get_confidence_label
            next_m = upcoming_df.iloc[0]

            with st.spinner("Running deep prediction..."):
                pred = predict_match_with_evidence(
                    home_team=next_m["home_team"],
                    away_team=next_m["away_team"],
                    round_name=next_m["round"],
                    wc_df=wc_df,
                )

            winner = get_winner_label(pred["predicted_outcome"], next_m["home_team"], next_m["away_team"])
            conf_label, conf_emoji = get_confidence_label(pred["confidence"])
            conf_cls = "conf-high" if conf_label == "High" else ("conf-medium" if conf_label == "Medium" else "conf-low")
            round_color = ROUND_COLORS.get(next_m["round"], "#60a5fa")

            st.markdown(f"""
            <div class="glass-card featured">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
                    <span class="round-badge" style="background:rgba(0,212,255,0.1);color:#00D4FF;border:1px solid rgba(0,212,255,0.25);">
                        🔜 {next_m['round'].upper()}
                    </span>
                    <span style="color:#475569;font-size:0.72rem;">📅 {next_m['date']}</span>
                </div>
                
                <div class="team-vs-block">
                    <div class="team-block">
                        <span class="team-flag">{get_team_flag(next_m['home_team'])}</span>
                        <div class="team-name">{next_m['home_team']}</div>
                        <div class="team-rank">#{pred.get('contributions', [{}])[0].get('strength','')}</div>
                    </div>
                    <div class="vs-separator">VS</div>
                    <div class="team-block">
                        <span class="team-flag">{get_team_flag(next_m['away_team'])}</span>
                        <div class="team-name">{next_m['away_team']}</div>
                    </div>
                </div>
                
                <div style="text-align:center;padding:12px 0;border-top:1px solid rgba(255,255,255,0.05);margin-top:8px;">
                    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">AI PREDICTED WINNER</div>
                    <div class="winner-tag">{get_team_flag(winner)} {winner}</div>
                    <span class="conf-badge {conf_cls}">{conf_emoji} {conf_label} Confidence · {pred['confidence']:.1%}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            # Probability bars
            probs = [(next_m["home_team"], pred["home_win_prob"], "home"),
                     ("Draw", pred["draw_prob"] if not pred["is_knockout"] else 0, "draw"),
                     (next_m["away_team"], pred["away_win_prob"], "away")]

            for team, prob, cls in probs:
                if prob and prob > 0:
                    st.markdown(f"""
                    <div class="prob-meter-container">
                        <div class="prob-meter-header">
                            <span class="prob-meter-label">{get_team_flag(team)} {team}</span>
                            <span class="prob-meter-value" style="color:{'#00D4FF' if cls=='home' else '#94a3b8' if cls=='draw' else '#FF3CAC'}">{prob:.1%}</span>
                        </div>
                        <div class="prob-meter-track">
                            <div class="prob-meter-fill {cls}" style="width:{prob*100:.1f}%"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            # One-line verdict
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="evidence-card">
                <div class="evidence-label">🧠 AI VERDICT</div>
                <div class="evidence-text" style="font-size:0.78rem;">{pred.get('verdict', '')}</div>
            </div>""", unsafe_allow_html=True)

# ── Tournament Roadmap ─────────────────────────────────────────────────────────
st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
st.markdown('<div class="section-header">🏆 Tournament Roadmap</div>', unsafe_allow_html=True)

rounds = [
    ("Group Stage", "Jun 11–26", "DONE", "#4ade80"),
    ("Round of 32", "Jun 27–Jul 4", "LIVE", "#00D4FF"),
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
    border = f"2px solid {color}" if is_live else f"1px solid rgba(255,255,255,0.07)"
    with col:
        st.markdown(f"""
        <div style="
            background:linear-gradient(135deg,#0d1117,#070914);
            border:{border}; border-radius:12px; padding:12px 8px;
            text-align:center; opacity:{opacity}; {glow} transition:all 0.3s;
        ">
            <div style="font-size:1.3rem;margin-bottom:4px;">
                {'✅' if is_done else '🔴' if is_live else '⏳'}
            </div>
            <div style="font-size:0.75rem;font-weight:700;color:{color};">{name}</div>
            <div style="font-size:0.65rem;color:#475569;margin-top:3px;">{dates}</div>
            <div style="font-size:0.6rem;font-weight:700;color:{color if is_live or is_done else '#374151'};
                letter-spacing:1px;margin-top:6px;">{status}</div>
        </div>""", unsafe_allow_html=True)

# ── Feature Importance ─────────────────────────────────────────────────────────
if model_ready:
    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🤖 What Drives the Model?</div>', unsafe_allow_html=True)

    c_imp, c_meta = st.columns([3, 2], gap="large")

    with c_imp:
        fi = model_metrics.get("feature_importance", {})
        if fi:
            fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).head(10).sort_values("Importance")
            fig = go.Figure(go.Bar(
                x=fi_df["Importance"], y=fi_df["Feature"],
                orientation="h",
                marker=dict(
                    color=fi_df["Importance"],
                    colorscale=[[0, "#7B2FFF"], [1, "#00D4FF"]],
                    line=dict(color="rgba(0,0,0,0)", width=0)
                ),
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", size=11, family="Space Grotesk"),
                title=dict(text="Top 10 Predictive Features", font=dict(color="#00D4FF", size=14)),
                coloraxis_showscale=False, height=320,
                margin=dict(l=10, r=20, t=50, b=10),
                xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", tickfont=dict(color="#475569")),
                yaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8")),
            )
            st.plotly_chart(fig, use_container_width=True)

    with c_meta:
        st.markdown(f"""
        <div class="glass-card">
            <div style="font-size:0.8rem;font-weight:700;color:#00D4FF;margin-bottom:16px;letter-spacing:1px;">
                MODEL PERFORMANCE SUMMARY
            </div>
            <div style="display:grid;gap:14px;">
                <div>
                    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:1px;">Algorithm</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;margin-top:3px;">
                        {model_metrics.get('algorithm','').replace('_',' ').title()}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:1px;">Test Accuracy</div>
                    <div style="font-size:1.4rem;font-weight:800;color:#10b981;margin-top:3px;">
                        {model_metrics.get('accuracy',0):.1%}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:1px;">5-Fold CV Accuracy</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;margin-top:3px;">
                        {model_metrics.get('cv_mean_accuracy',0):.1%} ± {model_metrics.get('cv_std_accuracy',0):.1%}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:1px;">Log Loss</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;margin-top:3px;">
                        {model_metrics.get('log_loss',0):.4f}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;letter-spacing:1px;">Training Samples</div>
                    <div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;margin-top:3px;">
                        {model_metrics.get('training_samples',0):,} WC matches
                    </div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;color:#1e293b;font-size:0.72rem;padding:12px;">
    ⚽ FIFA World Cup 2026 AI Predictor &nbsp;|&nbsp; Built with Python · scikit-learn · Streamlit · Plotly
    <br>Not affiliated with FIFA. For educational and portfolio purposes only.
</div>""", unsafe_allow_html=True)
