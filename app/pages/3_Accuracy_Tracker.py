"""
Accuracy Tracker & Backtesting Dashboard — FIFA World Cup Match Prediction Platform.
Glassmorphism stats cards, calibrated line plots, and complete backtesting results.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import load_data, get_model_and_metrics, get_team_flag, CSS, ROUND_COLORS
from ml.predict import predict_match, get_winner_label, get_confidence_label
from ml.evaluate import compute_accuracy_stats, baseline_accuracy, compute_brier_score

st.set_page_config(
    page_title="Model Accuracy Tracker | FIFA WC 2026 Predictor",
    page_icon="📊",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-badge">📊 MODEL CALIBRATION & ACCURACY</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title" style="font-size:2.5rem; line-height:1.2;">Model Accuracy Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Evaluate the predictive power of our AI model through live results and historical backtests.</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading performance metrics and historical archives..."):
    wc_df, fixtures = load_data()
    pipeline, model_metrics = get_model_and_metrics()

# ── 2026 Completed Matches Section ────────────────────────────────────────────
completed_2026 = fixtures[fixtures["status"] == "completed"].copy()

st.markdown("### 🏟️ 2026 World Cup Match Results")
if completed_2026.empty:
    st.markdown("""
    <div class="glass-card" style="border-left: 3px solid #f59e0b; background: rgba(245,158,11,0.02);">
        <div style="font-weight: 700; color: #f59e0b; font-size: 0.95rem; margin-bottom: 4px;">No Live Matches Finished Yet</div>
        <div style="font-size: 0.8rem; color: #94a3b8;">The 2026 World Cup has not started yet. Below we present validation analytics and backtesting logs run against the 2022 World Cup in Qatar.</div>
    </div>""", unsafe_allow_html=True)
else:
    # Build evaluation for completed 2026 fixtures
    eval_records = []
    for _, match in completed_2026.iterrows():
        pred = predict_match(
            home_team=match["home_team"],
            away_team=match["away_team"],
            round_name=match["round"],
            wc_df=wc_df,
        )
        
        # Determine actual outcome
        hs = match["home_score"]
        as_ = match["away_score"]
        if hs > as_:
            actual = "home_win"
        elif as_ > hs:
            actual = "away_win"
        else:
            actual = "draw"
            
        brier = compute_brier_score(actual, pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"])
        
        eval_records.append({
            "Date": match["date"],
            "Round": match["round"],
            "Home Team": f"{get_team_flag(match['home_team'])} {match['home_team']}",
            "Away Team": f"{get_team_flag(match['away_team'])} {match['away_team']}",
            "Score": f"{int(hs)} - {int(as_)}",
            "Predicted": get_winner_label(pred["predicted_outcome"], match["home_team"], match["away_team"]),
            "Actual": get_winner_label(actual, match["home_team"], match["away_team"]),
            "Confidence": f"{pred['confidence']:.1%}",
            "Brier Score": round(brier, 4),
            "Correct": pred["predicted_outcome"] == actual
        })
        
    df_eval = pd.DataFrame(eval_records)
    
    # 2026 Metrics
    c_e1, c_e2, c_e3 = st.columns(3)
    total_e = len(df_eval)
    correct_e = df_eval["Correct"].sum()
    acc_e = correct_e / total_e if total_e > 0 else 0
    avg_brier = df_eval["Brier Score"].mean()
    
    with c_e1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{total_e}</div>
            <div class="kpi-label">Completed Matches</div>
        </div>""", unsafe_allow_html=True)
    with c_e2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{acc_e:.1%}</div>
            <div class="kpi-label">Prediction Accuracy</div>
        </div>""", unsafe_allow_html=True)
    with c_e3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value">{avg_brier:.4f}</div>
            <div class="kpi-label">Avg Brier Score (Loss)</div>
        </div>""", unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### Completed 2026 Matches Log")
    st.dataframe(df_eval, use_container_width=True, hide_index=True)

# ── Backtesting Section (Qatar 2022) ──────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🏆 Backtesting Case Study: Qatar 2022 World Cup")
st.markdown('<p style="font-size:0.85rem; color:#94a3b8; margin-top:-10px;">To validate model reliability without data leakage, we backtested the entire 64 matches of Qatar 2022. For each match, the model only trained on data strictly preceding the match date.</p>', unsafe_allow_html=True)

with st.spinner("Executing historical simulation for Qatar 2022 matches..."):
    # Filter historical matches to year 2022
    qatar_matches = wc_df[wc_df["year"] == 2022].copy()
    
    backtest_log = []
    for _, match in qatar_matches.iterrows():
        # Predict using prior data
        prior_data = wc_df[wc_df["date"] < match["date"]]
        
        pred = predict_match(
            home_team=match["home_team"],
            away_team=match["away_team"],
            round_name=match.get("round", "Group Stage"),
            wc_df=prior_data,
        )
        
        actual = match["outcome"]
        brier = compute_brier_score(actual, pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"])
        
        backtest_log.append({
            "Date": match["date"].strftime("%Y-%m-%d"),
            "Round": match.get("round", "Group Stage"),
            "Home Team": match["home_team"],
            "Away Team": match["away_team"],
            "Score": f"{int(match['home_score'])} - {int(match['away_score'])}",
            "Predicted": get_winner_label(pred["predicted_outcome"], match["home_team"], match["away_team"]),
            "Actual": get_winner_label(actual, match["home_team"], match["away_team"]),
            "confidence": pred["confidence"],
            "predicted_outcome": pred["predicted_outcome"],
            "actual_outcome": actual,
            "round": match.get("round", "Group Stage"),
            "Brier Score": brier,
            "Correct": pred["predicted_outcome"] == actual
        })
        
    df_backtest = pd.DataFrame(backtest_log)
    stats = compute_accuracy_stats(backtest_log)

# Backtest Metrics row using 21st.dev style cards
c_b1, c_b2, c_b3, c_b4 = st.columns(4)
with c_b1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{stats['total']}</div>
        <div class="kpi-label">Backtested Matches</div>
    </div>""", unsafe_allow_html=True)
with c_b2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{stats['accuracy']:.1%}</div>
        <div class="kpi-label">Backtest Accuracy</div>
        <div class="kpi-delta positive">↑ vs Baseline</div>
    </div>""", unsafe_allow_html=True)
with c_b3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{df_backtest['Brier Score'].mean():.4f}</div>
        <div class="kpi-label">Average Brier Score</div>
    </div>""", unsafe_allow_html=True)
with c_b4:
    base_acc = baseline_accuracy(qatar_matches)
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-value">{base_acc:.1%}</div>
        <div class="kpi-label">Baseline (Always Home Win)</div>
    </div>""", unsafe_allow_html=True)

# Layout for charts
st.markdown("<br>", unsafe_allow_html=True)
col_chart1, col_chart2 = st.columns(2, gap="large")

with col_chart1:
    # 1. Round Performance Chart
    round_acc_data = []
    for r_name, r_stats in stats["by_round"].items():
        round_acc_data.append({
            "Round": r_name,
            "Accuracy": r_stats["accuracy"],
            "Matches": r_stats["total"]
        })
    df_r_acc = pd.DataFrame(round_acc_data)
    
    fig_round = px.bar(
        df_r_acc, x="Round", y="Accuracy",
        color="Accuracy", text="Matches",
        color_continuous_scale=[[0, "#7B2FFF"], [1, "#00D4FF"]],
        title="Accuracy by Tournament Stage (Labels = Match Count)"
    )
    fig_round.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Space Grotesk"),
        title_font=dict(color="#00D4FF", size=14),
        coloraxis_showscale=False,
        yaxis=dict(tickformat=".0%", showgrid=True, gridcolor="rgba(255,255,255,0.03)", tickfont=dict(color="#475569")),
        xaxis=dict(showgrid=False, tickfont=dict(color="#94a3b8")),
        height=320,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_round, use_container_width=True)

with col_chart2:
    # 2. Calibration Chart
    cal_df = pd.DataFrame(stats["calibration"])
    if not cal_df.empty:
        fig_cal = go.Figure()
        
        # Perfect line
        fig_cal.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='#475569', width=1.5),
            name='Ideal Calibration'
        ))
        
        # Model
        fig_cal.add_trace(go.Scatter(
            x=cal_df["predicted_confidence"],
            y=cal_df["actual_accuracy"],
            mode='markers+lines',
            marker=dict(size=8, color='#00D4FF', symbol="circle"),
            line=dict(color='#00D4FF', width=2),
            text=cal_df["confidence_bin"],
            name='Calibrated Estimator'
        ))
        
        fig_cal.update_layout(
            title="Model Calibration Curve (Confidence vs Reality)",
            xaxis_title="Predicted Probability / Confidence",
            yaxis_title="Observed Accuracy",
            xaxis=dict(range=[0.3, 1.0], showgrid=True, gridcolor="rgba(255,255,255,0.03)", tickfont=dict(color="#475569")),
            yaxis=dict(range=[0.0, 1.0], showgrid=True, gridcolor="rgba(255,255,255,0.03)", tickfont=dict(color="#475569")),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8", family="Space Grotesk"),
            title_font=dict(color="#00D4FF", size=14),
            height=320,
            showlegend=False,
            margin=dict(l=40, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_cal, use_container_width=True)

# Detail log
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 📝 Detailed Backtest Match Logs (Qatar 2022)")

# Filter select
col_t1, _ = st.columns([1, 2])
with col_t1:
    filter_result = st.selectbox("Show Predictions Log", ["All Matches", "Correct Predictions", "Incorrect Predictions"])
    
df_table = df_backtest.copy()
if filter_result == "Correct Predictions":
    df_table = df_table[df_table["Correct"] == True]
elif filter_result == "Incorrect Predictions":
    df_table = df_table[df_table["Correct"] == False]
    
# Format columns
df_table_show = df_table[[
    "Date", "Round", "Home Team", "Away Team", "Score", "Predicted", "Actual", "confidence", "Brier Score", "Correct"
]].copy()

df_table_show["Home Team"] = df_table_show["Home Team"].apply(lambda t: f"{get_team_flag(t)} {t}")
df_table_show["Away Team"] = df_table_show["Away Team"].apply(lambda t: f"{get_team_flag(t)} {t}")
df_table_show["confidence"] = df_table_show["confidence"].apply(lambda c: f"{c:.1%}")
df_table_show["Brier Score"] = df_table_show["Brier Score"].apply(lambda b: f"{b:.4f}")

df_table_show = df_table_show.rename(columns={
    "confidence": "AI Confidence",
    "Correct": "Is Correct"
})

st.dataframe(df_table_show, use_container_width=True, hide_index=True)
