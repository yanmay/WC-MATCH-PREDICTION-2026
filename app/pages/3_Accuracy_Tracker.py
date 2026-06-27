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

from app.utils import load_data, get_model_and_metrics, get_team_flag, format_team_emoji, CSS, ROUND_COLORS, render_sidebar, infer_wc_round
from ml.predict import predict_match, get_winner_label, get_confidence_label
from ml.evaluate import compute_accuracy_stats, baseline_accuracy, compute_brier_score

st.set_page_config(
    page_title="Model Accuracy Tracker | FIFA WC 2026 Predictor",
    page_icon="📊",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)
render_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-badge">📊 MODEL CALIBRATION & ACCURACY</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title" style="font-size:2.5rem; line-height:1.2;">Model Accuracy Tracker</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Evaluate the predictive power of our AI model through live results and historical backtests.</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading performance metrics and historical archives..."):
    wc_df, fixtures = load_data()
    pipeline, model_metrics = get_model_and_metrics()

# -- 2026 Live Prediction Accuracy Section ----------------------------------------
completed_2026 = fixtures[fixtures["status"] == "completed"].copy()

st.markdown("### \U0001f3df\ufe0f 2026 World Cup -- AI Prediction Accuracy")

try:
    from ml.prediction_log import get_live_accuracy_stats, log_prediction, get_outcome_badge_html

    # Ensure all completed matches are logged
    for _, match in completed_2026.iterrows():
        pred_c = predict_match(
            home_team=match["home_team"],
            away_team=match["away_team"],
            round_name=match.get("round", "Group Stage"),
            wc_df=wc_df,
        )
        log_prediction(
            match_id=int(match["match_id"]),
            home_team=match["home_team"],
            away_team=match["away_team"],
            round_name=match.get("round", "Group Stage"),
            predicted_outcome=pred_c["predicted_outcome"],
            confidence=pred_c["confidence"] or 0,
            home_win_prob=pred_c["home_win_prob"] or 0,
            draw_prob=pred_c["draw_prob"] or 0,
            away_win_prob=pred_c["away_win_prob"] or 0,
            match_date=str(match.get("date", "")),
        )

    plog_stats = get_live_accuracy_stats()
    resolved  = plog_stats["resolved"]
    correct   = plog_stats["correct"]
    wrong     = plog_stats["wrong"]
    pending   = plog_stats["pending"]
    accuracy  = plog_stats["accuracy"]

    if resolved == 0 and pending == 0:
        st.markdown("""
        <div class="glass-card" style="border-left: 3px solid #f59e0b; background: rgba(245,158,11,0.02);">
            <div style="font-weight: 700; color: #f59e0b; font-size: 0.95rem; margin-bottom: 4px;">No Predictions Logged Yet</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">Visit the <b>Upcoming Matches</b> page to generate predictions -- they are automatically logged here.</div>
        </div>""", unsafe_allow_html=True)
    else:
        c_e1, c_e2, c_e3, c_e4 = st.columns(4)
        with c_e1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{resolved}</div>
                <div class="kpi-label">Results Tracked</div>
            </div>""", unsafe_allow_html=True)
        with c_e2:
            acc_display = f"{accuracy:.1%}" if accuracy is not None else "--"
            st.markdown(f"""
            <div class="kpi-card kpi-accuracy-2026">
                <div class="kpi-value">{acc_display}</div>
                <div class="kpi-label">\U0001f3af 2026 AI Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with c_e3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:#34d399;">{correct}</div>
                <div class="kpi-label">Correct Predictions</div>
            </div>""", unsafe_allow_html=True)
        with c_e4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:#f87171;">{wrong}</div>
                <div class="kpi-label">Wrong Predictions</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Cumulative accuracy trend chart
        timeline = plog_stats["timeline"]
        if len(timeline) >= 2:
            import plotly.graph_objects as pgo
            tl_df = pd.DataFrame(timeline)
            fig_trend = pgo.Figure()
            fig_trend.add_trace(pgo.Scatter(
                x=tl_df["match_num"],
                y=tl_df["cumulative_accuracy"],
                mode="lines+markers",
                name="Cumulative Accuracy",
                line=dict(color="#34d399", width=2.5),
                marker=dict(
                    size=9,
                    color=["#34d399" if c else "#f87171" for c in tl_df["is_correct"]],
                    symbol="circle",
                    line=dict(width=1.5, color="#111827"),
                ),
                hovertemplate="Match %{x}: %{customdata}<br>Accuracy: %{y:.1%}<extra></extra>",
                customdata=tl_df["match"],
            ))
            fig_trend.add_hline(
                y=0.5, line_dash="dot", line_color="#6b7280", line_width=1,
                annotation_text="50% baseline", annotation_position="bottom right",
                annotation_font_color="#6b7280",
            )
            fig_trend.update_layout(
                title="\U0001f4c8 Cumulative Prediction Accuracy -- 2026 WC Matches",
                title_font=dict(color="#f3f4f6", size=14, weight="bold"),
                xaxis=dict(
                    title="Match Number",
                    showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(color="#9ca3af"), zeroline=False,
                ),
                yaxis=dict(
                    title="Cumulative Accuracy", tickformat=".0%", range=[0, 1],
                    showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                    tickfont=dict(color="#9ca3af"), zeroline=False,
                ),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#f3f4f6", family="Inter, sans-serif"),
                height=300, margin=dict(l=20, r=20, t=50, b=30), showlegend=False,
            )
            st.plotly_chart(fig_trend, use_container_width=True)

        # Detailed prediction log table
        st.markdown("##### \U0001f4dd 2026 AI Prediction Log")
        log_entries = plog_stats["entries"]
        table_rows = []
        for e in sorted(log_entries, key=lambda x: x.get("date", ""), reverse=True):
            is_correct = e.get("is_correct")
            actual = e.get("actual_outcome")
            predicted = e.get("predicted_outcome", "")

            def _label(outcome, home, away):
                if outcome == "home_win": return home
                if outcome == "away_win": return away
                if outcome == "draw": return "Draw"
                return str(outcome)

            home = e["home_team"]
            away = e["away_team"]

            result_icon = "\u2705 Correct" if is_correct is True else ("\u274c Wrong" if is_correct is False else "\u23f3 Pending")
            table_rows.append({
                "Date": e.get("date", ""),
                "Round": e.get("round", ""),
                "Match": f"{format_team_emoji(home)} vs {format_team_emoji(away)}",
                "AI Predicted": format_team_emoji(_label(predicted, home, away)),
                "Actual Result": format_team_emoji(_label(actual, home, away)) if actual else "\u23f3",
                "Confidence": f"{e.get('confidence', 0):.1%}",
                "Outcome": result_icon,
            })

        if table_rows:
            df_plog = pd.DataFrame(table_rows)
            st.dataframe(df_plog, use_container_width=True, hide_index=True)

except Exception as _err:
    st.warning(f"Prediction log unavailable: {_err}")
    if completed_2026.empty:
        st.markdown("""
        <div class="glass-card" style="border-left: 3px solid #f59e0b; background: rgba(245,158,11,0.02);">
            <div style="font-weight: 700; color: #f59e0b;">No Live Matches Finished Yet</div>
            <div style="font-size: 0.8rem; color: #94a3b8;">The 2026 World Cup has not started yet. Backtesting results are shown below.</div>
        </div>""", unsafe_allow_html=True)


# ── Backtesting Section (Qatar 2022) ──────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🏆 Backtesting Case Study: Qatar 2022 World Cup")
st.markdown('<p style="font-size:0.85rem; color:#94a3b8; margin-top:-10px;">To validate model reliability without data leakage, we backtested the entire 64 matches of Qatar 2022. For each match, the model only trained on data strictly preceding the match date.</p>', unsafe_allow_html=True)

with st.spinner("Executing historical simulation for Qatar 2022 matches..."):
    # Filter historical matches to year 2022 and sort chronologically
    qatar_matches = wc_df[(wc_df["year"] == 2022) & (wc_df["tournament"] == "FIFA World Cup")].copy().sort_values("date").reset_index(drop=True)
    
    backtest_log = []
    for idx, match in qatar_matches.iterrows():
        # Predict using prior data
        prior_data = wc_df[wc_df["date"] < match["date"]]
        
        round_name = infer_wc_round(idx, len(qatar_matches))
        
        pred = predict_match(
            home_team=match["home_team"],
            away_team=match["away_team"],
            round_name=round_name,
            wc_df=prior_data,
        )
        
        actual = match["outcome"]
        brier = compute_brier_score(actual, pred["home_win_prob"], pred["draw_prob"], pred["away_win_prob"])
        
        backtest_log.append({
            "Date": match["date"].strftime("%Y-%m-%d"),
            "Round": round_name,
            "Home Team": match["home_team"],
            "Away Team": match["away_team"],
            "Score": f"{int(match['home_score'])} - {int(match['away_score'])}",
            "Predicted": format_team_emoji(get_winner_label(pred['predicted_outcome'], match['home_team'], match['away_team'])),
            "Actual": format_team_emoji(get_winner_label(actual, match['home_team'], match['away_team'])),
            "confidence": pred["confidence"],
            "predicted_outcome": pred["predicted_outcome"],
            "actual_outcome": actual,
            "round": round_name,
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
    round_order = ["Group Stage", "Round of 16", "Quarterfinal", "Semifinal", "3rd Place", "Final"]
    df_r_acc["Round"] = pd.Categorical(df_r_acc["Round"], categories=round_order, ordered=True)
    df_r_acc = df_r_acc.sort_values("Round")
    
    fig_round = px.bar(
        df_r_acc, x="Round", y="Accuracy",
        color="Accuracy", text="Matches",
        color_continuous_scale=[[0, "#a7f3d0"], [1, "#10b981"]],
        title="Accuracy by Tournament Stage (Labels = Match Count)",
        hover_data={"Accuracy": ":.1%", "Matches": True}
    )
    fig_round.update_traces(
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.1%}<br>Matches Played: %{text}<extra></extra>"
    )
    fig_round.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f3f4f6", family="Inter, sans-serif"),
        title_font=dict(color="#f3f4f6", size=14, weight="bold"),
        coloraxis_showscale=False,
        yaxis=dict(tickformat=".0%", showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#9ca3af"), zeroline=False),
        xaxis=dict(showgrid=False, tickfont=dict(color="#f3f4f6")),
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
            line=dict(dash='dash', color='#6b7280', width=1.5),
            name='Ideal Calibration'
        ))
        
        # Model
        fig_cal.add_trace(go.Scatter(
            x=cal_df["predicted_confidence"],
            y=cal_df["actual_accuracy"],
            mode='markers+lines',
            marker=dict(size=8, color='#34d399', symbol="circle"),
            line=dict(color='#34d399', width=2),
            text=cal_df["confidence_bin"],
            name='Calibrated Estimator',
            hovertemplate="Confidence Bin: %{text}<br>Predicted Confidence: %{x:.1%}<br>Observed Accuracy: %{y:.1%}<extra></extra>"
        ))
        
        fig_cal.update_layout(
            title="Model Calibration Curve (Confidence vs Reality)",
            xaxis_title="Predicted Probability / Confidence",
            yaxis_title="Observed Accuracy",
            xaxis=dict(range=[0.3, 1.0], showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#9ca3af"), tickformat=".0%"),
            yaxis=dict(range=[0.0, 1.0], showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#9ca3af"), tickformat=".0%"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f3f4f6", family="Inter, sans-serif"),
            title_font=dict(color="#f3f4f6", size=14, weight="bold"),
            height=320,
            showlegend=True,
            legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99, bgcolor="rgba(17,24,39,0.9)", font=dict(color="#f3f4f6")),
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

df_table_show["Home Team"] = df_table_show["Home Team"].apply(format_team_emoji)
df_table_show["Away Team"] = df_table_show["Away Team"].apply(format_team_emoji)
df_table_show["confidence"] = df_table_show["confidence"].apply(lambda c: f"{c:.1%}")
df_table_show["Brier Score"] = df_table_show["Brier Score"].apply(lambda b: f"{b:.4f}")


df_table_show = df_table_show.rename(columns={
    "confidence": "AI Confidence",
    "Correct": "Is Correct"
})

st.dataframe(df_table_show, use_container_width=True, hide_index=True)
