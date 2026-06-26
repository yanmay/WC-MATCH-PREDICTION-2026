"""
Match Detail Dashboard & Simulator — FIFA World Cup Match Prediction Platform.
21st.dev-inspired UI with Plotly Radar comparison, H2H timeline, and deep evidence graphs.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import load_data, get_model_and_metrics, get_team_flag, CSS, ROUND_COLORS
from ml.predict import predict_match_with_evidence, get_winner_label, get_confidence_label
from ml.data_loader import get_head_to_head, get_team_stats
from ml.features import get_ranking, get_confederation

st.set_page_config(
    page_title="Match Predictor & Detail | FIFA WC 2026 Predictor",
    page_icon="🔍",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-badge">🔍 DEEP MATCH ANALYSIS & SIMULATOR</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title" style="font-size:2.5rem; line-height:1.2;">Match Analysis & Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Deep-dive into team statistics, head-to-head records, and AI match probability simulations.</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Initializing models and data..."):
    wc_df, fixtures = load_data()
    pipeline, model_metrics = get_model_and_metrics()

# ── Mode Selection Tab ────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📅 2026 Tournament Fixtures", "🧪 Custom Matchup Simulator"])


def make_radar_chart(t1, t2, t1_stats, t2_stats):
    """Generate a Plotly Radar chart comparing two teams."""
    categories = ['FIFA Rank Rating', 'WC Win Rate', 'Attack (Goals/G)', 'Defense Rating', 'Confederation']
    
    # Normalized ratings out of 100
    r1 = get_ranking(t1)
    r2 = get_ranking(t2)
    # Normalized rank rating: rank 1 = 100, rank 60 = 10
    rank1 = max(10, min(100, int((60 - r1) / 60 * 100)))
    rank2 = max(10, min(100, int((60 - r2) / 60 * 100)))
    
    wr1 = int(t1_stats.get('win_rate', 0.33) * 100)
    wr2 = int(t2_stats.get('win_rate', 0.33) * 100)
    
    # Normalized goals scored: 0 goals = 0, 3 goals/g = 100
    g1 = min(100, int((t1_stats.get('goals_scored_per_game', 1.0) / 3.0) * 100))
    g2 = min(100, int((t2_stats.get('goals_scored_per_game', 1.0) / 3.0) * 100))
    
    # Normalized conceded rating: 0 goals conceded = 100, 3 conceded/g = 0
    c1 = max(0, min(100, int((3.0 - t1_stats.get('goals_conceded_per_game', 1.0)) / 3.0 * 100)))
    c2 = max(0, min(100, int((3.0 - t2_stats.get('goals_conceded_per_game', 1.0)) / 3.0 * 100)))
    
    # Confederation factor
    conf1 = 95 if get_confederation(t1) in ['UEFA', 'CONMEBOL'] else 65
    conf2 = 95 if get_confederation(t2) in ['UEFA', 'CONMEBOL'] else 65
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[rank1, wr1, g1, c1, conf1],
        theta=categories,
        fill='toself',
        name=t1,
        line=dict(color='#00D4FF', width=2),
        fillcolor='rgba(0, 212, 255, 0.15)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[rank2, wr2, g2, c2, conf2],
        theta=categories,
        fill='toself',
        name=t2,
        line=dict(color='#FF3CAC', width=2),
        fillcolor='rgba(255, 60, 172, 0.15)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.05)",
                tickfont=dict(color="#475569", size=9)
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.05)",
                linecolor="rgba(255,255,255,0.05)",
                tickfont=dict(color="#94a3b8", size=10)
            ),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(color="#94a3b8")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=20, b=40),
        height=320
    )
    return fig


def render_evidence_ui(pred, home_team, away_team):
    """Common UI renderer for both real and simulated matches."""
    # 3 Column Prediction & Chart Row
    col_card, col_radar, col_bar = st.columns([1.1, 1, 1.2], gap="large")
    
    winner = get_winner_label(pred["predicted_outcome"], home_team, away_team)
    conf_label, conf_emoji = get_confidence_label(pred["confidence"])
    conf_cls = "conf-high" if conf_label == "High" else ("conf-medium" if conf_label == "Medium" else "conf-low")
    
    with col_card:
        st.markdown(f"""
        <div class="glass-card" style="height:100%; display:flex; flex-direction:column; justify-content:center;">
            <div style="text-align:center; margin-bottom:12px;">
                <span class="round-badge" style="background:rgba(0,212,255,0.1); color:#00D4FF; border:1px solid rgba(0,212,255,0.25);">
                    {pred['round'].upper()} Match Analysis
                </span>
            </div>
            <div class="team-vs-block" style="margin:16px 0;">
                <div class="team-block">
                    <span class="team-flag" style="font-size:2.8rem;">{get_team_flag(home_team)}</span>
                    <div class="team-name" style="font-size:1.1rem; font-weight:800;">{home_team}</div>
                    <div class="team-rank">Rank #{get_ranking(home_team)}</div>
                </div>
                <div class="vs-separator" style="font-size:1rem; color:#475569;">VS</div>
                <div class="team-block">
                    <span class="team-flag" style="font-size:2.8rem;">{get_team_flag(away_team)}</span>
                    <div class="team-name" style="font-size:1.1rem; font-weight:800;">{away_team}</div>
                    <div class="team-rank">Rank #{get_ranking(away_team)}</div>
                </div>
            </div>
            <div style="text-align:center; border-top:1px solid rgba(255,255,255,0.05); padding-top:14px;">
                <div style="font-size:0.68rem; color:#475569; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">AI PREDICTED OUTCOME</div>
                <div class="winner-tag" style="margin:4px 0; font-size:1.4rem;">{get_team_flag(winner)} {winner}</div>
                <span class="conf-badge {conf_cls}" style="margin-top:4px;">{conf_emoji} {conf_label} Confidence ({pred['confidence']:.1%})</span>
            </div>
        </div>""", unsafe_allow_html=True)
        
    with col_radar:
        # Radar Chart
        all_stats = get_team_stats(wc_df)
        h_s = all_stats.loc[home_team].to_dict() if home_team in all_stats.index else {"win_rate": 0.33, "goals_scored_per_game": 1.0, "goals_conceded_per_game": 1.0}
        a_s = all_stats.loc[away_team].to_dict() if away_team in all_stats.index else {"win_rate": 0.33, "goals_scored_per_game": 1.0, "goals_conceded_per_game": 1.0}
        
        st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#00D4FF; letter-spacing:1px; text-align:center; margin-bottom:10px;">📊 RADAR STAT COMPARISON</div>', unsafe_allow_html=True)
        fig_radar = make_radar_chart(home_team, away_team, h_s, a_s)
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with col_bar:
        st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#00D4FF; letter-spacing:1px; margin-bottom:10px;">📈 PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)
        hp = pred["home_win_prob"]
        dp = pred["draw_prob"] if not pred["is_knockout"] else 0
        ap = pred["away_win_prob"]
        
        # Home Prob
        st.markdown(f"""
        <div class="prob-meter-container">
            <div class="prob-meter-header">
                <span class="prob-meter-label">{get_team_flag(home_team)} {home_team} Win</span>
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
                <span class="prob-meter-label">{get_team_flag(away_team)} {away_team} Win</span>
                <span class="prob-meter-value" style="color:#FF3CAC;">{ap:.1%}</span>
            </div>
            <div class="prob-meter-track">
                <div class="prob-meter-fill away" style="width:{ap*100:.1f}%"></div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        # AI verdict summary card
        st.markdown(f"""
        <div class="evidence-card" style="margin-top:14px; background:rgba(0,212,255,0.05); border-color:rgba(0,212,255,0.2);">
            <div class="evidence-label" style="font-size:0.7rem;">🧠 DEEP PREDICTION VERDICT</div>
            <div class="evidence-text" style="font-size:0.8rem; line-height:1.4;">{pred['verdict']}</div>
        </div>""", unsafe_allow_html=True)

    # ── Evidence & Risks Row ──────────────────────────────────────────────────
    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
    col_ev, col_risk = st.columns(2, gap="large")
    
    with col_ev:
        st.markdown('<div class="section-header" style="margin-top:0;">📈 Supporting Evidence Chain</div>', unsafe_allow_html=True)
        for item in pred["evidence"]:
            st.markdown(f"""
            <div class="evidence-card">
                <div class="evidence-label">{item['label']}</div>
                <div class="evidence-text">{item['text']}</div>
                <div class="evidence-sub">{item['sub']}</div>
            </div>""", unsafe_allow_html=True)
            
    with col_risk:
        st.markdown('<div class="section-header" style="margin-top:0; border-bottom-color:rgba(239,68,68,0.25);">⚠️ Potential Risk Factors</div>', unsafe_allow_html=True)
        for item in pred["risks"]:
            st.markdown(f"""
            <div class="risk-card">
                <div class="risk-label">{item['label']}</div>
                <div class="evidence-text">{item['text']}</div>
            </div>""", unsafe_allow_html=True)

    # ── Form and Feature Contributions Row ───────────────────────────────────
    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
    col_form, col_cont = st.columns([1, 1.2], gap="large")
    
    with col_form:
        st.markdown('<div class="section-header" style="margin-top:0;">⚡ Team Form & Trajectory</div>', unsafe_allow_html=True)
        
        def render_form_timeline(team, form_data):
            html = f'<div style="background:rgba(255,255,255,0.02); padding:16px; border-radius:12px; border:1px solid rgba(255,255,255,0.03); margin-bottom:12px;">'
            html += f'<div style="font-weight:700; font-size:0.9rem; color:#E5E7EB; margin-bottom:8px;">{get_team_flag(team)} {team} Recent WC Games</div>'
            html += '<div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:8px;">'
            for game in form_data:
                bg = "#10b981" if game["result"] == "W" else "#ef4444" if game["result"] == "L" else "#64748b"
                html += f"""
                <div style="text-align:center; background:rgba(0,0,0,0.3); padding:8px; border-radius:8px; border:1px solid rgba(255,255,255,0.02);">
                    <div style="font-size:0.6rem; color:#475569;">{game['year']}</div>
                    <span style="
                        display:inline-flex; align-items:center; justify-content:center;
                        width:24px; height:24px; border-radius:50%;
                        font-size:0.8rem; font-weight:800; color:white;
                        background:{bg}; margin:6px 0;
                    ">{game['result']}</span>
                    <div style="font-size:0.58rem; font-weight:700; color:#94a3b8; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{game['opponent']}</div>
                    <div style="font-size:0.6rem; color:#475569;">{game['score']}</div>
                </div>"""
            html += '</div></div>'
            return html

        st.markdown(render_form_timeline(home_team, pred["home_form"]), unsafe_allow_html=True)
        st.markdown(render_form_timeline(away_team, pred["away_form"]), unsafe_allow_html=True)
        
    with col_cont:
        st.markdown('<div class="section-header" style="margin-top:0;">🤖 Model Feature Drivers (SHAP-style)</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.75rem; color:#94a3b8; margin-top:-10px; margin-bottom:12px;">Individual feature weights showing how strongly each attribute pushes the model prediction.</p>', unsafe_allow_html=True)
        
        for c in pred["contributions"]:
            arrow = "→" if c["direction"] == "home" else "←" if c["direction"] == "away" else "⚖️"
            col_accent = "#00D4FF" if c["direction"] == "home" else "#FF3CAC" if c["direction"] == "away" else "#94a3b8"
            chip_cls = "chip-positive" if c["type"] == "positive" else "chip-negative" if c["type"] == "negative" else "chip-neutral"
            
            st.markdown(f"""
            <div style="
                display:flex; justify-content:space-between; align-items:center; 
                background:rgba(255,255,255,0.01); border:1px solid rgba(255,255,255,0.03);
                border-radius:10px; padding:10px 14px; margin:6px 0;
            ">
                <div>
                    <span class="factor-chip {chip_cls}">{c['factor']}</span>
                    <div style="font-size:0.75rem; color:#475569; margin-top:4px;">{c['detail']}</div>
                </div>
                <div style="text-align:right; min-width:80px;">
                    <div style="font-size:0.95rem; font-weight:800; color:{col_accent};">{arrow} {c['strength']:.1%}</div>
                    <div style="font-size:0.58rem; color:#475569; text-transform:uppercase;">Influence</div>
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Head-to-Head encounters row ──────────────────────────────────────────
    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
    st.markdown("### ⚔️ Historical Head-to-Head World Cup Record")
    h2h_data = get_head_to_head(wc_df, home_team, away_team)
    
    if h2h_data["matches"] == 0:
        st.info(f"These teams have never played each other in a FIFA World Cup match. Predictions are based on historical ranking and performance features.")
    else:
        c_h1, c_h2, c_h3 = st.columns(3)
        with c_h1:
            st.metric("Total WC Matches", h2h_data["matches"])
        with c_h2:
            st.metric(f"{home_team} Wins", h2h_data["team1_wins"])
        with c_h3:
            st.metric(f"{away_team} Wins", h2h_data["team2_wins"])
            
        st.markdown("##### Past World Cup Matches")
        h2h_df = pd.DataFrame(h2h_data["history"])
        h2h_df["date"] = pd.to_datetime(h2h_df["date"]).dt.strftime("%Y-%m-%d")
        h2h_df = h2h_df.rename(columns={
            "date": "Date",
            "home_team": "Home Team",
            "away_team": "Away Team",
            "home_score": "Home Score",
            "away_score": "Away Score"
        })
        st.dataframe(h2h_df, use_container_width=True, hide_index=True)


# ── Tab 1: Real 2026 Fixtures ─────────────────────────────────────────────────
with tab1:
    scheduled_fixtures = fixtures[
        (fixtures["home_team"] != "TBD") & (fixtures["away_team"] != "TBD")
    ].copy()
    
    if scheduled_fixtures.empty:
        st.warning("No scheduled matches available with confirmed teams. Try the Custom Matchup Simulator!")
    else:
        scheduled_fixtures["display_name"] = scheduled_fixtures.apply(
            lambda r: f"{r['round']} — {r['home_team']} vs {r['away_team']} ({r['date']})", axis=1
        )
        
        selected_match_name = st.selectbox(
            "Select a 2026 World Cup Fixture to Analyze",
            options=scheduled_fixtures["display_name"].tolist()
        )
        
        selected_row = scheduled_fixtures[scheduled_fixtures["display_name"] == selected_match_name].iloc[0]
        home_t = selected_row["home_team"]
        away_t = selected_row["away_team"]
        round_t = selected_row["round"]
        
        # Predict with complete evidence
        pred_data = predict_match_with_evidence(
            home_team=home_t,
            away_team=away_t,
            round_name=round_t,
            wc_df=wc_df,
        )
        pred_data["round"] = round_t
        
        st.markdown("<br>", unsafe_allow_html=True)
        render_evidence_ui(pred_data, home_t, away_t)

# ── Tab 2: Custom Matchup Simulator ───────────────────────────────────────────
with tab2:
    all_teams_list = sorted(list(get_team_stats(wc_df).index))
    if not all_teams_list:
        all_teams_list = ["Argentina", "Brazil", "France", "Germany", "England", "Spain", "Italy", "Uruguay"]
        
    st.markdown("##### 🧪 Configure Simulated Match Setup")
    col_sel1, col_sel2, col_sel3 = st.columns(3)
    
    with col_sel1:
        sim_home = st.selectbox("Home Team Select", options=all_teams_list, index=0)
    with col_sel2:
        sim_away = st.selectbox("Away Team Select", options=all_teams_list, index=1 if len(all_teams_list) > 1 else 0)
    with col_sel3:
        sim_round = st.selectbox(
            "Tournament Round Selection", 
            options=["Group Stage", "Round of 32", "Round of 16", "Quarterfinal", "Semifinal", "Final"],
            index=0
        )
        
    if sim_home == sim_away:
        st.error("Please select two different teams to run a simulated prediction.")
    else:
        # Advanced configurations
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("⚙️ Customize Team Form & Simulation Modifiers"):
            st.markdown("Adjust team rest days and goalscoring rates to see how they impact predicted win probability:")
            c_form1, c_form2 = st.columns(2)
            with c_form1:
                st.markdown(f"**{sim_home} Simulation Config**")
                home_goals = st.slider(f"{sim_home} Average Goals Scored Form", 0.0, 5.0, 1.5, 0.1)
                home_conceded = st.slider(f"{sim_home} Average Goals Conceded Form", 0.0, 5.0, 1.0, 0.1)
                home_rest = st.slider(f"{sim_home} Rest Days Mod", 1, 14, 7, 1)
            with c_form2:
                st.markdown(f"**{sim_away} Simulation Config**")
                away_goals = st.slider(f"{sim_away} Average Goals Scored Form", 0.0, 5.0, 1.5, 0.1)
                away_conceded = st.slider(f"{sim_away} Average Goals Conceded Form", 0.0, 5.0, 1.0, 0.1)
                away_rest = st.slider(f"{sim_away} Rest Days Mod", 1, 14, 7, 1)
                
        # Run Simulation with custom form inputs
        pred_sim = predict_match_with_evidence(
            home_team=sim_home,
            away_team=sim_away,
            round_name=sim_round,
            wc_df=wc_df,
            home_goals_pg=home_goals,
            away_goals_pg=away_goals,
            home_conceded_pg=home_conceded,
            away_conceded_pg=away_conceded,
            home_rest_days=home_rest,
            away_rest_days=away_rest,
        )
        pred_sim["round"] = sim_round
        
        st.markdown("<br>", unsafe_allow_html=True)
        render_evidence_ui(pred_sim, sim_home, sim_away)
