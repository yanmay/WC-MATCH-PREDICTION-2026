"""
Tournament Bracket Simulator — FIFA World Cup Match Prediction Platform.
Glassmorphism bracket nodes, neon glowing links, and champion spotlights.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import load_data, get_model_and_metrics, get_team_flag, CSS, format_team_html, format_team_emoji, render_sidebar
from ml.data_loader import get_2026_fixtures
from ml.predict import predict_match, get_winner_label

st.set_page_config(
    page_title="Tournament Bracket Predictor | FIFA WC 2026 Predictor",
    page_icon="🏆",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)
render_sidebar()

# Custom CSS extension for 21st.dev styling on the Bracket Visual Flow
st.markdown("""
<style>
.bracket-scroll-container {
    width: 100%;
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
    padding-bottom: 16px;
    margin-top: 10px;
}
.bracket-flow-wrapper {
    display: flex;
    gap: 16px;
    min-width: 1400px;
    justify-content: space-between;
    align-items: stretch;
}
.bracket-column {
    display: flex;
    flex-direction: column;
    justify-content: space-around;
    height: 950px;
    flex: 1;
    min-width: 130px;
}
.bracket-match {
    background: var(--bg-card);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 10px 14px;
    margin: 6px 0;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    font-size: 0.78rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}
.bracket-match::before {
    content: '';
    position: absolute;
    top: 0; left: 0; bottom: 0;
    width: 3px;
    background: linear-gradient(180deg, var(--accent-green), var(--accent-purple));
    border-radius: 12px 0 0 12px;
    opacity: 0.7;
}
.bracket-match:hover {
    border-color: var(--border-accent);
    box-shadow: 0 0 15px rgba(52, 211, 153, 0.15), 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    transform: translateY(-1px);
}
.bracket-team {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 3px 0;
}
.bracket-winner {
    color: var(--accent-green) !important;
    font-weight: 800;
}
.bracket-loser {
    color: var(--text-muted) !important;
    font-weight: 400;
}
.bracket-prob {
    font-size: 0.68rem;
    color: var(--text-muted);
    font-weight: 700;
}
.champion-spotlight-21 {
    background: linear-gradient(135deg, rgba(250, 204, 21, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%);
    border: 1px solid rgba(250, 204, 21, 0.3);
    box-shadow: 0 0 35px rgba(250, 204, 21, 0.08), inset 0 0 20px rgba(250, 204, 21, 0.02);
    border-radius: 20px;
    padding: 24px;
    text-align: center;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.champion-spotlight-21::before {
    content: '';
    position: absolute;
    top: 0; left: 25%; right: 25%;
    height: 1px;
    background: linear-gradient(90deg, transparent, #FACC15, transparent);
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-badge">🏆 KNOCKOUT BRACKET PREDICTION</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title" style="font-size:2.5rem; line-height:1.2;">Tournament Bracket Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Simulate the entire knockout stage of the 2026 FIFA World Cup and predict the next Champion.</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Preparing tournament simulator..."):
    wc_df, fixtures = load_data()
    pipeline, model_metrics = get_model_and_metrics()

# ── Simulation Logic ──────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_bracket_simulation(wc_df_local):
    """Propagate predictions stage-by-stage to simulate the full tournament."""
    # Round of 32
    r32_fixtures = get_2026_fixtures(wc_df_local)
    r32_matches = r32_fixtures[r32_fixtures["round"] == "Round of 32"].copy()
    
    r32_results = []
    for _, match in r32_matches.iterrows():
        pred = predict_match(match["home_team"], match["away_team"], "Round of 32", wc_df_local)
        winner = get_winner_label(pred["predicted_outcome"], match["home_team"], match["away_team"])
        loser = match["away_team"] if winner == match["home_team"] else match["home_team"]
        r32_results.append({
            "match_id": match["match_id"],
            "home": match["home_team"],
            "away": match["away_team"],
            "winner": winner,
            "loser": loser,
            "pred": pred
        })
        
    # Round of 16
    r16_results = []
    r16_pairs = [
        (1, 2), (3, 4), (5, 6), (7, 8),
        (9, 10), (11, 12), (13, 14), (15, 16)
    ]
    for idx, (m1_id, m2_id) in enumerate(r16_pairs):
        t1 = next(r["winner"] for r in r32_results if r["match_id"] == m1_id)
        t2 = next(r["winner"] for r in r32_results if r["match_id"] == m2_id)
        
        pred = predict_match(t1, t2, "Round of 16", wc_df_local)
        winner = get_winner_label(pred["predicted_outcome"], t1, t2)
        loser = t2 if winner == t1 else t1
        r16_results.append({
            "match_id": 17 + idx,
            "home": t1,
            "away": t2,
            "winner": winner,
            "loser": loser,
            "pred": pred
        })
        
    # Quarterfinals
    qf_results = []
    qf_pairs = [
        (17, 18), (19, 20), (21, 22), (23, 24)
    ]
    for idx, (m1_id, m2_id) in enumerate(qf_pairs):
        t1 = next(r["winner"] for r in r16_results if r["match_id"] == m1_id)
        t2 = next(r["winner"] for r in r16_results if r["match_id"] == m2_id)
        
        pred = predict_match(t1, t2, "Quarterfinal", wc_df_local)
        winner = get_winner_label(pred["predicted_outcome"], t1, t2)
        loser = t2 if winner == t1 else t1
        qf_results.append({
            "match_id": 25 + idx,
            "home": t1,
            "away": t2,
            "winner": winner,
            "loser": loser,
            "pred": pred
        })
        
    # Semifinals
    sf_results = []
    sf_pairs = [
        (25, 26), (27, 28)
    ]
    for idx, (m1_id, m2_id) in enumerate(sf_pairs):
        t1 = next(r["winner"] for r in qf_results if r["match_id"] == m1_id)
        t2 = next(r["winner"] for r in qf_results if r["match_id"] == m2_id)
        
        pred = predict_match(t1, t2, "Semifinal", wc_df_local)
        winner = get_winner_label(pred["predicted_outcome"], t1, t2)
        loser = t2 if winner == t1 else t1
        sf_results.append({
            "match_id": 29 + idx,
            "home": t1,
            "away": t2,
            "winner": winner,
            "loser": loser,
            "pred": pred
        })
        
    # Final
    t1 = next(r["winner"] for r in sf_results if r["match_id"] == 29)
    t2 = next(r["winner"] for r in sf_results if r["match_id"] == 30)
    pred = predict_match(t1, t2, "Final", wc_df_local)
    winner = get_winner_label(pred["predicted_outcome"], t1, t2)
    loser = t2 if winner == t1 else t1
    final_result = {
        "match_id": 32,
        "home": t1,
        "away": t2,
        "winner": winner,
        "loser": loser,
        "pred": pred
    }
    
    return {
        "Round of 32": r32_results,
        "Round of 16": r16_results,
        "Quarterfinal": qf_results,
        "Semifinal": sf_results,
        "Final": final_result
    }

# Run simulation
sim_data = run_bracket_simulation(wc_df)

# Champion Spotlight Card
champion = sim_data["Final"]["winner"]
final_pred = sim_data["Final"]["pred"]
champ_prob = final_pred["home_win_prob"] if champion == sim_data["Final"]["home"] else final_pred["away_win_prob"]
champ_prob_str = f"{champ_prob:.1%}" if champ_prob is not None else "N/A"

st.markdown(f"""
<div class="champion-spotlight-21">
    <div style="font-size:3rem; margin-bottom:6px; filter: drop-shadow(0 0 10px rgba(250,204,21,0.6));">🏆</div>
    <div style="color:#FACC15; font-size:0.75rem; font-weight:800; letter-spacing:2px; text-transform:uppercase; margin-bottom:4px;">
        AI WORLD CUP CHAMPION FORECAST
    </div>
    <div style="font-size:2.5rem; font-weight:900; color:#f3f4f6; margin:4px 0;">
        {format_team_html(champion)}
    </div>
    <div style="color:#9ca3af; font-size:0.88rem; max-width:600px; margin:0 auto; line-height:1.5;">
        Predicted to defeat {format_team_html(sim_data['Final']['loser'])} in the Final match with a simulated probability of <b>{champ_prob_str}</b>.
    </div>
</div>""", unsafe_allow_html=True)

# ── Bracket Visual Layout ─────────────────────────────────────────────────────
st.markdown("### 📊 Symmetrical Knockout Bracket Flow")
st.markdown('<p style="font-size:0.8rem; color:#94a3b8; margin-top:-10px; margin-bottom:20px;">Follow the tournament progression step-by-step from the Round of 32 up to the Final match.</p>', unsafe_allow_html=True)

def render_match_html(match):
    """Generate HTML block for a bracket match."""
    home_win = match["winner"] == match["home"]
    away_win = match["winner"] == match["away"]
    
    hp = match["pred"]["home_win_prob"]
    ap = match["pred"]["away_win_prob"]
    
    h_class = "bracket-winner" if home_win else "bracket-loser"
    a_class = "bracket-winner" if away_win else "bracket-loser"
    
    hp_str = f"{hp:.0%}" if hp is not None else "N/A"
    ap_str = f"{ap:.0%}" if ap is not None else "N/A"
    
    return (
        f'<div class="bracket-match">'
        f'<div class="bracket-team {h_class}">'
        f'<span>{format_team_html(match["home"])}</span>'
        f'<span class="bracket-prob">{hp_str}</span>'
        f'</div>'
        f'<div style="height:1px; background:var(--border-subtle); margin:4px 0;"></div>'
        f'<div class="bracket-team {a_class}">'
        f'<span>{format_team_html(match["away"])}</span>'
        f'<span class="bracket-prob">{ap_str}</span>'
        f'</div>'
        f'</div>'
    )

# Symmetrical bracket inside a single horizontally scrollable HTML container
html = '<div class="bracket-scroll-container">'
html += '<div class="bracket-flow-wrapper">'

# Column 1: Round of 32 Left
html += '<div class="bracket-column">'
html += '<div style="text-align:center; font-size:0.6rem; font-weight:700; color:#9ca3af; letter-spacing:1px; margin-bottom:4px;">ROUND OF 32</div>'
for i in range(8):
    match = sim_data["Round of 32"][i]
    html += render_match_html(match)
html += '</div>'

# Column 2: Round of 16 Left
html += '<div class="bracket-column">'
html += '<div style="text-align:center; font-size:0.6rem; font-weight:700; color:#9ca3af; letter-spacing:1px; margin-bottom:4px;">ROUND OF 16</div>'
for i in range(4):
    match = sim_data["Round of 16"][i]
    html += render_match_html(match)
html += '</div>'

# Column 3: Quarterfinal Left
html += '<div class="bracket-column">'
html += '<div style="text-align:center; font-size:0.6rem; font-weight:700; color:#9ca3af; letter-spacing:1px; margin-bottom:4px;">QUARTERFINAL</div>'
for i in range(2):
    match = sim_data["Quarterfinal"][i]
    html += render_match_html(match)
html += '</div>'

# Column 4: Semifinal Left
html += '<div class="bracket-column">'
html += '<div style="text-align:center; font-size:0.6rem; font-weight:700; color:#9ca3af; letter-spacing:1px; margin-bottom:4px;">SEMIFINAL</div>'
match = sim_data["Semifinal"][0]
html += render_match_html(match)
html += '</div>'

# Column 5: Center (Final)
html += '<div class="bracket-column">'
html += '<div style="text-align:center; font-size:0.6rem; font-weight:800; color:#FACC15; letter-spacing:1.5px; margin-bottom:4px;">FINAL</div>'
match = sim_data["Final"]
html += render_match_html(match)
html += '</div>'

# Column 6: Semifinal Right
html += '<div class="bracket-column">'
html += '<div style="text-align:center; font-size:0.6rem; font-weight:700; color:#9ca3af; letter-spacing:1px; margin-bottom:4px;">SEMIFINAL</div>'
match = sim_data["Semifinal"][1]
html += render_match_html(match)
html += '</div>'

# Column 7: Quarterfinal Right
html += '<div class="bracket-column">'
html += '<div style="text-align:center; font-size:0.6rem; font-weight:700; color:#9ca3af; letter-spacing:1px; margin-bottom:4px;">QUARTERFINAL</div>'
for i in range(2, 4):
    match = sim_data["Quarterfinal"][i]
    html += render_match_html(match)
html += '</div>'

# Column 8: Round of 16 Right
html += '<div class="bracket-column">'
html += '<div style="text-align:center; font-size:0.6rem; font-weight:700; color:#9ca3af; letter-spacing:1px; margin-bottom:4px;">ROUND OF 16</div>'
for i in range(4, 8):
    match = sim_data["Round of 16"][i]
    html += render_match_html(match)
html += '</div>'

# Column 9: Round of 32 Right
html += '<div class="bracket-column">'
html += '<div style="text-align:center; font-size:0.6rem; font-weight:700; color:#9ca3af; letter-spacing:1px; margin-bottom:4px;">ROUND OF 32</div>'
for i in range(8, 16):
    match = sim_data["Round of 32"][i]
    html += render_match_html(match)
html += '</div>'

html += '</div></div>'

st.markdown(html, unsafe_allow_html=True)
