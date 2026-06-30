"""
Match Detail Dashboard & Simulator — FIFA World Cup Match Prediction Platform.
21st.dev-inspired dark UI with Plotly Radar comparison, H2H timeline, and deep evidence graphs.
"""

import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import load_data, get_model_and_metrics, get_team_flag, format_team_html, format_team_emoji, CSS, ROUND_COLORS, render_sidebar
from ml.predict import predict_match_with_evidence, get_winner_label, get_confidence_label
from ml.data_loader import get_head_to_head, get_team_stats
from ml.features import get_ranking, get_confederation

st.set_page_config(
    page_title="Match Predictor & Detail | FIFA WC 2026 Predictor",
    page_icon="🔍",
    layout="wide",
)
st.markdown(CSS, unsafe_allow_html=True)
render_sidebar()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-badge">🔍 DEEP MATCH ANALYSIS &amp; SIMULATOR</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title" style="font-size:2.5rem; line-height:1.2;">Match Analysis &amp; Simulator</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Deep-dive into team statistics, head-to-head records, and AI match probability simulations.</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Load Data ─────────────────────────────────────────────────────────────────
with st.spinner("Initializing models and data..."):
    wc_df, fixtures = load_data()
    pipeline, model_metrics = get_model_and_metrics()

# ── Mode Selection Tab ────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📅 2026 Tournament Fixtures", "🧪 Custom Matchup Simulator"])


def make_radar_chart(t1, t2, t1_stats, t2_stats):
    """Generate a Plotly Radar chart comparing two teams — dark mode."""
    categories = ['FIFA Rank Rating', 'WC Win Rate', 'Attack (Goals/G)', 'Defense Rating', 'Confederation']
    
    r1 = get_ranking(t1)
    r2 = get_ranking(t2)
    rank1 = max(10, min(100, int((60 - r1) / 60 * 100)))
    rank2 = max(10, min(100, int((60 - r2) / 60 * 100)))
    
    wr1 = int(t1_stats.get('win_rate', 0.33) * 100)
    wr2 = int(t2_stats.get('win_rate', 0.33) * 100)
    
    g1 = min(100, int((t1_stats.get('goals_scored_per_game', 1.0) / 3.0) * 100))
    g2 = min(100, int((t2_stats.get('goals_scored_per_game', 1.0) / 3.0) * 100))
    
    c1 = max(0, min(100, int((3.0 - t1_stats.get('goals_conceded_per_game', 1.0)) / 3.0 * 100)))
    c2 = max(0, min(100, int((3.0 - t2_stats.get('goals_conceded_per_game', 1.0)) / 3.0 * 100)))
    
    conf1 = 95 if get_confederation(t1) in ['UEFA', 'CONMEBOL'] else 65
    conf2 = 95 if get_confederation(t2) in ['UEFA', 'CONMEBOL'] else 65
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[rank1, wr1, g1, c1, conf1],
        theta=categories,
        fill='toself',
        name=t1,
        line=dict(color='#34d399', width=2),
        fillcolor='rgba(52, 211, 153, 0.15)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[rank2, wr2, g2, c2, conf2],
        theta=categories,
        fill='toself',
        name=t2,
        line=dict(color='#f87171', width=2),
        fillcolor='rgba(248, 113, 113, 0.15)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(255,255,255,0.08)",
                linecolor="rgba(255,255,255,0.08)",
                tickfont=dict(color="#6b7280", size=9)
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.08)",
                linecolor="rgba(255,255,255,0.08)",
                tickfont=dict(color="#9ca3af", size=10)
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
            font=dict(color="#9ca3af"),
            bgcolor="rgba(0,0,0,0)"
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=20, b=40),
        height=320
    )
    return fig


TOP_PLAYERS = {
    "Argentina": [
        ("Emi Martínez", "GK", 89), ("Cristian Romero", "DF", 87), ("Nicolás Otamendi", "DF", 84),
        ("Nahuel Molina", "DF", 82), ("Enzo Fernández", "MF", 85), ("Alexis Mac Allister", "MF", 86),
        ("Rodrigo De Paul", "MF", 84), ("Lionel Messi", "FW", 92), ("Lautaro Martínez", "FW", 89),
        ("Julián Álvarez", "FW", 86), ("Nicolás González", "FW", 81)
    ],
    "France": [
        ("Mike Maignan", "GK", 87), ("William Saliba", "DF", 88), ("Dayot Upamecano", "DF", 84),
        ("Jules Koundé", "DF", 84), ("Theo Hernández", "DF", 85), ("Aurélien Tchouaméni", "MF", 86),
        ("Eduardo Camavinga", "MF", 85), ("Antoine Griezmann", "MF", 88), ("Kylian Mbappé", "FW", 93),
        ("Ousmane Dembélé", "FW", 86), ("Marcus Thuram", "FW", 84)
    ],
    "Belgium": [
        ("Koen Casteels", "GK", 84), ("Wout Faes", "DF", 81), ("Zeno Debast", "DF", 79),
        ("Timothy Castagne", "DF", 80), ("Arthur Theate", "DF", 80), ("Amadou Onana", "MF", 83),
        ("Youri Tielemans", "MF", 82), ("Kevin De Bruyne", "MF", 91), ("Leandro Trossard", "FW", 84),
        ("Jérémy Doku", "FW", 85), ("Romelu Lukaku", "FW", 84)
    ],
    "Norway": [
        ("Ørjan Nyland", "GK", 78), ("Leo Østigård", "DF", 79), ("Kristoffer Ajer", "DF", 79),
        ("Marcus Pedersen", "DF", 76), ("Julian Ryerson", "DF", 80), ("Martin Ødegaard", "MF", 89),
        ("Sander Berge", "MF", 80), ("Patrick Berg", "MF", 77), ("Erling Haaland", "FW", 91),
        ("Alexander Sørloth", "FW", 82), ("Antonio Nusa", "FW", 79)
    ],
    "Spain": [
        ("Unai Simón", "GK", 86), ("Robin Le Normand", "DF", 83), ("Aymeric Laporte", "DF", 84),
        ("Dani Carvajal", "DF", 86), ("Marc Cucurella", "DF", 82), ("Rodri", "MF", 91),
        ("Fabián Ruiz", "MF", 84), ("Dani Olmo", "MF", 86), ("Lamine Yamal", "FW", 88),
        ("Nico Williams", "FW", 86), ("Alvaro Morata", "FW", 83)
    ],
    "Uruguay": [
        ("Sergio Rochet", "GK", 81), ("Ronald Araujo", "DF", 86), ("José María Giménez", "DF", 83),
        ("Mathías Olivera", "DF", 80), ("Federico Valverde", "MF", 89), ("Manuel Ugarte", "MF", 83),
        ("Nicolás de la Cruz", "MF", 80), ("Rodrigo Bentancur", "MF", 82), ("Facundo Pellistri", "FW", 79),
        ("Darwin Núñez", "FW", 85), ("Luis Suárez", "FW", 80)
    ],
    "Senegal": [
        ("Edouard Mendy", "GK", 82), ("Kalidou Koulibaly", "DF", 84), ("Abdou Diallo", "DF", 79),
        ("Moussa Niakhaté", "DF", 79), ("Idrissa Gueye", "MF", 78), ("Pape Matar Sarr", "MF", 81),
        ("Lamine Camara", "MF", 79), ("Sadio Mané", "FW", 84), ("Nicolas Jackson", "FW", 82),
        ("Ismaïla Sarr", "FW", 79), ("Iliman Ndiaye", "FW", 78)
    ],
    "Iraq": [
        ("Jalal Hassan", "GK", 72), ("Rebin Sulaka", "DF", 70), ("Saad Natiq", "DF", 69),
        ("Hussein Ali", "DF", 71), ("Mergas Doski", "DF", 70), ("Amir Al-Ammari", "MF", 73),
        ("Osama Rashid", "MF", 70), ("Ibrahim Bayesh", "MF", 72), ("Youssef Amyn", "FW", 71),
        ("Ali Jasim", "FW", 74), ("Aymen Hussein", "FW", 76)
    ],
    "Egypt": [
        ("Mohamed El Shenawy", "GK", 79), ("Mohamed Abdelmonem", "DF", 79), ("Ahmed Hegazi", "DF", 75),
        ("Mohamed Hany", "DF", 74), ("Marwan Attia", "MF", 75), ("Mohamed Elneny", "MF", 76),
        ("Emam Ashour", "MF", 78), ("Mohamed Salah", "FW", 89), ("Mostafa Mohamed", "FW", 79),
        ("Trezeguet", "FW", 78), ("Omar Marmoush", "FW", 82)
    ],
    "Iran": [
        ("Alireza Beiranvand", "GK", 77), ("Shojae Khalilzadeh", "DF", 73), ("Hossein Kanaanizadegan", "DF", 74),
        ("Milad Mohammadi", "DF", 73), ("Ramin Rezaeian", "DF", 75), ("Saeid Ezatolahi", "MF", 75),
        ("Saman Ghoddos", "MF", 74), ("Alireza Jahanbakhsh", "FW", 75), ("Mehdi Taremi", "FW", 82),
        ("Sardar Azmoun", "FW", 80), ("Ali Gholizadeh", "FW", 74)
    ],
    "New Zealand": [
        ("Alex Paulsen", "GK", 72), ("Michael Boxall", "DF", 71), ("Tyler Bindon", "DF", 70),
        ("Liberato Cacace", "DF", 74), ("Tommy Smith", "DF", 67), ("Joe Bell", "MF", 72),
        ("Sarpreet Singh", "MF", 73), ("Marko Stamenic", "MF", 74), ("Chris Wood", "FW", 79),
        ("Elijah Just", "FW", 71), ("Ben Waine", "FW", 68)
    ]
}


def get_squad(team_name: str) -> list:
    if team_name in TOP_PLAYERS:
        return TOP_PLAYERS[team_name]
    positions = ["GK", "DF", "DF", "DF", "DF", "MF", "MF", "MF", "FW", "FW", "FW"]
    squad = []
    import hashlib
    for i, pos in enumerate(positions, 1):
        h = hashlib.md5(f"{team_name}_{i}".encode()).hexdigest()
        rating = 72 + (int(h[:2], 16) % 15)
        # Select realistic sounding generic names
        if pos == "GK":
            name = f"GK {team_name[:3].upper()} {i}"
        elif pos == "DF":
            name = f"DF {team_name[:3].upper()} {i}"
        elif pos == "MF":
            name = f"MF {team_name[:3].upper()} {i}"
        else:
            name = f"FW {team_name[:3].upper()} {i}"
        squad.append((name, pos, rating))
    return squad


def get_player_match_stats(name, pos, rating, is_home, hs, as_, scorers=None, home_team=None, away_team=None):
    import hashlib
    h = int(hashlib.md5(f"{name}_{hs}_{as_}".encode()).hexdigest()[:4], 16)
    
    base_rating = 6.0 + (rating - 70) / 10 + (h % 10) / 10
    if is_home:
        base_rating += 0.5 if hs > as_ else (-0.5 if as_ > hs else 0)
    else:
        base_rating += 0.5 if as_ > hs else (-0.5 if hs > as_ else 0)
    base_rating = min(10.0, max(5.0, base_rating))
    
    # Check actual goals from scorers if available
    goals = 0
    if scorers and isinstance(scorers, dict):
        scorer_list = scorers.get("home" if is_home else "away") or scorers.get(home_team if is_home else away_team) or []
        if isinstance(scorer_list, str):
            try:
                import json
                scorer_list = json.loads(scorer_list)
            except Exception:
                scorer_list = [scorer_list]
        
        import unicodedata
        def norm(text):
            t = unicodedata.normalize('NFD', str(text))
            return "".join([c for c in t if not unicodedata.combining(c)]).lower().replace("-", " ").strip()
            
        norm_name = norm(name)
        for s in scorer_list:
            if s:
                norm_scorer = norm(s)
                # Check if full name matches, or if last name matches
                if norm_name in norm_scorer or (len(norm_name.split()) > 1 and norm_name.split()[-1] in norm_scorer):
                    goals += 1
                
    details = ""
    if pos == "GK":
        saves = 1 + (h % 5)
        conceded = as_ if is_home else hs
        details = f"{saves} Saves, {conceded} Conceded"
    elif pos == "DF":
        tackles = 1 + (h % 6)
        blocks = h % 3
        details = f"{tackles} Tackles, {blocks} Blocks"
    elif pos == "MF":
        passes = 30 + (h % 50)
        key_passes = h % 4
        if goals > 0:
            details = f"{passes} Passes, {goals} Goal" + ("s" if goals > 1 else "")
        else:
            details = f"{passes} Passes, {key_passes} Key Pass"
    else:
        shots = 1 + (h % 5)
        if not scorers:
            if is_home and hs > 0 and (h % 3) == 0:
                goals = min(hs, 1 + (h % 2))
            elif not is_home and as_ > 0 and (h % 3) == 0:
                goals = min(as_, 1 + (h % 2))
        if goals > 0:
            details = f"{shots} Shots, {goals} Goal" + ("s" if goals > 1 else "")
        else:
            details = f"{shots} Shots"
            
    return round(base_rating, 1), details


def generate_timeline(home_team, away_team, hs, as_, h_squad, a_squad, scorers=None):
    import hashlib
    events = []
    
    # Use real scorers if available
    if scorers and isinstance(scorers, dict):
        home_list = scorers.get("home") or scorers.get(home_team) or []
        away_list = scorers.get("away") or scorers.get(away_team) or []
        
        if isinstance(home_list, str):
            try:
                import json
                home_list = json.loads(home_list)
            except Exception:
                home_list = [home_list]
        if isinstance(away_list, str):
            try:
                import json
                away_list = json.loads(away_list)
            except Exception:
                away_list = [away_list]
                
        import re
        for s in home_list:
            if not s:
                continue
            minute_match = re.search(r"(\d+)(?:\+\d+)?'", s)
            minute = int(minute_match.group(1)) if minute_match else 45
            events.append({"minute": minute, "team": home_team, "type": "Goal", "detail": f"⚽ Goal: {s}"})
            
        for s in away_list:
            if not s:
                continue
            minute_match = re.search(r"(\d+)(?:\+\d+)?'", s)
            minute = int(minute_match.group(1)) if minute_match else 45
            events.append({"minute": minute, "team": away_team, "type": "Goal", "detail": f"⚽ Goal: {s}"})
            
        # Draw some yellow cards for realistic timeline display
        h_yellow = int(hashlib.md5(f"yellow_h_{home_team}".encode()).hexdigest()[:2], 16) % 3
        for i in range(h_yellow):
            h = int(hashlib.md5(f"card_h_{home_team}_{i}".encode()).hexdigest()[:4], 16)
            df_players = [p for p in h_squad if p[1] in ("DF", "MF")]
            player = df_players[h % len(df_players)][0] if df_players else "Player"
            minute = 10 + (h % 75)
            events.append({"minute": minute, "team": home_team, "type": "Card", "detail": f"🟨 Yellow Card: {player} ({minute}')"})
            
        a_yellow = int(hashlib.md5(f"yellow_a_{away_team}".encode()).hexdigest()[:2], 16) % 3
        for i in range(a_yellow):
            h = int(hashlib.md5(f"card_a_{away_team}_{i}".encode()).hexdigest()[:4], 16)
            df_players = [p for p in a_squad if p[1] in ("DF", "MF")]
            player = df_players[h % len(df_players)][0] if df_players else "Player"
            minute = 10 + (h % 75)
            events.append({"minute": minute, "team": away_team, "type": "Card", "detail": f"🟨 Yellow Card: {player} ({minute}')"})
            
        return sorted(events, key=lambda x: x["minute"])

    for i in range(hs):
        h = int(hashlib.md5(f"goal_h_{home_team}_{i}".encode()).hexdigest()[:4], 16)
        fw_players = [p for p in h_squad if p[1] in ("FW", "MF")]
        scorer = fw_players[h % len(fw_players)][0] if fw_players else "Player"
        minute = 5 + (h % 80)
        events.append({"minute": minute, "team": home_team, "type": "Goal", "detail": f"⚽ Goal: {scorer} ({minute}')"})
        
    for i in range(as_):
        h = int(hashlib.md5(f"goal_a_{away_team}_{i}".encode()).hexdigest()[:4], 16)
        fw_players = [p for p in a_squad if p[1] in ("FW", "MF")]
        scorer = fw_players[h % len(fw_players)][0] if fw_players else "Player"
        minute = 5 + (h % 80)
        events.append({"minute": minute, "team": away_team, "type": "Goal", "detail": f"⚽ Goal: {scorer} ({minute}')"})
        
    h_yellow = int(hashlib.md5(f"yellow_h_{home_team}".encode()).hexdigest()[:2], 16) % 3
    for i in range(h_yellow):
        h = int(hashlib.md5(f"card_h_{home_team}_{i}".encode()).hexdigest()[:4], 16)
        df_players = [p for p in h_squad if p[1] in ("DF", "MF")]
        player = df_players[h % len(df_players)][0] if df_players else "Player"
        minute = 10 + (h % 75)
        events.append({"minute": minute, "team": home_team, "type": "Card", "detail": f"🟨 Yellow Card: {player} ({minute}')"})
        
    a_yellow = int(hashlib.md5(f"yellow_a_{away_team}".encode()).hexdigest()[:2], 16) % 3
    for i in range(a_yellow):
        h = int(hashlib.md5(f"card_a_{away_team}_{i}".encode()).hexdigest()[:4], 16)
        df_players = [p for p in a_squad if p[1] in ("DF", "MF")]
        player = df_players[h % len(df_players)][0] if df_players else "Player"
        minute = 10 + (h % 75)
        events.append({"minute": minute, "team": away_team, "type": "Card", "detail": f"🟨 Yellow Card: {player} ({minute}')"})
        
    return sorted(events, key=lambda x: x["minute"])


def render_evidence_ui(pred, home_team, away_team, hs=None, as_=None, scorers=None):
    """Common UI renderer for both real and simulated matches."""
    col_card, col_radar, col_bar = st.columns([1.1, 1, 1.2], gap="large")
    
    winner = get_winner_label(pred["predicted_outcome"], home_team, away_team)
    conf_label, conf_emoji = get_confidence_label(pred["confidence"])
    conf_cls = "conf-high" if conf_label == "High" else ("conf-medium" if conf_label == "Medium" else "conf-low")
    
    # Establish scores for stats/lineups
    if hs is None:
        hs = round(pred.get("home_goals_pg", 1))
    if as_ is None:
        as_ = round(pred.get("away_goals_pg", 1))

    with col_card:
        st.markdown(f"""
        <div class="glass-card" style="height:100%; display:flex; flex-direction:column; justify-content:center;">
            <div style="text-align:center; margin-bottom:12px;">
                <span class="round-badge" style="background:rgba(52,211,153,0.1); color:#34d399; border:1px solid rgba(52,211,153,0.25);">
                    {pred['round'].upper()} Match Analysis
                </span>
            </div>
            <div class="team-vs-block" style="margin:16px 0;">
                <div class="team-block">
                    <div class="team-name" style="font-size:1.1rem; font-weight:800;">{format_team_html(home_team)}</div>
                    <div class="team-rank">Rank #{get_ranking(home_team)}</div>
                </div>
                <div class="vs-separator" style="font-size:1rem; color:#6b7280;">{hs} - {as_}</div>
                <div class="team-block">
                    <div class="team-name" style="font-size:1.1rem; font-weight:800;">{format_team_html(away_team)}</div>
                    <div class="team-rank">Rank #{get_ranking(away_team)}</div>
                </div>
            </div>
            <div style="text-align:center; border-top:1px solid var(--border-subtle); padding-top:14px;">
                <div style="font-size:0.68rem; color:#6b7280; text-transform:uppercase; letter-spacing:1px; margin-bottom:4px;">AI PREDICTED OUTCOME</div>
                <div class="winner-tag" style="margin:4px 0; font-size:1.4rem;">{format_team_html(winner)}</div>
                <span class="conf-badge {conf_cls}" style="margin-top:4px;">{conf_emoji} {conf_label} Confidence ({pred['confidence']:.1%})</span>
            </div>
        </div>""", unsafe_allow_html=True)
        
    with col_radar:
        all_stats = get_team_stats(wc_df)
        h_s = all_stats.loc[home_team].to_dict() if home_team in all_stats.index else {"win_rate": 0.33, "goals_scored_per_game": 1.0, "goals_conceded_per_game": 1.0}
        a_s = all_stats.loc[away_team].to_dict() if away_team in all_stats.index else {"win_rate": 0.33, "goals_scored_per_game": 1.0, "goals_conceded_per_game": 1.0}
        
        st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#34d399; letter-spacing:1px; text-align:center; margin-bottom:10px;">📊 RADAR STAT COMPARISON</div>', unsafe_allow_html=True)
        fig_radar = make_radar_chart(home_team, away_team, h_s, a_s)
        st.plotly_chart(fig_radar, use_container_width=True)
        
    with col_bar:
        st.markdown('<div style="font-size:0.8rem; font-weight:700; color:#34d399; letter-spacing:1px; margin-bottom:10px;">📈 PROBABILITY DISTRIBUTION</div>', unsafe_allow_html=True)
        hp = pred["home_win_prob"]
        dp = pred["draw_prob"] if not pred["is_knockout"] else 0
        ap = pred["away_win_prob"]
        
        st.markdown(f"""
        <div class="prob-meter-container">
            <div class="prob-meter-header">
                <span class="prob-meter-label">{format_team_html(home_team)} Win</span>
                <span class="prob-meter-value" style="color:#34d399;">{hp:.1%}</span>
            </div>
            <div class="prob-meter-track">
                <div class="prob-meter-fill home" style="width:{hp*100:.1f}%"></div>
            </div>
        </div>""", unsafe_allow_html=True)

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

        st.markdown(f"""
        <div class="prob-meter-container">
            <div class="prob-meter-header">
                <span class="prob-meter-label">{format_team_html(away_team)} Win</span>
                <span class="prob-meter-value" style="color:#f87171;">{ap:.1%}</span>
            </div>
            <div class="prob-meter-track">
                <div class="prob-meter-fill away" style="width:{ap*100:.1f}%"></div>
            </div>
        </div>""", unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="evidence-card" style="margin-top:14px; background:rgba(52,211,153,0.05); border-color:rgba(52,211,153,0.2);">
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

    # ── Match Performance Center (Complicated stats, lineups, timeline) ───────
    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🏟️ Live Match Center (Lineups &amp; Stats)</div>', unsafe_allow_html=True)
    
    h_squad = get_squad(home_team)
    a_squad = get_squad(away_team)
    
    tab_stats, tab_lineups, tab_events = st.tabs([
        "📊 Detailed Team Statistics", 
        "📋 Squad Lineups & Ratings", 
        "⏱️ Chronological Match Timeline"
    ])
    
    with tab_stats:
        from ml.features import _get_or_estimate_match_stats
        m_stats = _get_or_estimate_match_stats(home_team, away_team, hs, as_, "2026-06-27")
        h_perf = m_stats.get(home_team, {})
        a_perf = m_stats.get(away_team, {})
        
        def render_stat_row(label, home_val, away_val, is_pct=False):
            try:
                h_num = float(str(home_val).replace("%", "")) if is_pct else float(home_val)
                a_num = float(str(away_val).replace("%", "")) if is_pct else float(away_val)
                total = h_num + a_num
                h_pct = (h_num / total * 100) if total > 0 else 50
                a_pct = 100 - h_pct
            except Exception:
                h_pct = 50
                a_pct = 50
                
            return f"""
            <div style="margin:14px 0;">
                <div style="display:flex; justify-content:space-between; font-size:0.82rem; font-weight:700; color:#e5e7eb; margin-bottom:4px;">
                    <span style="color:#34d399;">{home_val}</span>
                    <span style="color:#9ca3af; text-transform:uppercase; font-size:0.68rem; letter-spacing:1px;">{label}</span>
                    <span style="color:#f87171;">{away_val}</span>
                </div>
                <div style="height:6px; display:flex; border-radius:3px; overflow:hidden; background:rgba(255,255,255,0.06);">
                    <div style="width:{h_pct}%; background:#34d399;"></div>
                    <div style="width:{a_pct}%; background:#f87171;"></div>
                </div>
            </div>"""
            
        st.markdown('<div class="glass-card" style="padding:16px 20px; border-radius:12px;">', unsafe_allow_html=True)
        st.markdown(render_stat_row("Ball Possession", h_perf.get("Ball Possession", "50%"), a_perf.get("Ball Possession", "50%"), is_pct=True), unsafe_allow_html=True)
        st.markdown(render_stat_row("Shots", h_perf.get("Total Shots", 10), a_perf.get("Total Shots", 10)), unsafe_allow_html=True)
        st.markdown(render_stat_row("Shots on Target", h_perf.get("Shots on Goal", 4), a_perf.get("Shots on Goal", 4)), unsafe_allow_html=True)
        st.markdown(render_stat_row("Total Passes", h_perf.get("Total passes", 450), a_perf.get("Total passes", 450)), unsafe_allow_html=True)
        st.markdown(render_stat_row("Pass Accuracy", h_perf.get("Passes %", "82%"), a_perf.get("Passes %", "82%"), is_pct=True), unsafe_allow_html=True)
        st.markdown(render_stat_row("Fouls Commited", h_perf.get("Fouls", 11), a_perf.get("Fouls", 11)), unsafe_allow_html=True)
        st.markdown(render_stat_row("Yellow Cards", h_perf.get("Yellow Cards", 1), a_perf.get("Yellow Cards", 1)), unsafe_allow_html=True)
        st.markdown(render_stat_row("Red Cards", h_perf.get("Red Cards", 0), a_perf.get("Red Cards", 0)), unsafe_allow_html=True)
        st.markdown(render_stat_row("Offsides", h_perf.get("Offsides", 1), a_perf.get("Offsides", 1)), unsafe_allow_html=True)
        st.markdown(render_stat_row("Corner Kicks", h_perf.get("Corner Kicks", 4), a_perf.get("Corner Kicks", 4)), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_lineups:
        col_l1, col_l2 = st.columns(2)
        
        def render_squad_lineup(team, squad, is_home, scorers=None):
            theme_color = "#34d399" if is_home else "#f87171"
            html = f"""<div style="background:rgba(17,24,39,0.95); padding:16px; border-radius:12px; border:1px solid rgba(255,255,255,0.08);">
<div style="font-weight:800; font-size:1.0rem; color:{theme_color}; margin-bottom:12px; display:flex; justify-content:space-between; align-items:center;">
<span>{get_team_flag(team)} {team} Squad</span>
<span style="font-size:0.75rem; color:#6b7280; font-weight:normal;">Formation: 4-3-3</span>
</div>
<table style="width:100%; border-collapse:collapse; text-align:left;">
<thead>
<tr style="border-bottom:1px solid rgba(255,255,255,0.08); font-size:0.68rem; color:#9ca3af; text-transform:uppercase;">
<th style="padding:6px 0;">Pos</th>
<th style="padding:6px 0;">Player</th>
<th style="padding:6px 0; text-align:center;">Rating</th>
<th style="padding:6px 0; text-align:right;">Performance Stats</th>
</tr>
</thead>
<tbody>"""
            for name, pos, rating in squad:
                p_rating, p_details = get_player_match_stats(name, pos, rating, is_home, hs, as_, scorers=scorers, home_team=home_team, away_team=away_team)
                html += f"""<tr style="border-bottom:1px solid rgba(255,255,255,0.04); font-size:0.8rem; color:#f3f4f6;">
<td style="padding:8px 0; font-weight:700; color:#9ca3af;">{pos}</td>
<td style="padding:8px 0; font-weight:600;">{name}</td>
<td style="padding:8px 0; text-align:center; font-weight:800; color:{theme_color};">{p_rating}</td>
<td style="padding:8px 0; text-align:right; font-size:0.72rem; color:#9ca3af;">{p_details}</td>
</tr>"""
            html += "</tbody></table></div>"
            return html
            
        with col_l1:
            st.markdown(render_squad_lineup(home_team, h_squad, is_home=True, scorers=scorers), unsafe_allow_html=True)
        with col_l2:
            st.markdown(render_squad_lineup(away_team, a_squad, is_home=False, scorers=scorers), unsafe_allow_html=True)

    with tab_events:
        events = generate_timeline(home_team, away_team, hs, as_, h_squad, a_squad, scorers=scorers)
        if not events:
            st.info("No match events (goals or cards) registered in this simulation.")
        else:
            html = """<div style="background:rgba(17,24,39,0.95); padding:20px; border-radius:12px; border:1px solid rgba(255,255,255,0.08);">
<div style="font-weight:800; font-size:0.95rem; color:#f3f4f6; margin-bottom:16px;">Chronological Match Events</div>
<div style="position:relative; padding-left:24px; border-left:2px solid rgba(255,255,255,0.08); margin-left:12px;">"""
            for ev in events:
                accent = "#34d399" if ev["team"] == home_team else "#f87171"
                bullet_style = f"background:{accent}; box-shadow:0 0 8px {accent};"
                html += f"""<div style="position:relative; margin-bottom:18px;">
<div style="position:absolute; left:-31px; top:4px; width:12px; height:12px; border-radius:50%; border:2px solid #111827; {bullet_style}"></div>
<div style="font-size:0.7rem; font-weight:700; color:#6b7280;">{ev['minute']}' MINUTE</div>
<div style="font-size:0.85rem; font-weight:600; color:#f3f4f6; margin-top:2px;">{ev['detail']}</div>
<div style="font-size:0.72rem; color:#9ca3af;">{ev['team']}</div>
</div>"""
            html += "</div></div>"
            st.markdown(html, unsafe_allow_html=True)

    # ── Form and Feature Contributions Row ───────────────────────────────────
    st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)
    col_form, col_cont = st.columns([1, 1.2], gap="large")
    
    with col_form:
        st.markdown('<div class="section-header" style="margin-top:0;">⚡ Team Form &amp; Trajectory</div>', unsafe_allow_html=True)
        
        def render_form_timeline(team, form_data):
            html = f'<div style="background:rgba(17,24,39,0.95); padding:16px; border-radius:12px; border:1px solid rgba(255,255,255,0.08); margin-bottom:12px;">'
            html += f'<div style="font-weight:700; font-size:0.9rem; color:#f3f4f6; margin-bottom:8px;">{get_team_flag(team)} {team} Recent WC Games</div>'
            html += '<div style="display:grid; grid-template-columns: repeat(5, 1fr); gap:8px;">'
            for game in form_data:
                bg = "#10b981" if game["result"] == "W" else "#ef4444" if game["result"] == "L" else "#6b7280"
                html += f"""
                <div style="text-align:center; background:rgba(31,41,55,0.9); padding:8px; border-radius:8px; border:1px solid rgba(255,255,255,0.08);">
                    <div style="font-size:0.6rem; color:#6b7280;">{game['year']}</div>
                    <span style="
                        display:inline-flex; align-items:center; justify-content:center;
                        width:24px; height:24px; border-radius:50%;
                        font-size:0.8rem; font-weight:800; color:white;
                        background:{bg}; margin:6px 0;
                    ">{game['result']}</span>
                    <div style="font-size:0.58rem; font-weight:700; color:#9ca3af; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">{game['opponent']}</div>
                    <div style="font-size:0.6rem; color:#6b7280;">{game['score']}</div>
                </div>"""
            html += '</div></div>'
            return html

        st.markdown(render_form_timeline(home_team, pred["home_form"]), unsafe_allow_html=True)
        st.markdown(render_form_timeline(away_team, pred["away_form"]), unsafe_allow_html=True)
        
    with col_cont:
        st.markdown('<div class="section-header" style="margin-top:0;">🤖 Model Feature Drivers (SHAP-style)</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.75rem; color:#9ca3af; margin-top:-10px; margin-bottom:12px;">Individual feature weights showing how strongly each attribute pushes the model prediction.</p>', unsafe_allow_html=True)
        
        for c in pred["contributions"]:
            arrow = "→" if c["direction"] == "home" else "←" if c["direction"] == "away" else "⚖️"
            col_accent = "#34d399" if c["direction"] == "home" else "#f87171" if c["direction"] == "away" else "#6b7280"
            chip_cls = "chip-positive" if c["type"] == "positive" else "chip-negative" if c["type"] == "negative" else "chip-neutral"
            
            st.markdown(f"""
            <div style="
                display:flex; justify-content:space-between; align-items:center; 
                background:rgba(255,255,255,0.02); border:1px solid var(--border-subtle);
                border-radius:10px; padding:10px 14px; margin:6px 0;
            ">
                <div>
                    <span class="factor-chip {chip_cls}">{c['factor']}</span>
                    <div style="font-size:0.75rem; color:#6b7280; margin-top:4px;">{c['detail']}</div>
                </div>
                <div style="text-align:right; min-width:80px;">
                    <div style="font-size:0.95rem; font-weight:800; color:{col_accent};">{arrow} {c['strength']:.1%}</div>
                    <div style="font-size:0.58rem; color:#6b7280; text-transform:uppercase;">Influence</div>
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
        h2h_df["home_team"] = h2h_df["home_team"].apply(format_team_emoji)
        h2h_df["away_team"] = h2h_df["away_team"].apply(format_team_emoji)
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
    fixture_options = fixtures[
        (fixtures["home_team"] != "TBD") & (fixtures["away_team"] != "TBD")
    ].copy()
    
    if fixture_options.empty:
        st.warning("No scheduled matches available with confirmed teams. Try the Custom Matchup Simulator!")
    else:
        today = pd.Timestamp.now(tz=None).normalize()
        fixture_options["date_sort"] = pd.to_datetime(fixture_options["date"], errors="coerce")
        status_priority = {"live": 0, "scheduled": 1, "completed": 2}
        fixture_options["status_priority"] = fixture_options["status"].map(status_priority).fillna(9)
        fixture_options["round_priority"] = fixture_options["round"].apply(lambda r: 0 if r == "Round of 32" else 1)
        fixture_options["future_priority"] = fixture_options["date_sort"].apply(
            lambda d: 0 if pd.notna(d) and d >= today else 1
        )
        fixture_options["completed_sort"] = fixture_options["date_sort"].fillna(pd.Timestamp("1900-01-01"))
        fixture_options = fixture_options.sort_values(
            ["status_priority", "round_priority", "future_priority", "date_sort", "match_id"],
            ascending=[True, True, True, True, True],
        )

        has_active_r32 = (
            fixture_options["status"].isin(["live", "scheduled"]) &
            (fixture_options["round"] == "Round of 32")
        ).any()
        completed_r32_mask = (
            (fixture_options["round"] == "Round of 32") &
            (fixture_options["status"] == "completed")
        )
        if not has_active_r32 and completed_r32_mask.any():
            completed_r32 = fixture_options[completed_r32_mask].sort_values(
                ["completed_sort", "match_id"], ascending=[False, False]
            )
            fixture_options = pd.concat([
                completed_r32,
                fixture_options.drop(index=completed_r32.index)
            ])

        def _fixture_label(r):
            status = str(r.get("status", "scheduled")).upper()
            score = ""
            if pd.notna(r.get("home_score")) and pd.notna(r.get("away_score")):
                score = f" - {int(r['home_score'])}-{int(r['away_score'])}"
                if pd.notna(r.get("home_penalty_score")) and pd.notna(r.get("away_penalty_score")):
                    score += f" pens {int(r['home_penalty_score'])}-{int(r['away_penalty_score'])}"
            return f"{status} - {r['round']} - {r['home_team']} vs {r['away_team']}{score} - {r['date']}"

        fixture_options["display_name"] = fixture_options.apply(
            lambda r: f"{r['round']} — {r['home_team']} vs {r['away_team']} ({r['date']})", axis=1
        )
        
        selected_match_name = st.selectbox(
            "Select a 2026 World Cup Fixture to Analyze",
            options=fixture_options["display_name"].tolist()
        )
        
        selected_row = fixture_options[fixture_options["display_name"] == selected_match_name].iloc[0]
        home_t = selected_row["home_team"]
        away_t = selected_row["away_team"]
        round_t = selected_row["round"]
        
        pred_data = predict_match_with_evidence(
            home_team=home_t,
            away_team=away_t,
            round_name=round_t,
            wc_df=wc_df,
        )
        pred_data["round"] = round_t
        
        hs_actual = selected_row.get("home_score")
        as_actual = selected_row.get("away_score")
        hs_val = int(hs_actual) if hs_actual is not None and not pd.isna(hs_actual) else None
        as_val = int(as_actual) if as_actual is not None and not pd.isna(as_actual) else None
        
        scorers_val = selected_row.get("scorers") if "scorers" in selected_row else None
        render_evidence_ui(pred_data, home_t, away_t, hs=hs_val, as_=as_val, scorers=scorers_val)

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
        render_evidence_ui(pred_sim, sim_home, sim_away, hs=round(home_goals), as_=round(away_goals))
