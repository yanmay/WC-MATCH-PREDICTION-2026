import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import plotly.express as px

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import CSS, render_sidebar

def render_view():
    st.markdown('<div class="hero-badge">⚽ HISTORICAL WORLD CUP DATABASE</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:2.5rem; line-height:1.2;">World Cup History</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Explore historical statistics, champions, and performance distributions from past tournaments.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    c_chart, c_table = st.columns([1, 1], gap="large")

    with c_chart:
        st.markdown("""
        <div class="glass-card" style="margin-bottom:16px; border-left:4px solid #facc15; background:rgba(250,204,21,0.01); padding:16px; border-radius:8px;">
            <div style="font-weight:700; color:#facc15; font-size:0.95rem; margin-bottom:6px;">🏆 World Cup Title Distribution</div>
            <div style="font-size:0.8rem; color:#9ca3af; line-height:1.5;">
                Brazil holds the record for the most tournament wins with 5 titles, closely followed by Germany and Italy with 4 titles each.
            </div>
        </div>
        """, unsafe_allow_html=True)

        champions_data = pd.DataFrame([
            {"Country": "Brazil", "Titles": 5},
            {"Country": "Germany", "Titles": 4},
            {"Country": "Italy", "Titles": 4},
            {"Country": "Argentina", "Titles": 3},
            {"Country": "France", "Titles": 2},
            {"Country": "Uruguay", "Titles": 2},
            {"Country": "England", "Titles": 1},
            {"Country": "Spain", "Titles": 1}
        ])

        fig = px.bar(
            champions_data,
            x="Titles",
            y="Country",
            orientation="h",
            color="Titles",
            color_continuous_scale="Viridis",
            labels={"Titles": "Total Champions Titles", "Country": "National Team"}
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f3f4f6", size=11, family="Inter, sans-serif"),
            title_font=dict(color="#f3f4f6", size=14, weight="bold"),
            height=340,
            margin=dict(l=10, r=20, t=20, b=10),
            xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", tickfont=dict(color="#9ca3af"), zeroline=False),
            yaxis=dict(showgrid=False, tickfont=dict(color="#f3f4f6")),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with c_table:
        st.markdown("""
        <div class="glass-card" style="padding: 16px;">
            <div style="font-size:0.85rem; font-weight:700; color:#facc15; margin-bottom:12px;">🌍 Recent Editions & Champions</div>
            <table style="width:100%; border-collapse:collapse; font-size:0.8rem; color:#f3f4f6;">
                <thead>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.1); text-align:left;">
                        <th style="padding:8px 4px; color:#9ca3af;">Year</th>
                        <th style="padding:8px 4px; color:#9ca3af;">Host</th>
                        <th style="padding:8px 4px; color:#9ca3af;">Winner</th>
                        <th style="padding:8px 4px; color:#9ca3af;">Runner-up</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="padding:8px 4px; font-weight:700;">2022</td>
                        <td style="padding:8px 4px;">Qatar</td>
                        <td style="padding:8px 4px; color:#34d399; font-weight:700;">🇦🇷 Argentina</td>
                        <td style="padding:8px 4px;">🇫🇷 France</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="padding:8px 4px; font-weight:700;">2018</td>
                        <td style="padding:8px 4px;">Russia</td>
                        <td style="padding:8px 4px; color:#34d399; font-weight:700;">🇫🇷 France</td>
                        <td style="padding:8px 4px;">🇭🇷 Croatia</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="padding:8px 4px; font-weight:700;">2014</td>
                        <td style="padding:8px 4px;">Brazil</td>
                        <td style="padding:8px 4px; color:#34d399; font-weight:700;">🇩🇪 Germany</td>
                        <td style="padding:8px 4px;">🇦🇷 Argentina</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="padding:8px 4px; font-weight:700;">2010</td>
                        <td style="padding:8px 4px;">South Africa</td>
                        <td style="padding:8px 4px; color:#34d399; font-weight:700;">🇪🇸 Spain</td>
                        <td style="padding:8px 4px;">🇳🇱 Netherlands</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="padding:8px 4px; font-weight:700;">2006</td>
                        <td style="padding:8px 4px;">Germany</td>
                        <td style="padding:8px 4px; color:#34d399; font-weight:700;">🇮🇹 Italy</td>
                        <td style="padding:8px 4px;">🇫🇷 France</td>
                    </tr>
                    <tr style="border-bottom:1px solid rgba(255,255,255,0.05);">
                        <td style="padding:8px 4px; font-weight:700;">2002</td>
                        <td style="padding:8px 4px;">South Korea/Japan</td>
                        <td style="padding:8px 4px; color:#34d399; font-weight:700;">🇧🇷 Brazil</td>
                        <td style="padding:8px 4px;">🇩🇪 Germany</td>
                    </tr>
                </tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)
