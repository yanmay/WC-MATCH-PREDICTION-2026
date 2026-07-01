import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import textwrap

ROOT = Path(__file__).parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import load_data, get_model_and_metrics, map_feature_name, CSS, render_sidebar

def render_view():
    st.markdown('<div class="hero-badge">🤖 MODEL INSIGHTS & ABILITIES</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-title" style="font-size:2.5rem; line-height:1.2;">AI Model Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Understand how the AI model uses team rankings, goals, and historic features to compute probabilities.</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    with st.spinner("Loading metrics database..."):
        wc_df, fixtures = load_data()
        pipeline, model_metrics = get_model_and_metrics()

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
        <div class="glass-card" style="margin-bottom:16px; border-left:4px solid #3b82f6; background:rgba(59,130,246,0.01); padding:16px; border-radius:8px;">
            <div style="font-weight:700; color:#3b82f6; font-size:0.95rem; margin-bottom:6px;">📊 How Team Stats Affect Match Predictions</div>
            <div style="font-size:0.8rem; color:#9ca3af; line-height:1.5;">
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
            <div style="font-size:0.8rem;font-weight:700;color:#3b82f6;margin-bottom:16px;letter-spacing:1px;">
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
