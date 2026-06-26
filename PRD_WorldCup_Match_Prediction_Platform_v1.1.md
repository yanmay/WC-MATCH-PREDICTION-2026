# Product Requirements Document
## Match-Prediction Analytics Platform for the FIFA World Cup

**Document owner:** Product Management
**Status:** Draft v1.1 *(updated: tech stack locked to Python / scikit-learn / Streamlit; live-update pipeline scoped)*
**Last updated:** June 26, 2026

---

## 1. Context

The 2026 FIFA World Cup is currently underway — the first 48-team edition, played across 16 cities in the US, Mexico, and Canada, running June 11 to July 19. The group stage has just concluded and the tournament is moving into the new Round of 32, then Round of 16, quarterfinals, semifinals, and final. This expanded format (104 matches vs. 64 in 2022) creates more matches to predict, more data points to model, and more moments where fans, media, and analysts want a credible probability — not just a gut-feel pick.

This PRD defines a **data science SaaS platform** that generates match-outcome predictions for the current World Cup, trained on historical World Cup data, using a **Python + scikit-learn ML pipeline** with a **Streamlit-based frontend**.

---

## 2. Product Summary

**What it is:** A web-based analytics platform that predicts the **winner** (or draw) of each remaining World Cup match, using models trained on decades of historical World Cup results, team statistics, and contextual factors (host advantage, rest days, squad strength, etc.). Predictions are **automatically refreshed after every completed match**, so the model stays current as the tournament evolves.

**Tech stack (locked):**

| Layer | Technology | Role |
|---|---|---|
| **ML / Modelling** | Python 3.11+, scikit-learn | Model training, prediction, evaluation |
| **Data pipeline** | Python (pandas, requests/httpx) | Historical data prep, live result ingestion, feature engineering |
| **Frontend / UI** | Streamlit | Dashboard, match detail views, accuracy tracker — all served as a Streamlit app |
| **Model persistence** | joblib / pickle | Serialize trained models for fast prediction without retraining from scratch |
| **Hosting** | Streamlit Community Cloud (MVP) → cloud VM / container post-MVP | Deployment target for the Streamlit app |

**One-line pitch:** "See the odds before the whistle blows — data-driven World Cup predictions, not punditry."

---

## 3. Problem Statement

Football fans, fantasy/prediction-pool players, sports content creators, and casual bettors currently rely on:
- Pundit opinion and gut feel, which is inconsistent and unaccountable.
- Bookmaker odds, which embed margin (vig) and aren't explained or educational.
- Scattered free stats sites with no predictive layer — they show *what happened*, not *what's likely to happen*.

There is no accessible, transparent, explainable tool that turns 90+ years of World Cup history into a clear, current-tournament prediction product for non-technical users.

**The gap we fill:** a tool that is *more rigorous than punditry*, *more transparent than bookmaker odds*, and *easier to use than building your own model in a notebook.*

---

## 4. Target Users (Personas)

| Persona | Description | Core need |
|---|---|---|
| **Casual Fan "Maria"** | Watches the World Cup every 4 years, joins office prediction pools | Simple, confident predictions she can act on in seconds |
| **Prediction-Pool Power User "Dev"** | Runs or competes seriously in bracket pools | Match-by-match probabilities, bracket simulation, accuracy tracking |
| **Sports Content Creator "Jay"** | YouTuber/blogger covering the tournament | Shareable visuals, exportable charts, narrative-ready stats |
| **Data-Curious Analyst "Priya"** | Has technical background, wants to see *why* | Feature importance, model confidence, historical backtests |

The MVP should be built primarily for **Maria and Dev** (highest volume, clearest willingness to pay during the tournament window), with **Priya**'s transparency needs satisfied as a secondary layer rather than a separate product.

---

## 5. Goals

### Business goals
- Capture tournament-driven demand spikes (search and social interest in World Cup predictions peaks sharply around each matchday).
- Convert free users to paid tiers via bracket/pool features and deeper insight tools.
- Establish a reusable platform that can be revived for Euros, Copa América, and future World Cups (4-year reactivation engine).

### User goals
- Get a clear, justified prediction for any upcoming match in under 10 seconds.
- Track how earlier predictions have performed (build trust over time).
- Use predictions to make decisions in pools, fantasy leagues, or conversation.

---

## 6. Core Features (MoSCoW)

### MUST-HAVE (MVP-critical)

| Feature | Description | Tech notes |
|---|---|---|
| **Match Winner Prediction Engine** | scikit-learn model outputting Win/Draw/Loss probabilities for each scheduled match, trained on historical World Cup data. **Primary output is the predicted winner** (highest-probability outcome), displayed alongside the full probability split. Start with Logistic Regression and Random Forest as interpretable baselines; promote to Gradient Boosting (XGBoost/LightGBM) if accuracy warrants. | Python, scikit-learn, joblib |
| **Post-Game Model Update Pipeline** | After every completed match, the pipeline automatically (1) ingests the result, (2) appends it to the running dataset, (3) re-evaluates model calibration, and (4) re-serves updated predictions for remaining fixtures. This is the key freshness mechanism — predictions are never stale by more than one completed game. | Python scheduler (APScheduler or cron), pandas, scikit-learn `.fit()` / `.predict()` |
| **Upcoming Matches Dashboard (Streamlit)** | Streamlit page listing all remaining 2026 fixtures in chronological order. Each row shows: predicted winner, Win/Draw/Loss probability bars, and a confidence indicator. Automatically reflects the latest model state after each post-game pipeline run. | Streamlit (`st.dataframe`, `st.progress`, `st.metric`) |
| **Match Detail View (Streamlit)** | Per-match Streamlit page (routed via `st.selectbox` or sidebar navigation): full probability breakdown, top contributing features (feature importances from the trained model), head-to-head historical record. | Streamlit, matplotlib / plotly for charts |
| **Accuracy Tracker (Streamlit)** | Running table of all predictions made vs. actual outcomes, refreshed post-game. Displays overall accuracy %, accuracy by round, and calibration check (does a 70% prediction win ~70% of the time?). Public-facing — the platform's core trust mechanism. | Streamlit, pandas |
| **User Accounts & Onboarding** | Sign-up, login, basic preference capture (favorite team, interest level). | Can use Streamlit-Authenticator for MVP-speed auth |
| **Tiered Access (Free / Paid)** | Free: upcoming match predictions with limited detail. Paid: full feature importance panel, bracket simulator (V1.1), exports. Paywall enforced at the session level via auth state in Streamlit. | Streamlit session state, payment via Stripe link (external) |
| **Mobile-Responsive Streamlit App** | Streamlit is inherently responsive, but layouts must be tested on mobile widths given that most matchday traffic is on phones. Use `st.columns` conservatively; avoid wide tables without horizontal scroll. | Streamlit responsive layout guidelines |

### NICE-TO-HAVE (post-MVP / V1.x+)

| Feature | Description |
|---|---|
| **Bracket Simulator** | Monte Carlo simulation of remaining knockout rounds (thousands of iterations) to estimate each team's title probability. Runs via a Python backend function, displayed in a Streamlit results panel. |
| **Score-line Prediction** | Move beyond W/D/L to predicting exact or most-likely scorelines using Poisson regression (easily added in scikit-learn / statsmodels). |
| **Shareable Visual Cards** | Auto-generated social-shareable PNG/SVG prediction cards per match, generated via matplotlib and downloadable from Streamlit (`st.download_button`). |
| **Personalized "My Team" Feed** | Streamlit session-based filtering to show a user's followed team's fixtures and prediction history. |
| **Model Explainability Panel (SHAP)** | SHAP-value breakdown for the Priya persona; add `shap` to the Python requirements and render force plots in Streamlit via `st.pyplot`. |
| **Prediction Pool/League Integration** | Let groups compete against the model and each other via a shared leaderboard Streamlit page. |
| **API Access** | Expose the scikit-learn prediction pipeline as a FastAPI endpoint for a paid developer tier; decouple the Streamlit UI from the model serving layer. |
| **Historical "What-If" Sandbox** | Let users test the model against past World Cups (1994–2022) to build confidence in its accuracy pre-2026. |

### EXPLICITLY DEFERRED — see Section 9 (Out of Scope for V1)

---

## 7. User Flow: Onboarding → Insight

1. **Landing page (Streamlit)** — value prop ("World Cup predictions built on 90 years of data") + live snapshot of next match prediction (no login required, to hook visitors during searches).
2. **Sign-up** — Streamlit-Authenticator login; optional quick preference (favorite team, how closely they follow football) stored in session state.
3. **Home dashboard** — chronological list of upcoming matches with predicted winner badges and probability bars (e.g., "Argentina 58% / Draw 22% / Morocco 20%").
4. **Match detail drill-down** — select a match from a sidebar or dropdown to see the probability breakdown, key features driving the prediction, head-to-head history.
5. **Tournament view** — bracket/group visualization showing where predictions feed into the bigger picture (especially valuable during the Round-of-32 transition).
6. **Accuracy/trust check** — user can see "how did we do last matchday" on the Accuracy Tracker page before deciding whether to trust today's predictions — a deliberate trust-building step in the flow.
7. **Upgrade prompt** — contextual nudge to paid tier when a user hits a free-tier limit (e.g., trying to view feature importance panel or run a bracket simulation).
8. **Return loop** — push/email notification ahead of a followed team's next match, bringing the user back into the Streamlit dashboard.

---

## 8. MVP Definition

**Goal:** Ship something that can serve real predictions for the *remaining* matches of the current (2026) tournament — Round of 32 through the Final — since the group stage has already concluded.

**MVP includes:**
- Python scikit-learn pipeline: trained baseline model (Logistic Regression → Random Forest → optionally Gradient Boosting) on historical World Cup match data.
- **Post-game update script:** runs after every completed match to ingest the result, retrain/recalibrate, and push fresh predictions. This is non-negotiable for MVP — a static model that doesn't update after each game is not competitive.
- **Streamlit web app** with: Upcoming Matches dashboard, Match Detail view, Accuracy Tracker. Deployed to Streamlit Community Cloud for zero-infrastructure MVP.
- Basic auth (Streamlit-Authenticator) and a single paywall gate (free limited-detail view vs. paid full-detail knockout-stage view).
- Accuracy tracker updated automatically via the post-game pipeline.

**MVP excludes:** bracket simulator, scoreline prediction, notifications, API, social sharing graphics, SHAP panel — all pushed to V1.1+.

**Why this scope:** the tournament clock is the real deadline — a perfect model shipped after the Final is worthless. The MVP optimizes for *getting a credible, explainable, always-current prediction live before the Round of 32 kicks off*, not for feature completeness.

---

## 9. Out of Scope for Version 1 (Deliberately NOT Building)

- **Betting/wagering functionality of any kind** — no odds-style wagering, no integration with sportsbooks. This is an analytics product, not a gambling product, both for regulatory and brand-safety reasons.
- **Player-level prediction** (injuries, individual performance, top scorer modeling) — match-outcome/winner only in V1.
- **Other competitions** (club football, other international tournaments) — strictly World Cup-focused in V1 to keep data scope and model tractable.
- **Real-time in-match prediction updates** (live win-probability shifting minute-by-minute) — this requires live event-stream data and a much heavier real-time infra investment; deferred. The post-game update pipeline is the V1 freshness mechanism.
- **Native mobile apps** — mobile-responsive Streamlit web app only for V1; no iOS/Android builds.
- **Multi-language localization** — English-only at launch.
- **User-generated content / community features** (comments, forums) — not core to the prediction value prop yet.
- **Custom/user-trained models** — users consume the platform's model; they cannot upload their own data or tune model parameters in V1.

---

## 10. Success Metrics

| Category | Metric | Target signal |
|---|---|---|
| **Model quality** | Prediction accuracy (correct winner called) vs. actual outcomes; calibration quality | Outperform a naive baseline (always-pick-higher-ranked-team) by a meaningful margin |
| **Pipeline freshness** | Time between a match result being official and predictions updating | Target: updated predictions live within 30 minutes of final whistle |
| **Engagement** | Daily active users on matchdays vs. non-matchdays; return rate after a match result | Matchday DAU significantly higher than baseline; repeat visits across multiple rounds |
| **Trust** | % of users who view the Accuracy Tracker before/after viewing predictions | Indicates the trust loop is being used as designed |
| **Monetization** | Free-to-paid conversion rate at the knockout-stage paywall | Establishes whether the tiering point is correctly placed |
| **Retention beyond the tournament** | Email/account list size still active 30 days after the Final | Signals viability of reactivating the product for Euros 2028 / WC 2030 |

---

## 11. Key Assumptions & Open Questions

- **Data sourcing** is not yet finalized. The scikit-learn and Python pipeline is in place, but the specific historical World Cup datasets — match results, team stats, FIFA rankings — still need to be identified and licensing/usage terms confirmed (open datasets vs. a licensed sports-data provider). This is a **pre-MVP blocking dependency** and should be the very first engineering task.
- **Live result ingestion approach** (manual entry vs. automated API feed) is unresolved and directly determines how quickly the post-game update pipeline can fire after a result. An automated football data API (e.g., football-data.org, API-Football) is strongly preferred over manual entry.
- **Streamlit deployment constraints:** Streamlit Community Cloud is sufficient for MVP traffic but may throttle under high matchday load. Have a fallback plan (Heroku, Railway, or a small cloud VM) if traffic exceeds free-tier limits.
- **Model retraining strategy:** Full retraining after every game is computationally cheap for a scikit-learn model on a small dataset. If the dataset grows significantly in V1.1+, consider incremental learning or warm-starts.
- **Regulatory/compliance** review is needed before any feature resembling odds or betting language, even in V1.1+, especially given US/regional regulations on sports prediction products.

---

## 12. Risks

| Risk | Mitigation |
|---|---|
| Tournament ends before the platform reaches users | Ruthlessly scope MVP to ship before the Round of 32; treat the Final as a hard deadline |
| Post-game pipeline fails silently — stale predictions served without users knowing | Add a "last updated" timestamp prominently on the Streamlit dashboard; alert engineering if pipeline hasn't run within N hours of a match finishing |
| Model performs poorly on the 48-team format (no historical precedent for this exact structure) | Communicate confidence intervals, not false precision; wider uncertainty bands for novel scenarios |
| Users mistake this for a betting tool | Clear product framing, no odds/wagering language, explicit disclaimers — enforced in Streamlit UI copy |
| Streamlit performance issues on mobile under matchday traffic | Test mobile layouts early; cache heavy computations with `@st.cache_data`; have cloud VM fallback ready |
| Small/clean historical dataset limits model power | Start with simple, interpretable scikit-learn models; avoid overfitting on a small historical sample |

---

## 13. Appendix: Current Tournament Context (for reference)

The 2026 FIFA World Cup is the first 48-team edition, organized into 12 groups of 4, with the top 2 from each group plus the 8 best third-placed teams advancing to a new Round of 32 — an additional knockout round compared to prior tournaments. It runs June 11–July 19, 2026, hosted across 16 cities in the US, Mexico, and Canada, with the Final at MetLife/New York New Jersey Stadium. This context directly shapes MVP scope: the platform needs to handle a knockout bracket size and structure that has no exact historical precedent, which is itself a model and product communication challenge — worth designing for explicitly (e.g., wider confidence bands, explicit uncertainty language in the Streamlit UI for the new third-place qualification scenarios).
