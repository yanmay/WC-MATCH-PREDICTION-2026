# Technical Architecture Document
## FIFA World Cup Match Prediction Platform

**Role:** Senior Software Architect
**Companion to:** PRD v1.1 — Match-Prediction Analytics Platform for the FIFA World Cup
**Status:** Draft v1.0
**Last updated:** June 26, 2026

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Recommended Tech Stack with Reasoning](#2-recommended-tech-stack-with-reasoning)
3. [System Component Map](#3-system-component-map)
4. [Database Schema](#4-database-schema)
5. [ML Pipeline Architecture](#5-ml-pipeline-architecture)
6. [Data Ingestion & Post-Game Update Pipeline](#6-data-ingestion--post-game-update-pipeline)
7. [Backend API Design (FastAPI)](#7-backend-api-design-fastapi)
8. [Frontend Architecture (Streamlit)](#8-frontend-architecture-streamlit)
9. [Data Source Integrations](#9-data-source-integrations)
10. [Environment Variables & Configuration](#10-environment-variables--configuration)
11. [Security Architecture](#11-security-architecture)
12. [Deployment Architecture](#12-deployment-architecture)
13. [Development Setup & Local Environment](#13-development-setup--local-environment)
14. [Architectural Decision Log (ADRs)](#14-architectural-decision-log-adrs)

---

## 1. Architecture Overview

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        EXTERNAL DATA SOURCES                        │
│  football-data.org API  │  API-Football  │  Kaggle / Static CSVs   │
└────────────┬────────────────────┬──────────────────┬────────────────┘
             │                    │                  │
             ▼                    ▼                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER (Python)                   │
│  Historical ETL Script  │  Live Result Poller  │  APScheduler Cron  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        STORAGE LAYER                                │
│   PostgreSQL (primary)   │   Redis (cache)   │   S3 (ML artifacts) │
└──────────────┬───────────────────────────────────────────────────────┘
               │
     ┌─────────┴──────────┐
     ▼                    ▼
┌──────────┐    ┌───────────────────────────────┐
│  ML      │    │     BACKEND API (FastAPI)      │
│  Pipeline│    │  /matches  /predictions        │
│  scikit  │    │  /accuracy /auth /subscriptions│
│  -learn  │    └──────────────┬────────────────┘
└──────────┘                   │
                               ▼
               ┌───────────────────────────────┐
               │   FRONTEND (Streamlit App)     │
               │  Dashboard │ Match Detail      │
               │  Accuracy  │ Account/Upgrade   │
               └───────────────────────────────┘
                               │
                               ▼
               ┌───────────────────────────────┐
               │           END USER            │
               │     (Browser / Mobile Web)    │
               └───────────────────────────────┘
```

### 1.2 Guiding Architectural Principles

- **Python-first, always.** Every layer from data ingestion to the UI is Python. This eliminates context-switching costs and means any engineer on the team can debug end-to-end.
- **Simple beats clever at tournament speed.** The World Cup ends on July 19 — interpretable logistic regression shipped in 2 days beats a tuned XGBoost model shipped after the Final.
- **Predictions are never stale by more than one game.** The post-game pipeline is a first-class architectural concern, not an afterthought.
- **Separate concerns, but don't over-engineer.** FastAPI handles data + business logic; Streamlit handles UI. They talk over HTTP. This keeps the ML pipeline independently deployable and testable.

---

## 2. Recommended Tech Stack with Reasoning

### 2.1 Language & Runtime

| Technology | Version | Role | Reasoning |
|---|---|---|---|
| **Python** | 3.11+ | Everything | The de-facto standard for ML, data engineering, and increasingly for web APIs. 3.11 gives a meaningful speed improvement over 3.9/3.10. |

### 2.2 ML & Analytics Libraries

| Library | Version | Role | Reasoning |
|---|---|---|---|
| **scikit-learn** | ≥ 1.4 | Core ML pipeline (training, prediction, evaluation) | Mature, well-documented, has Pipeline API which bundles preprocessing + model into a single serializable artifact. Perfect for a dataset of ~900 historical matches. |
| **pandas** | ≥ 2.0 | Data wrangling, feature engineering, result ingestion | Industry standard. Handles the structured tabular sports data natively. |
| **numpy** | ≥ 1.26 | Numerical operations | Required by both scikit-learn and pandas; used directly for probability calculations. |
| **XGBoost / LightGBM** | ≥ 2.0 | Gradient boosting (V1.1+ upgrade path) | Both are scikit-learn compatible (fit into the same Pipeline object), so the baseline → upgrade path requires changing one line. Keep as a conditional import. |
| **SHAP** | ≥ 0.45 | Model explainability (V1.1+, Priya persona) | Industry standard for feature attribution. Works out-of-the-box with scikit-learn estimators. |
| **joblib** | ≥ 1.3 | Model serialization | Ships with scikit-learn; purpose-built for persisting large numpy arrays (model weights). Faster than pickle for ML objects. |
| **plotly** | ≥ 5.20 | Interactive charts in Streamlit | Native Streamlit integration via `st.plotly_chart`. Renders correctly on mobile. |
| **matplotlib** | ≥ 3.8 | Static chart generation for exports/cards | Fallback for server-side image generation where Plotly interactivity isn't needed. |
| **scipy** | ≥ 1.12 | Poisson distribution for scoreline prediction (V1.1+) | Lightweight, already a transitive dependency of scikit-learn. |

### 2.3 Data Ingestion Pipeline

| Library | Role | Reasoning |
|---|---|---|
| **httpx** | Async HTTP calls to football data APIs | Supports async out-of-the-box (unlike requests), which matters when polling APIs on a tight schedule. Drop-in requests replacement. |
| **APScheduler** | Cron scheduling for post-game pipeline | Pure Python, no Redis/Celery dependency needed for MVP. Can run inside the FastAPI process or as a standalone process. Supports cron expressions, interval triggers, and one-shot jobs. |
| **pydantic** | Data validation on API responses | Catches upstream data quality issues (e.g., API returning null scores) before they corrupt the database. |
| **tenacity** | Retry logic on API calls | Football data APIs occasionally return 429 (rate limit) or 503; exponential backoff prevents pipeline failures on transient errors. |

### 2.4 Storage Layer

| Technology | Role | Reasoning |
|---|---|---|
| **PostgreSQL 16** | Primary relational database | Best-in-class for structured sports data. JSONB support handles variable feature importance payloads without schema changes. Row-level locking is important during concurrent prediction writes. |
| **SQLAlchemy 2.0** | Python ORM | Async-compatible in 2.0+. Decouples the application from the database dialect. Type-safe with Python type hints. |
| **Alembic** | Database migrations | Works natively with SQLAlchemy. Critical for safely evolving the schema mid-tournament without data loss. |
| **Redis 7** | Cache layer | Caches rendered prediction payloads (probabilities, feature importances) so the API doesn't re-query Postgres on every Streamlit page refresh. 15-minute TTL aligned to the pipeline schedule. |
| **AWS S3** (or local filesystem for dev) | Model artifact storage | Serialized `.joblib` model files. S3 gives versioned, durable storage with a simple URL reference. Local filesystem is fine for MVP/dev. |
| **SQLite** | Development / testing only | Zero-config for local development. SQLAlchemy makes the switch to Postgres for staging/prod a single config line change. |

### 2.5 Backend API

| Technology | Role | Reasoning |
|---|---|---|
| **FastAPI** | REST API server | Async Python, auto-generates OpenAPI docs, native Pydantic integration, fastest Python web framework. Ideal for serving ML predictions where response time matters. |
| **Uvicorn** | ASGI server | Production-grade ASGI server, pairs with FastAPI. Handles concurrent requests without blocking during model inference. |
| **Gunicorn + Uvicorn workers** | Production process management | Standard production pattern: Gunicorn manages worker processes, each worker runs Uvicorn. |

### 2.6 Frontend Dashboard

| Technology | Role | Reasoning |
|---|---|---|
| **Streamlit ≥ 1.35** | Web UI / dashboard | Python-native, zero JS required. Multi-page support, mobile-responsive layouts, native Plotly/matplotlib integration. Allows the same team that builds the ML pipeline to ship the UI without context-switching. |
| **streamlit-authenticator** | MVP auth within Streamlit | YAML-config-based auth that adds login/logout UI to any Streamlit page in under 20 lines. Sufficient for MVP; swap for JWT-based auth in V1.1+ when FastAPI handles all auth. |
| **streamlit-extras** | UI enhancements | Provides additional components (`metric_cards`, `colored_header`, `stoggle`) that make the Streamlit UI look less default. |

### 2.7 Auth & Payments

| Technology | Role | Reasoning |
|---|---|---|
| **JWT (python-jose)** | Token-based auth (V1.1+, via FastAPI) | Stateless; works well across the Streamlit ↔ FastAPI boundary. |
| **bcrypt (passlib)** | Password hashing | Industry standard; bcrypt's intentional slowness resists brute-force attacks. |
| **Stripe** | Subscription payments | Best developer experience for recurring billing. Webhooks notify the backend of subscription status changes without polling. |

### 2.8 Infrastructure & DevOps

| Technology | Role | Reasoning |
|---|---|---|
| **Docker + Docker Compose** | Containerization | Packages the app + Postgres + Redis into a reproducible environment. `docker-compose.yml` for local dev; single Dockerfile per service for production. |
| **GitHub Actions** | CI/CD | Runs tests, linting, and deploys on push to `main`. Free for public repos; low overhead to set up. |
| **Streamlit Community Cloud** | MVP hosting (Streamlit frontend) | Free, zero-config deployment for Streamlit apps. Acceptable for MVP traffic; upgrade to dedicated VM at first sign of load issues. |
| **Railway / Render** | MVP hosting (FastAPI + Postgres) | Managed Postgres + Python service in one platform. Cheaper and simpler than AWS for MVP. |
| **pytest + pytest-asyncio** | Testing | Async test support for FastAPI routes. Use `pytest-mock` for mocking the football data API. |

---

## 3. System Component Map

```
worldcup-platform/
│
├── ingestion/                   # Data ingestion layer
│   ├── historical_etl.py        # One-time: load Kaggle/static CSV data into Postgres
│   ├── live_poller.py           # Polls football-data.org after each match
│   ├── scheduler.py             # APScheduler setup and job definitions
│   └── validators.py            # Pydantic models for API response validation
│
├── ml/                          # ML pipeline
│   ├── features.py              # Feature engineering functions
│   ├── train.py                 # Model training + evaluation
│   ├── predict.py               # Generate predictions for upcoming matches
│   ├── evaluate.py              # Accuracy + calibration metrics
│   └── artifacts/               # Serialized .joblib model files (gitignored)
│
├── api/                         # FastAPI backend
│   ├── main.py                  # App entrypoint, router registration
│   ├── routers/
│   │   ├── matches.py
│   │   ├── predictions.py
│   │   ├── accuracy.py
│   │   ├── teams.py
│   │   ├── auth.py
│   │   └── subscriptions.py
│   ├── models/                  # SQLAlchemy ORM models
│   │   └── *.py
│   ├── schemas/                 # Pydantic request/response schemas
│   │   └── *.py
│   ├── crud/                    # Database query functions
│   │   └── *.py
│   └── dependencies.py          # FastAPI dependency injection (db session, auth)
│
├── app/                         # Streamlit frontend
│   ├── Home.py                  # Landing page
│   └── pages/
│       ├── 1_Upcoming_Matches.py
│       ├── 2_Match_Detail.py
│       ├── 3_Accuracy_Tracker.py
│       ├── 4_Tournament_Bracket.py  # V1.1+
│       └── 5_Account.py
│
├── db/
│   ├── migrations/              # Alembic migration files
│   └── seed/                    # Seed data scripts (teams, tournaments)
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_ml.py
│   └── test_api.py
│
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.app
├── pyproject.toml               # Dependencies (Poetry or pip-tools)
├── .env.example
└── README.md
```

---

## 4. Database Schema

### 4.1 Entity Relationship Overview

```
tournaments ──< team_tournament_stats >── teams ──< players
                                           │
                    ┌──────────────────────┤
                    │                      │
               venues                   matches ──< match_stats
                    └──────────┬───────────┘        match_goals
                               │
                          match_predictions ──< prediction_outcomes
                               │
                          model_versions

users ──< user_subscriptions >── subscription_tiers
users ──< user_favorite_teams >── teams
```

### 4.2 Full Table Definitions (PostgreSQL DDL)

```sql
-- ============================================================
-- CORE TOURNAMENT ENTITIES
-- ============================================================

CREATE TABLE tournaments (
    id              SERIAL PRIMARY KEY,
    year            INTEGER         NOT NULL,
    name            VARCHAR(150)    NOT NULL,           -- e.g. "FIFA World Cup 2026"
    host_countries  VARCHAR(255),                       -- "USA, Mexico, Canada"
    host_continent  VARCHAR(50),                        -- "North America"
    num_teams       INTEGER         NOT NULL DEFAULT 32,
    num_groups      INTEGER,
    format          VARCHAR(100),                       -- "12-group + Round of 32 + knockout"
    start_date      DATE,
    end_date        DATE,
    is_current      BOOLEAN         NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMP       NOT NULL DEFAULT NOW()
);

-- ============================================================

CREATE TABLE teams (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(150)    NOT NULL,           -- "Brazil"
    official_name   VARCHAR(255),                       -- "Brazil Football Confederation"
    fifa_code       CHAR(3)         UNIQUE,             -- "BRA"
    confederation   VARCHAR(10),                        -- "CONMEBOL", "UEFA", "CAF", etc.
    logo_url        VARCHAR(512),
    created_at      TIMESTAMP       NOT NULL DEFAULT NOW()
);

-- ============================================================

-- Team-level stats per tournament (ranking at time of tournament)
CREATE TABLE team_tournament_stats (
    id                  SERIAL PRIMARY KEY,
    team_id             INTEGER     NOT NULL REFERENCES teams(id),
    tournament_id       INTEGER     NOT NULL REFERENCES tournaments(id),
    fifa_ranking        INTEGER,                        -- ranking at tournament start
    group_name          CHAR(1),                        -- "A", "B", "C" ... NULL for future
    seeded              BOOLEAN     DEFAULT FALSE,
    eliminated_in       VARCHAR(50),                    -- "group", "R32", "QF", etc.
    final_position      INTEGER,                        -- 1 = champion
    UNIQUE (team_id, tournament_id)
);

-- ============================================================

CREATE TABLE venues (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(150)    NOT NULL,
    city            VARCHAR(100),
    country         VARCHAR(100),
    capacity        INTEGER,
    latitude        DECIMAL(9,6),
    longitude       DECIMAL(9,6)
);

-- ============================================================

CREATE TABLE players (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(150)    NOT NULL,
    team_id         INTEGER         REFERENCES teams(id),
    position        VARCHAR(10),                        -- "GK", "DEF", "MID", "FWD"
    date_of_birth   DATE,
    caps            INTEGER         DEFAULT 0,          -- international appearances
    is_active       BOOLEAN         DEFAULT TRUE,
    created_at      TIMESTAMP       NOT NULL DEFAULT NOW()
);

-- ============================================================
-- MATCH DATA
-- ============================================================

CREATE TABLE matches (
    id                  SERIAL PRIMARY KEY,
    tournament_id       INTEGER     NOT NULL REFERENCES tournaments(id),
    venue_id            INTEGER     REFERENCES venues(id),
    match_date          TIMESTAMP,
    round               VARCHAR(50) NOT NULL,           -- "group", "R32", "R16", "QF", "SF", "3rd", "F"
    group_name          CHAR(1),                        -- only for group stage
    match_number        INTEGER,                        -- FIFA official match number
    home_team_id        INTEGER     REFERENCES teams(id),
    away_team_id        INTEGER     REFERENCES teams(id),

    -- Full-time scores
    home_score_ft       INTEGER,
    away_score_ft       INTEGER,

    -- Extra-time scores (NULL if not played)
    home_score_et       INTEGER,
    away_score_et       INTEGER,

    -- Penalty shootout scores (NULL if not played)
    home_score_pen      INTEGER,
    away_score_pen      INTEGER,

    -- Derived outcome (home_win / away_win / draw)
    -- "draw" only valid in group stage; in knockout, the team that
    -- advances after ET/pens is recorded as winner
    outcome             VARCHAR(20),

    status              VARCHAR(20) NOT NULL DEFAULT 'scheduled',
                                                        -- scheduled | live | completed | postponed
    attendance          INTEGER,
    created_at          TIMESTAMP   NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_matches_tournament ON matches(tournament_id);
CREATE INDEX idx_matches_date ON matches(match_date);
CREATE INDEX idx_matches_status ON matches(status);

-- ============================================================

CREATE TABLE match_stats (
    id                          SERIAL PRIMARY KEY,
    match_id                    INTEGER NOT NULL UNIQUE REFERENCES matches(id),
    home_possession             DECIMAL(5,2),           -- percentage
    away_possession             DECIMAL(5,2),
    home_shots                  INTEGER,
    away_shots                  INTEGER,
    home_shots_on_target        INTEGER,
    away_shots_on_target        INTEGER,
    home_corners                INTEGER,
    away_corners                INTEGER,
    home_fouls                  INTEGER,
    away_fouls                  INTEGER,
    home_yellow_cards           INTEGER,
    away_yellow_cards           INTEGER,
    home_red_cards              INTEGER,
    away_red_cards              INTEGER,
    created_at                  TIMESTAMP NOT NULL DEFAULT NOW()
);

-- ============================================================

CREATE TABLE match_goals (
    id              SERIAL PRIMARY KEY,
    match_id        INTEGER     NOT NULL REFERENCES matches(id),
    team_id         INTEGER     NOT NULL REFERENCES teams(id),
    player_id       INTEGER     REFERENCES players(id),     -- NULL for own goals
    minute          INTEGER,                                -- 0-90 + added time
    extra_time      BOOLEAN     NOT NULL DEFAULT FALSE,
    penalty         BOOLEAN     NOT NULL DEFAULT FALSE,
    own_goal        BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_match_goals_match ON match_goals(match_id);

-- ============================================================
-- ML MODEL TRACKING
-- ============================================================

CREATE TABLE model_versions (
    id                      SERIAL PRIMARY KEY,
    version_tag             VARCHAR(50)     NOT NULL,   -- "v1.0.0", "post-R32-g1"
    algorithm               VARCHAR(80)     NOT NULL,   -- "logistic_regression", "random_forest"
    training_data_through   DATE,                       -- date of latest match used in training
    training_matches_count  INTEGER,                    -- # matches in training set
    training_accuracy       DECIMAL(6,4),               -- fraction correct on test split
    log_loss                DECIMAL(8,6),               -- lower = better calibration
    artifact_path           VARCHAR(512),               -- s3://bucket/model.joblib or local path
    hyperparameters         JSONB,                      -- {"n_estimators": 200, ...}
    is_active               BOOLEAN         NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMP       NOT NULL DEFAULT NOW()
);

-- Only one model active at a time
CREATE UNIQUE INDEX idx_model_active ON model_versions(is_active) WHERE is_active = TRUE;

-- ============================================================

CREATE TABLE match_predictions (
    id                  SERIAL PRIMARY KEY,
    match_id            INTEGER     NOT NULL REFERENCES matches(id),
    model_version_id    INTEGER     NOT NULL REFERENCES model_versions(id),
    home_win_prob       DECIMAL(6,4) NOT NULL,           -- 0.0 to 1.0
    draw_prob           DECIMAL(6,4) NOT NULL,
    away_win_prob       DECIMAL(6,4) NOT NULL,
    predicted_outcome   VARCHAR(20) NOT NULL,            -- "home_win" | "draw" | "away_win"
    confidence_score    DECIMAL(6,4) NOT NULL,           -- max(home, draw, away) prob
    feature_importances JSONB,                           -- {"home_ranking": 0.23, ...}
    created_at          TIMESTAMP   NOT NULL DEFAULT NOW(),
    UNIQUE (match_id, model_version_id)
);

CREATE INDEX idx_predictions_match ON match_predictions(match_id);

-- ============================================================

CREATE TABLE prediction_outcomes (
    id                  SERIAL PRIMARY KEY,
    prediction_id       INTEGER     NOT NULL UNIQUE REFERENCES match_predictions(id),
    actual_outcome      VARCHAR(20) NOT NULL,            -- "home_win" | "away_win" | "draw"
    prediction_correct  BOOLEAN     NOT NULL,
    probability_of_actual DECIMAL(6,4),                 -- calibration: assigned prob to correct outcome
    brier_score         DECIMAL(8,6),                   -- per-match Brier score
    evaluated_at        TIMESTAMP   NOT NULL DEFAULT NOW()
);

-- ============================================================
-- USER & SUBSCRIPTION
-- ============================================================

CREATE TABLE users (
    id                  SERIAL PRIMARY KEY,
    email               VARCHAR(255)    NOT NULL UNIQUE,
    password_hash       VARCHAR(255)    NOT NULL,
    username            VARCHAR(100)    UNIQUE,
    full_name           VARCHAR(150),
    favorite_team_id    INTEGER         REFERENCES teams(id),
    follow_level        VARCHAR(20)     DEFAULT 'casual',   -- casual | regular | hardcore
    email_verified      BOOLEAN         NOT NULL DEFAULT FALSE,
    is_active           BOOLEAN         NOT NULL DEFAULT TRUE,
    created_at          TIMESTAMP       NOT NULL DEFAULT NOW(),
    last_login          TIMESTAMP
);

-- ============================================================

CREATE TABLE subscription_tiers (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(50)     NOT NULL UNIQUE,    -- "free", "pro"
    display_name    VARCHAR(100),
    price_usd       DECIMAL(10,2)   NOT NULL DEFAULT 0,
    billing_cycle   VARCHAR(20),                        -- "one-time", "monthly", "annual"
    feature_flags   JSONB           NOT NULL DEFAULT '{}',
    -- Example feature_flags:
    -- {"knockout_predictions": true, "feature_importance": false,
    --  "bracket_simulator": false, "export_csv": false}
    is_active       BOOLEAN         NOT NULL DEFAULT TRUE,
    created_at      TIMESTAMP       NOT NULL DEFAULT NOW()
);

-- ============================================================

CREATE TABLE user_subscriptions (
    id                      SERIAL PRIMARY KEY,
    user_id                 INTEGER     NOT NULL REFERENCES users(id),
    tier_id                 INTEGER     NOT NULL REFERENCES subscription_tiers(id),
    stripe_subscription_id  VARCHAR(255) UNIQUE,
    stripe_customer_id      VARCHAR(255),
    status                  VARCHAR(20) NOT NULL DEFAULT 'active',
                                                        -- active | cancelled | expired | trialing
    started_at              TIMESTAMP   NOT NULL DEFAULT NOW(),
    expires_at              TIMESTAMP,
    cancelled_at            TIMESTAMP,
    auto_renew              BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at              TIMESTAMP   NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_subscriptions_user ON user_subscriptions(user_id);

-- ============================================================

CREATE TABLE user_favorite_teams (
    user_id     INTEGER     NOT NULL REFERENCES users(id),
    team_id     INTEGER     NOT NULL REFERENCES teams(id),
    added_at    TIMESTAMP   NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, team_id)
);

-- ============================================================
-- PIPELINE AUDIT
-- ============================================================

CREATE TABLE pipeline_runs (
    id                  SERIAL PRIMARY KEY,
    trigger             VARCHAR(50),                    -- "scheduled", "manual", "post-match"
    match_id            INTEGER         REFERENCES matches(id),
    status              VARCHAR(20),                    -- "success" | "failed" | "partial"
    model_version_id    INTEGER         REFERENCES model_versions(id),
    predictions_updated INTEGER,                        -- count of predictions refreshed
    error_message       TEXT,
    started_at          TIMESTAMP       NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMP
);
```

### 4.3 Key Design Decisions

**JSONB for feature importances:** Feature lists will evolve as the model improves. Using JSONB avoids schema migrations every time a new feature is added to the model. Fast to query when indexed.

**Separate `prediction_outcomes` from `match_predictions`:** Predictions are written before the match; outcomes are written after. Keeping them separate makes it structurally impossible to accidentally mix pre-match and post-match data.

**`is_active` unique partial index on `model_versions`:** `CREATE UNIQUE INDEX ... WHERE is_active = TRUE` enforces at the database level that only one model is ever "active" — no application-layer guard needed.

**`pipeline_runs` audit table:** Every pipeline execution is logged with its outcome. This is what populates the "last updated at" timestamp in the Streamlit UI and enables post-mortem debugging if predictions go stale.

---

## 5. ML Pipeline Architecture

### 5.1 Feature Engineering

The following features are extracted from the database for each match row in the training/prediction set:

| Feature | Type | Source | Notes |
|---|---|---|---|
| `home_fifa_ranking` | Numeric | `team_tournament_stats.fifa_ranking` | At tournament start |
| `away_fifa_ranking` | Numeric | `team_tournament_stats.fifa_ranking` | At tournament start |
| `ranking_diff` | Numeric | Derived | `home_ranking - away_ranking` (negative = underdog at home) |
| `home_win_rate_all_wcs` | Numeric | Derived from `matches` | Historical W/(W+D+L) across all prior World Cups |
| `away_win_rate_all_wcs` | Numeric | Derived | Same |
| `h2h_home_wins` | Numeric | Derived | Head-to-head home team wins in all World Cup meetings |
| `h2h_away_wins` | Numeric | Derived | Head-to-head away team wins |
| `h2h_draws` | Numeric | Derived | Head-to-head draws |
| `home_goals_scored_per_game` | Numeric | Derived (current tournament) | Rolling average |
| `away_goals_scored_per_game` | Numeric | Derived (current tournament) | Rolling average |
| `home_goals_conceded_per_game` | Numeric | Derived (current tournament) | Rolling average |
| `away_goals_conceded_per_game` | Numeric | Derived (current tournament) | Rolling average |
| `home_rest_days` | Numeric | Derived | Days since last match in current tournament |
| `away_rest_days` | Numeric | Derived | Same |
| `host_nation_flag` | Binary | Derived | 1 if home team is a host nation (US, Mexico, Canada) |
| `round` | Categorical | `matches.round` | Encoded: group=0, R32=1, R16=2, QF=3, SF=4, F=5 |
| `confederation_matchup` | Categorical | Derived | e.g. "UEFA_vs_CONMEBOL" |

**Target variable:** `outcome` ∈ {`home_win`, `draw`, `away_win`} — multiclass classification (3 classes).

Note: For knockout rounds where draws are impossible in regulation + ET + pens, the "draw" outcome is excluded from the prediction and probabilities are redistributed between home_win and away_win. This is enforced in the prediction post-processing step.

### 5.2 scikit-learn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import joblib

NUMERIC_FEATURES = [
    "home_fifa_ranking", "away_fifa_ranking", "ranking_diff",
    "home_win_rate_all_wcs", "away_win_rate_all_wcs",
    "h2h_home_wins", "h2h_away_wins", "h2h_draws",
    "home_goals_scored_per_game", "away_goals_scored_per_game",
    "home_rest_days", "away_rest_days",
    "host_nation_flag",
]
CATEGORICAL_FEATURES = ["confederation_matchup", "round"]

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
])

# MVP baseline: Logistic Regression (fast, interpretable, good calibration)
base_model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=500,
    C=1.0,
    random_state=42,
)

# Wrap in CalibratedClassifierCV to improve probability calibration
# (critical — we're selling probabilities, not just predictions)
calibrated_model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", calibrated_model),
])

# Training
pipeline.fit(X_train, y_train)

# Serialization — single artifact, includes preprocessing + model
joblib.dump(pipeline, "artifacts/model_v1.0.joblib")

# Prediction (returns array of [home_win_prob, draw_prob, away_win_prob])
probabilities = pipeline.predict_proba(X_upcoming)
```

### 5.3 Model Evaluation

```python
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    brier_score_loss,
    classification_report,
)

# Primary metrics logged to model_versions table
accuracy   = accuracy_score(y_test, y_pred)
calibration = log_loss(y_test, y_proba)
brier      = brier_score_loss(y_test_binary, y_proba_positive_class)
```

Evaluation is run:
- On a held-out test split at initial training time (logged to `model_versions`)
- On every completed match after the tournament starts (logged to `prediction_outcomes`)

### 5.4 Post-Game Retraining Cycle

```
Match completed (result ingested)
           │
           ▼
  Append result to matches table
           │
           ▼
  Recompute rolling features
  (goals per game, rest days)
           │
           ▼
  Retrain pipeline on ALL
  historical data + current
  tournament completed matches
           │
           ▼
  Evaluate on test split
  (is new model ≥ current accuracy?)
           │
     Yes  / \ No
          │   └──► Keep current model, log warning
          ▼
  Serialize new model to artifacts/
  + upload to S3
           │
           ▼
  Update model_versions: set new
  version is_active=TRUE, deactivate
  previous
           │
           ▼
  Regenerate predictions for all
  remaining (scheduled) matches
           │
           ▼
  Write to match_predictions table
           │
           ▼
  Invalidate Redis cache keys
  for updated matches
           │
           ▼
  Log pipeline_run (success, count,
  timestamp)
           │
           ▼
  Update "last_updated" key in Redis
  (displayed in Streamlit UI header)
```

---

## 6. Data Ingestion & Post-Game Update Pipeline

### 6.1 Historical Data ETL (One-Time, Pre-MVP)

**Sources (in order of preference):**

| Dataset | URL | Coverage | License |
|---|---|---|---|
| Kaggle FIFA World Cup Dataset | kaggle.com/datasets/abecklas/fifa-world-cup | 1930–2022, all matches + scorers | CC0 Public Domain |
| football-data.org (free tier) | football-data.org/documentation/footballdata | 1994–present, structured JSON | Attribution required |
| Statsbomb Open Data | github.com/statsbomb/open-data | 2018, 2022 WC with deep event-level stats | CC BY-SA 4.0 |
| GitHub: jfjelstul/worldcup | github.com/jfjelstul/worldcup | 1930–2022, clean R/CSV format | MIT |

**ETL script flow (`ingestion/historical_etl.py`):**

```
1. Download CSV/JSON from source
2. Validate against Pydantic schemas
3. Resolve team name variations
   (e.g., "West Germany" → teams entry; "Germany" → separate teams entry)
4. Map to database entities: tournaments → teams → venues → matches
5. Compute derived stats (head-to-head, win rates) and store
6. Run data quality checks (no duplicate matches, score integrity)
7. Log import summary
```

### 6.2 Live Result Poller (`ingestion/live_poller.py`)

```python
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=30))
async def fetch_live_results(tournament_id: int) -> list[MatchResult]:
    """
    Calls football-data.org (or API-Football) to check for completed
    matches since the last pipeline run.
    Returns list of MatchResult pydantic models.
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.FOOTBALL_DATA_BASE_URL}/competitions/2000/matches",
            headers={"X-Auth-Token": settings.FOOTBALL_DATA_API_KEY},
            params={"status": "FINISHED"},
        )
        response.raise_for_status()
        return [MatchResult(**m) for m in response.json()["matches"]]
```

### 6.3 APScheduler Configuration (`ingestion/scheduler.py`)

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

# Poll every 15 min during match windows (2hr before to 3hr after any match)
scheduler.add_job(
    poll_and_update_pipeline,
    trigger="interval",
    minutes=15,
    id="live_result_poller",
)

# Full model retrain nightly at 03:00 UTC (catches any corrections)
scheduler.add_job(
    full_retrain_pipeline,
    trigger="cron",
    hour=3,
    minute=0,
    id="nightly_retrain",
)

scheduler.start()
```

**Pipeline is smart about cost:** polling doesn't retrain if no new results are detected. Retraining only fires when `new_completed_matches_count > 0`.

---

## 7. Backend API Design (FastAPI)

### 7.1 Core Endpoints

```
BASE URL: /api/v1

── PUBLIC (no auth required) ────────────────────────────────────────────

GET  /matches/upcoming
     Returns all scheduled matches with latest prediction (home_win_prob,
     draw_prob, away_win_prob, predicted_outcome, last_updated_at).

GET  /matches/{match_id}
     Full match detail: teams, date, venue, prediction breakdown,
     feature importances (truncated for free tier), H2H history.

GET  /matches/{match_id}/prediction
     Just the prediction payload (lightweight, for Streamlit caching).

GET  /tournaments/current
     Current tournament info: name, round, remaining matches count.

GET  /teams
     All teams in the current tournament.

GET  /teams/{team_id}
     Team profile + performance in current + historical tournaments.

GET  /accuracy
     Aggregated accuracy stats: overall, by round, calibration metrics.

GET  /accuracy/history
     Row-by-row: each completed match, what we predicted, what happened.

── AUTH ──────────────────────────────────────────────────────────────────

POST /auth/register          { email, password, username? }
POST /auth/login             { email, password } → { access_token }
POST /auth/refresh           { refresh_token } → { access_token }
POST /auth/verify-email      { token }

── USER (JWT required) ───────────────────────────────────────────────────

GET  /user/profile
PUT  /user/profile           { favorite_team_id, username, follow_level }
GET  /user/subscription      { tier, status, expires_at, features }

── SUBSCRIPTIONS ─────────────────────────────────────────────────────────

POST /subscriptions/checkout { tier_id } → { stripe_checkout_url }
POST /subscriptions/cancel
POST /webhooks/stripe        (Stripe webhook — not authenticated via JWT)

── INTERNAL / ADMIN (separate auth header) ───────────────────────────────

POST /internal/pipeline/run          Trigger pipeline manually
GET  /internal/model/versions        List all model versions + metrics
PUT  /internal/model/versions/{id}/activate
GET  /internal/pipeline/runs         Audit log of all pipeline executions
```

### 7.2 Response Shapes (key schemas)

```python
# Prediction payload
class MatchPredictionResponse(BaseModel):
    match_id: int
    home_team: str
    away_team: str
    match_date: datetime
    round: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    predicted_outcome: str          # "home_win" | "draw" | "away_win"
    confidence_score: float
    feature_importances: dict | None  # None for free-tier users
    model_version: str
    last_updated_at: datetime

# Accuracy summary
class AccuracySummaryResponse(BaseModel):
    total_predictions: int
    correct_predictions: int
    accuracy_pct: float
    log_loss: float
    by_round: list[RoundAccuracy]
    last_evaluated_at: datetime
```

---

## 8. Frontend Architecture (Streamlit)

### 8.1 App Structure & Page Map

```
app/
├── Home.py                     Landing: next match prediction widget, platform pitch
└── pages/
    ├── 1_Upcoming_Matches.py   All fixtures with prediction badges
    ├── 2_Match_Detail.py       Per-match deep-dive (selectbox to pick match)
    ├── 3_Accuracy_Tracker.py   Historical prediction log + calibration chart
    ├── 4_Tournament_Bracket.py  (V1.1+) Bracket + Monte Carlo simulator
    └── 5_Account.py            Login, registration, subscription management
```

### 8.2 Streamlit Best Practices for This App

```python
# Cache API responses to avoid hammering FastAPI on every rerender
@st.cache_data(ttl=900)  # 15-minute TTL aligned to pipeline schedule
def get_upcoming_matches() -> list[dict]:
    response = requests.get(f"{API_BASE}/matches/upcoming")
    return response.json()

# Cache model metadata (changes rarely)
@st.cache_resource
def load_accuracy_summary() -> dict:
    response = requests.get(f"{API_BASE}/accuracy")
    return response.json()

# Show last updated timestamp in sidebar — critical trust signal
def show_last_updated():
    last_updated = st.session_state.get("last_updated_at", "unknown")
    st.sidebar.caption(f"🕐 Predictions last updated: {last_updated}")
```

### 8.3 Mobile Layout Guidelines

- Use `st.columns([1, 1])` maximum on mobile — no 3-column layouts on narrow viewports.
- Probability bars: use `st.progress()` with percentage text above (not beside) on mobile.
- Avoid wide `st.dataframe()` tables without horizontal scroll wrapper.
- Test on a 390px-wide viewport (iPhone 15 width) before each matchday.

### 8.4 Tiered Access in Streamlit

```python
def check_feature_access(feature: str) -> bool:
    """Returns True if current user's subscription allows this feature."""
    tier_features = st.session_state.get("tier_features", {})
    return tier_features.get(feature, False)

# In Match Detail page:
if check_feature_access("feature_importance"):
    st.plotly_chart(render_feature_importance_chart(prediction))
else:
    st.info("🔒 Feature importance breakdown — available on Pro tier.")
    if st.button("Upgrade to Pro"):
        st.switch_page("pages/5_Account.py")
```

---

## 9. Data Source Integrations

### 9.1 football-data.org (Live Results — Primary)

| Property | Value |
|---|---|
| Endpoint | `https://api.football-data.org/v4` |
| Auth | `X-Auth-Token` header |
| Free tier limit | 10 req/min |
| 2026 World Cup competition ID | `2000` (FIFA World Cup) |
| Key endpoint | `GET /competitions/2000/matches?status=FINISHED` |
| Latency | Results typically available 15–30 min after final whistle |
| Backup | API-Football (RapidAPI) as failover |

### 9.2 API-Football via RapidAPI (Failover)

| Property | Value |
|---|---|
| Host | `api-football-v1.p.rapidapi.com` |
| Auth | `X-RapidAPI-Key` header |
| Free tier limit | 100 req/day |
| Key endpoint | `GET /fixtures?league=1&season=2026&status=FT` |

### 9.3 Historical Data (Static, One-Time ETL)

| Dataset | Format | Load method |
|---|---|---|
| Kaggle FIFA WC Dataset | CSV | `pandas.read_csv()` → Postgres |
| jfjelstul/worldcup (GitHub) | CSV | `pandas.read_csv()` → Postgres |
| Statsbomb Open Data | JSON (event-level) | `json.load()` → aggregated → Postgres |

### 9.4 Stripe (Payments)

| Property | Value |
|---|---|
| Integration point | Checkout Sessions API for subscription creation |
| Webhook events | `customer.subscription.updated`, `customer.subscription.deleted`, `invoice.payment_failed` |
| Secret storage | Environment variable `STRIPE_SECRET_KEY` — never in code or DB |
| Webhook verification | Always verify `Stripe-Signature` header before processing |

---

## 10. Environment Variables & Configuration

### 10.1 `.env.example`

```bash
# ── DATABASE ────────────────────────────────────────────────
DATABASE_URL=postgresql+asyncpg://wcuser:password@localhost:5432/worldcup_db
DATABASE_URL_SYNC=postgresql://wcuser:password@localhost:5432/worldcup_db
DATABASE_URL_TEST=postgresql://wcuser:password@localhost:5432/worldcup_test
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20

# ── REDIS ───────────────────────────────────────────────────
REDIS_URL=redis://localhost:6379/0
REDIS_CACHE_TTL_SECONDS=900         # 15 min — aligned to pipeline schedule

# ── FOOTBALL DATA API (primary) ─────────────────────────────
FOOTBALL_DATA_API_KEY=your_key_here
FOOTBALL_DATA_BASE_URL=https://api.football-data.org/v4
FOOTBALL_DATA_WC_COMPETITION_ID=2000

# ── API-FOOTBALL (failover) ──────────────────────────────────
API_FOOTBALL_KEY=your_rapidapi_key
API_FOOTBALL_HOST=api-football-v1.p.rapidapi.com
API_FOOTBALL_LEAGUE_ID=1            # FIFA World Cup

# ── ML PIPELINE ─────────────────────────────────────────────
MODEL_ARTIFACT_DIR=./ml/artifacts
MODEL_ARTIFACT_S3_BUCKET=worldcup-model-artifacts
ACTIVE_MODEL_VERSION=v1.0.0
MODEL_MIN_ACCURACY_THRESHOLD=0.50   # Don't deploy model below this accuracy
PIPELINE_POLL_INTERVAL_MINUTES=15
NIGHTLY_RETRAIN_HOUR_UTC=3

# ── AWS (model artifact storage) ────────────────────────────
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1

# ── FASTAPI ─────────────────────────────────────────────────
APP_ENV=development                 # development | staging | production
SECRET_KEY=generate-a-long-random-string-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=30
INTERNAL_API_KEY=your-internal-admin-key
ALLOWED_ORIGINS=http://localhost:8501,https://your-domain.streamlit.app

# ── STREAMLIT ────────────────────────────────────────────────
API_BASE_URL=http://localhost:8000/api/v1
STREAMLIT_SERVER_PORT=8501

# ── STRIPE ──────────────────────────────────────────────────
STRIPE_SECRET_KEY=sk_test_xxx
STRIPE_PUBLISHABLE_KEY=pk_test_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx
STRIPE_PRO_PRICE_ID=price_xxx      # Stripe price object ID for Pro tier

# ── LOGGING ─────────────────────────────────────────────────
LOG_LEVEL=INFO                      # DEBUG | INFO | WARNING | ERROR
LOG_FORMAT=json                     # json | text (json for production)
```

### 10.2 Configuration Loading (FastAPI)

```python
# api/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str
    redis_url: str
    secret_key: str
    football_data_api_key: str
    stripe_secret_key: str
    app_env: str = "development"
    model_artifact_dir: str = "./ml/artifacts"
    pipeline_poll_interval_minutes: int = 15
    model_min_accuracy_threshold: float = 0.50

settings = Settings()
```

### 10.3 Secrets Management

| Environment | Method |
|---|---|
| Local dev | `.env` file (gitignored) |
| Staging / Production (Railway) | Railway environment variable panel |
| CI/CD (GitHub Actions) | GitHub Actions Secrets |
| **Never** | Hardcoded in source, committed to version control, stored in database plaintext |

---

## 11. Security Architecture

| Concern | Mitigation |
|---|---|
| **Password storage** | bcrypt via passlib; minimum cost factor 12; never stored plaintext |
| **JWT signing** | HS256, 60-min access token + 30-day refresh token; refresh token stored in httpOnly cookie |
| **Stripe webhooks** | Always verify `Stripe-Signature` header using `STRIPE_WEBHOOK_SECRET`; reject unverified payloads |
| **API rate limiting** | FastAPI middleware (slowapi) — 60 req/min for public endpoints, 10 req/min for auth endpoints |
| **SQL injection** | All queries via SQLAlchemy ORM parameterized queries; no raw string interpolation |
| **CORS** | `ALLOWED_ORIGINS` env var whitelist; only Streamlit app origin permitted |
| **Internal admin endpoints** | Separate `X-Internal-Key` header auth; not exposed publicly (firewall rule or private network) |
| **Data API keys** | Stored only in environment variables; rotated if leaked; football data APIs have read-only keys |
| **Dependency scanning** | `pip-audit` in CI pipeline; Dependabot alerts on GitHub |

---

## 12. Deployment Architecture

### 12.1 MVP Topology

```
┌─────────────────────┐    HTTP    ┌──────────────────────────────────┐
│  Streamlit App      │ ─────────► │  FastAPI + APScheduler           │
│  (Streamlit Cloud)  │            │  (Railway / Render — 1 dyno)     │
└─────────────────────┘            └──────┬───────────────────────────┘
                                          │
                        ┌─────────────────┴──────────────────┐
                        │                                    │
               ┌────────▼────────┐                 ┌────────▼────────┐
               │  PostgreSQL 16  │                 │  Redis 7        │
               │  (Railway DB)   │                 │  (Railway Redis)│
               └─────────────────┘                 └─────────────────┘
                        │
               ┌────────▼────────┐
               │  AWS S3         │
               │  (model .joblib │
               │   artifacts)    │
               └─────────────────┘
```

### 12.2 Docker Compose (Local Dev)

```yaml
version: "3.9"
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - db
      - redis
    volumes:
      - ./ml/artifacts:/app/ml/artifacts

  app:
    build:
      context: .
      dockerfile: Dockerfile.app
    ports:
      - "8501:8501"
    env_file: .env
    depends_on:
      - api

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: wcuser
      POSTGRES_PASSWORD: password
      POSTGRES_DB: worldcup_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### 12.3 CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.11" }
      - run: pip install -e ".[dev]"
      - run: pip-audit                        # dependency vulnerability scan
      - run: ruff check .                     # linting
      - run: pytest tests/ --cov=api --cov=ml # tests + coverage
      - run: alembic upgrade head             # verify migrations apply cleanly

  deploy:
    needs: test
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - run: railway up --service api         # deploy FastAPI to Railway
```

---

## 13. Development Setup & Local Environment

```bash
# 1. Clone the repo
git clone https://github.com/your-org/worldcup-platform.git
cd worldcup-platform

# 2. Create Python virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Copy and populate environment variables
cp .env.example .env
# Edit .env with your API keys and database credentials

# 5. Start Postgres + Redis via Docker Compose
docker compose up -d db redis

# 6. Run database migrations
alembic upgrade head

# 7. Seed static data (teams, venues, tournament structure)
python db/seed/seed_teams.py
python db/seed/seed_tournament_2026.py

# 8. Run historical data ETL (one-time)
python ingestion/historical_etl.py --source kaggle --path ./data/WorldCupMatches.csv

# 9. Train initial model
python ml/train.py --output ml/artifacts/model_v1.0.joblib

# 10. Generate predictions for all upcoming 2026 matches
python ml/predict.py --model ml/artifacts/model_v1.0.joblib

# 11. Start FastAPI (with hot reload)
uvicorn api.main:app --reload --port 8000

# 12. Start Streamlit (separate terminal)
streamlit run app/Home.py
```

---

## 14. Architectural Decision Log (ADRs)

### ADR-001: Streamlit over React/Next.js for the Frontend

**Decision:** Use Streamlit instead of a JavaScript frontend framework.
**Rationale:** The team is Python-only; a React app would require a separate skill set, separate CI pipeline, and a REST contract that adds days of work. Streamlit's native Plotly/matplotlib integration eliminates a data serialization layer. The tradeoff is less fine-grained UI control and Streamlit's re-render model (reruns the whole script on interaction), but for a data-display app with limited interactivity, this is acceptable.
**Review trigger:** If the team hires a frontend engineer or if Streamlit's performance under matchday load proves unacceptable.

---

### ADR-002: FastAPI + Streamlit over an All-in-One Streamlit App

**Decision:** FastAPI serves data; Streamlit is a pure UI layer.
**Rationale:** A pure Streamlit app would embed database queries and ML inference in the frontend code — untestable, unscalable, and impossible to expose as an API (V1.1+ requirement). Separating concerns means the ML pipeline, API, and UI are independently deployable and testable.
**Tradeoff:** One extra HTTP hop and two deployment targets instead of one.

---

### ADR-003: Retrain After Every Match, Not Batch Weekly

**Decision:** Trigger model retraining after every completed match result is ingested.
**Rationale:** World Cup match volume is low (≤8 matches per round), and a scikit-learn Logistic Regression / Random Forest on ~900 rows trains in under 5 seconds. There is no cost to retraining frequently, and it keeps predictions current. Weekly batch retraining would mean predictions are stale for up to 7 days mid-tournament.
**Review trigger:** If model training time exceeds 60 seconds (e.g., after migrating to a deep learning approach), switch to scheduled batch retraining.

---

### ADR-004: PostgreSQL over a NoSQL Store

**Decision:** PostgreSQL as the primary database.
**Rationale:** The domain model is highly relational: matches → teams → tournaments → predictions → outcomes. A document store would require denormalization that makes accuracy reporting (cross-referencing predictions with results) error-prone. PostgreSQL's JSONB handles the variable `feature_importances` payload without losing relational integrity for everything else.

---

### ADR-005: Single Deployed Model Version at a Time

**Decision:** Only one `model_versions` row has `is_active = TRUE` at any time; all predictions are regenerated on activation.
**Rationale:** Serving predictions from mixed model versions (some matches from v1.0, some from v1.1) makes accuracy reporting ambiguous and undermines user trust. The Accuracy Tracker must be attributable to a consistent model state. The unique partial index enforces this at the database level.

---

*End of Technical Architecture Document v1.0*
