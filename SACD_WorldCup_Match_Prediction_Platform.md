# Security and Access Control Document
## FIFA World Cup Match Prediction Platform

**Author role:** Senior Security Engineer — SaaS / Tiered Subscription Products
**Companion to:** PRD v1.1 · Technical Architecture Document v1.0
**Status:** Draft v1.0
**Last updated:** June 26, 2026
**Classification:** Internal — Engineering & Product

---

## Table of Contents

1. [Authentication Architecture](#1-authentication-architecture)
2. [User Roles — Definitions & Hierarchy](#2-user-roles--definitions--hierarchy)
3. [Role × Feature Access Matrix](#3-role--feature-access-matrix)
4. [API Endpoint Security Matrix](#4-api-endpoint-security-matrix)
5. [Row-Level Security — Subscription Tier Enforcement](#5-row-level-security--subscription-tier-enforcement)
6. [Rate Limiting Rules](#6-rate-limiting-rules)
7. [Standardized Error Response Contract](#7-standardized-error-response-contract)
8. [Error Handling Guide — API Layer](#8-error-handling-guide--api-layer)
9. [Error Handling Guide — Data Pipeline Layer](#9-error-handling-guide--data-pipeline-layer)
10. [Error Handling Guide — ML Model Layer](#10-error-handling-guide--ml-model-layer)
11. [Error Handling Guide — External API Layer](#11-error-handling-guide--external-api-layer)
12. [Edge Cases — Data Platform Specific](#12-edge-cases--data-platform-specific)
13. [Security Headers & Transport](#13-security-headers--transport)
14. [Secrets & Credential Management](#14-secrets--credential-management)
15. [Audit Logging](#15-audit-logging)
16. [Incident Response Playbook](#16-incident-response-playbook)

---

## 1. Authentication Architecture

### 1.1 Recommended Method: JWT with Refresh Token Rotation

**Verdict for this product:** JWT (JSON Web Tokens) with short-lived access tokens, long-lived rotating refresh tokens, and bcrypt password hashing. Social OAuth (Google) as a secondary login path.

**Why JWT over alternatives:**

| Method | Verdict | Reason |
|---|---|---|
| **JWT + Refresh Rotation** | ✅ Recommended | Stateless; works cleanly across the Streamlit ↔ FastAPI split-origin setup. Tier/role claims embedded in token reduce DB lookups per request. Refresh rotation limits blast radius of a stolen token. |
| Session cookies + server-side store | ⚠️ Avoid for MVP | Requires a shared session store (Redis) that both Streamlit and FastAPI can read — adds infra coupling. Also problematic across different origins. |
| API keys only | ⚠️ Machine-to-machine only | Excellent for the internal pipeline service account and the future paid API tier (V1.1+). Poor UX for human users — no expiry, no per-session scope, dangerous if leaked. |
| OAuth only (no email/password) | ⚠️ Reduces user ownership | Forces dependency on Google/GitHub account state. Users lose access if their OAuth provider suspends them. Offer as supplement, not replacement. |
| Opaque tokens | ❌ Reject | Requires a DB lookup on every single API request to validate. Unacceptable for a prediction platform with bursty matchday traffic. |

### 1.2 Token Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ACCESS TOKEN (JWT)                         │
│  Lifetime: 60 minutes                                           │
│  Storage: In-memory on Streamlit (st.session_state)            │
│           Authorization header for FastAPI calls               │
│  Signed with: HS256 + SECRET_KEY env var                       │
│                                                                  │
│  Payload:                                                        │
│  {                                                               │
│    "sub":   "user:123",                                         │
│    "email": "user@example.com",                                 │
│    "role":  "pro_user",                                         │
│    "tier":  "pro",                                              │
│    "tier_features": {                                            │
│      "knockout_predictions": true,                              │
│      "feature_importance":   true,                              │
│      "bracket_simulator":    false,                             │
│      "export_csv":           true,                              │
│      "api_access":           false                              │
│    },                                                            │
│    "jti":  "unique-token-id",   ← for revocation               │
│    "iat":  1719388800,                                           │
│    "exp":  1719392400                                            │
│  }                                                               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     REFRESH TOKEN (opaque)                      │
│  Lifetime: 30 days                                              │
│  Storage: httpOnly, Secure, SameSite=Strict cookie             │
│           Hashed value stored in DB (refresh_tokens table)     │
│  Rotation: Issued a new token on every refresh call            │
│  Revocation: Previous token invalidated on rotation            │
└─────────────────────────────────────────────────────────────────┘
```

**Why embed tier_features in the JWT:** Every API request needs to gate on feature access. Embedding `tier_features` in the token means FastAPI dependencies read from the token claim — no database round-trip per request. Tradeoff: if an admin changes a user's tier mid-session, the change takes effect at the next token refresh (max 60 min lag). This is acceptable for a subscription product.

### 1.3 Token Refresh & Rotation Flow

```
1. User logs in → FastAPI issues:
   - Access token (60 min, returned in JSON body)
   - Refresh token (30 days, httpOnly cookie)
   - Refresh token hash stored in refresh_tokens table

2. Access token expires → Streamlit sends refresh token cookie to
   POST /auth/refresh

3. FastAPI:
   a. Validates refresh token against DB hash
   b. Checks refresh token not revoked / expired
   c. Issues NEW access token + NEW refresh token
   d. Invalidates OLD refresh token in DB
   e. Sets new httpOnly cookie

4. If refresh token is also expired → user must log in again

5. Stolen refresh token reuse:
   - Attacker uses stolen token → new token pair issued
   - Legitimate user uses old token → MISMATCH detected
   - ALL tokens for that user are immediately revoked
   - User is forced to re-authenticate
   - Security team alerted (see Audit Logging, Section 15)
```

### 1.4 Password Security

```python
# api/security/passwords.py
from passlib.context import CryptContext

# bcrypt with cost factor 12 (takes ~300ms — intentionally slow)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

def hash_password(plain: str) -> str:
    return pwd_context.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)
```

**Password policy enforced at registration and change:**

| Rule | Minimum |
|---|---|
| Length | 10 characters |
| Complexity | 1 uppercase + 1 digit + 1 special character |
| Breach check | Pwned Passwords API (k-anonymity SHA-1 prefix check) — reject if count > 0 |
| History | Last 5 passwords rejected (stored as hashes in `password_history` table) |
| Expiry | Not enforced for end users; 90-day rotation required for Admin accounts |

### 1.5 Social OAuth (Google)

Google OAuth2 is offered as an alternative login path for higher conversion (reduces signup friction for the Casual Fan persona). Implementation notes:

- **Library:** `Authlib` (Python OAuth2 client)
- **Flow:** Authorization Code with PKCE
- **On first OAuth login:** Create a `users` record with `password_hash = NULL`, `oauth_provider = 'google'`, `oauth_sub = google_sub_id`. Never allow password login for OAuth accounts (and vice versa — no OAuth for password-registered accounts).
- **On subsequent OAuth logins:** Look up by `oauth_sub`, not email — email can change on the provider side.
- **Scope requested:** `openid email profile` — minimal. Never request write scopes.

### 1.6 Multi-Factor Authentication (MFA)

| Role | MFA Requirement |
|---|---|
| `anonymous` | N/A |
| `free_user` | Optional (TOTP, encouraged on account page) |
| `pro_user` | Optional (TOTP, encouraged — higher value account) |
| `admin` | **Mandatory.** TOTP via Google Authenticator / Authy. Cannot log in without MFA. |
| `service_account` | N/A — uses API key, not password auth |

**TOTP implementation:** `pyotp` library. 30-second window, 6-digit code, HMAC-SHA1. Backup codes: 10 single-use codes generated at MFA setup, shown once, stored as bcrypt hashes.

### 1.7 Service Account Authentication (Internal Pipeline)

The data ingestion pipeline and APScheduler jobs authenticate to the FastAPI internal endpoints using a long-lived API key — not JWT. This key:

- Is stored in `INTERNAL_API_KEY` environment variable only (never in DB or code)
- Is sent as `X-Internal-Key: <key>` header
- Internal endpoints reject any request without this header with `403 Forbidden`
- The key is 256-bit random hex, rotated on each deployment
- It is never exposed to Streamlit or any user-facing surface

---

## 2. User Roles — Definitions & Hierarchy

```
                        service_account
                              │
                           admin
                          /       \
                    pro_user    free_user
                         \       /
                         anonymous
```

### 2.1 Role Definitions

**`anonymous`**
A visitor who has not logged in. Can see the landing page, the next upcoming match prediction (truncated), and the platform pitch. Cannot access any user-specific data, full prediction detail, or accuracy tracker beyond a summary card. Designed to hook visitors into signing up during search-driven matchday traffic.

**`free_user`**
A registered user on the free tier. Subscription status: `free` in `subscription_tiers`. Can see all upcoming match predictions with Win/Draw/Loss probabilities. Cannot see feature importance breakdowns, knockout-stage detail, bracket simulation, or CSV export. No API access.

**`pro_user`**
A registered user with an active paid subscription. Has access to full knockout-stage predictions, feature importance panels, CSV export, and (V1.1+) bracket simulator. Still cannot access admin functions or the internal pipeline API.

**`admin`**
Internal company staff. Has full read access to all user accounts (for support), subscription management, model version management, and the ability to trigger pipeline runs manually. Cannot delete prediction history (audit integrity). Must use MFA. Actions are fully audited.

**`service_account`**
A non-human identity used by the data ingestion pipeline and APScheduler. Authenticates via API key. Has write access to the `matches`, `match_predictions`, and `pipeline_runs` tables via internal endpoints. Has no access to user data or subscription management.

### 2.2 Subscription Tier ↔ Role Mapping

| Subscription Status | Role Assigned | Notes |
|---|---|---|
| Not registered | `anonymous` | No JWT issued |
| Registered, no payment | `free_user` | JWT issued with `tier: "free"` |
| Active paid subscription | `pro_user` | JWT issued with `tier: "pro"` |
| Subscription expired | `free_user` | Tier downgraded on Stripe webhook event |
| Subscription cancelled | `free_user` | Effective immediately on Stripe event |
| Company staff account | `admin` | Manually granted; never via Stripe flow |
| Pipeline process | `service_account` | API key, no JWT |

---

## 3. Role × Feature Access Matrix

Legend: ✅ Full access · 🔒 Blocked (upgrade prompt shown) · 👁️ Preview only (truncated/teaser) · ❌ Not accessible · 🔑 Admin/internal only

### 3.1 UI / Dashboard Features

| Feature | anonymous | free_user | pro_user | admin |
|---|---|---|---|---|
| Landing page — next match preview (1 match, no detail) | 👁️ | ✅ | ✅ | ✅ |
| Upcoming Matches Dashboard — full list | ❌ | ✅ | ✅ | ✅ |
| Match prediction: Win/Draw/Loss probabilities | 👁️ (1 match) | ✅ | ✅ | ✅ |
| Match prediction: confidence score | ❌ | ✅ | ✅ | ✅ |
| Match prediction: feature importance breakdown | ❌ | 🔒 | ✅ | ✅ |
| Match prediction: head-to-head historical record | ❌ | ✅ | ✅ | ✅ |
| Group stage predictions | ❌ | ✅ (historical only — group stage done) | ✅ | ✅ |
| Knockout stage predictions (R32 through Final) | ❌ | 🔒 | ✅ | ✅ |
| Accuracy Tracker — summary card | 👁️ (overall % only) | ✅ | ✅ | ✅ |
| Accuracy Tracker — per-match history | ❌ | ✅ | ✅ | ✅ |
| Accuracy Tracker — calibration chart | ❌ | 🔒 | ✅ | ✅ |
| Tournament Bracket view | ❌ | 👁️ (no predictions shown) | ✅ | ✅ |
| Bracket Simulator (Monte Carlo) | ❌ | 🔒 | ✅ (V1.1+) | ✅ |
| CSV export — predictions | ❌ | 🔒 | ✅ | ✅ |
| My Team feed / notifications | ❌ | ✅ | ✅ | ✅ |
| SHAP model explainability panel | ❌ | 🔒 | ✅ (V1.1+) | ✅ |
| Account management | ❌ | ✅ | ✅ | ✅ |
| Admin panel (user management, model control) | ❌ | ❌ | ❌ | 🔑 |

### 3.2 Data Access

| Data Entity | anonymous | free_user | pro_user | admin |
|---|---|---|---|---|
| `tournaments` — all fields | ❌ | ✅ | ✅ | ✅ |
| `teams` — name, confederation | 👁️ | ✅ | ✅ | ✅ |
| `matches` — group stage, scheduled/completed | 👁️ (1) | ✅ | ✅ | ✅ |
| `matches` — knockout stage | ❌ | 🔒 | ✅ | ✅ |
| `match_predictions` — probabilities only | 👁️ (1) | ✅ | ✅ | ✅ |
| `match_predictions` — `feature_importances` field | ❌ | 🔒 | ✅ | ✅ |
| `prediction_outcomes` — results log | ❌ | ✅ | ✅ | ✅ |
| `model_versions` — version tag + accuracy | ❌ | ✅ | ✅ | ✅ |
| `model_versions` — hyperparameters, artifact path | ❌ | ❌ | ❌ | 🔑 |
| `users` — own record only | ❌ | ✅ (own) | ✅ (own) | 🔑 (all) |
| `user_subscriptions` — own record | ❌ | ✅ (own) | ✅ (own) | 🔑 (all) |
| `pipeline_runs` audit log | ❌ | ❌ | ❌ | 🔑 |
| `match_stats` (possession, shots, etc.) | ❌ | ✅ | ✅ | ✅ |
| `players` table | ❌ | ✅ | ✅ | ✅ |

---

## 4. API Endpoint Security Matrix

Format: `METHOD /path` → [min role required] | [rate limit bucket]

### 4.1 Public Endpoints (No Auth Required)

| Endpoint | Min Role | Notes |
|---|---|---|
| `GET /api/v1/matches/upcoming` | `anonymous` | Returns max 1 match for anonymous; full list for authenticated |
| `GET /api/v1/tournaments/current` | `anonymous` | Tournament metadata only |
| `GET /api/v1/accuracy` | `anonymous` | Summary card only (overall accuracy %) |
| `POST /api/v1/auth/register` | `anonymous` | Rate-limited aggressively (see Section 6) |
| `POST /api/v1/auth/login` | `anonymous` | Rate-limited aggressively |
| `POST /api/v1/auth/refresh` | `anonymous` | Requires valid httpOnly refresh token cookie |
| `POST /api/v1/auth/oauth/google` | `anonymous` | Google OAuth callback |
| `GET /api/v1/health` | `anonymous` | Liveness check — returns `{"status": "ok"}` only |

### 4.2 Free User Endpoints (JWT Required, Any Tier)

| Endpoint | Min Role | Tier Notes |
|---|---|---|
| `GET /api/v1/matches/upcoming` | `free_user` | Full list; feature_importances field stripped for free tier |
| `GET /api/v1/matches/{id}` | `free_user` | Knockout matches → 403 for free_user |
| `GET /api/v1/matches/{id}/prediction` | `free_user` | Probabilities only for free; full payload for pro |
| `GET /api/v1/teams` | `free_user` | Full list |
| `GET /api/v1/teams/{id}` | `free_user` | Full team profile |
| `GET /api/v1/accuracy` | `free_user` | Full accuracy table |
| `GET /api/v1/accuracy/history` | `free_user` | Row-by-row history |
| `GET /api/v1/user/profile` | `free_user` | Own record only |
| `PUT /api/v1/user/profile` | `free_user` | Own record only |
| `GET /api/v1/user/subscription` | `free_user` | Own subscription |
| `POST /api/v1/subscriptions/checkout` | `free_user` | Initiates Stripe checkout |
| `POST /api/v1/auth/logout` | `free_user` | Revokes refresh token |
| `POST /api/v1/auth/change-password` | `free_user` | Requires current password |
| `POST /api/v1/auth/setup-mfa` | `free_user` | Optional MFA setup |

### 4.3 Pro User Endpoints (JWT Required, `pro` Tier)

| Endpoint | Min Role | Notes |
|---|---|---|
| `GET /api/v1/matches/{id}` (knockout) | `pro_user` | Returns full detail including feature_importances |
| `GET /api/v1/predictions/export` | `pro_user` | CSV download of all predictions |
| `GET /api/v1/simulator/bracket` | `pro_user` | V1.1+ Monte Carlo simulation |
| `GET /api/v1/matches/{id}/shap` | `pro_user` | V1.1+ SHAP breakdown |
| `POST /api/v1/subscriptions/cancel` | `pro_user` | Cancels their own subscription |

### 4.4 Admin Endpoints (JWT + `admin` Role + MFA verified)

| Endpoint | Min Role | Notes |
|---|---|---|
| `GET /api/v1/admin/users` | `admin` | All users list with pagination |
| `GET /api/v1/admin/users/{id}` | `admin` | Full user record for support |
| `PUT /api/v1/admin/users/{id}/tier` | `admin` | Manual tier override (e.g. for refunds) |
| `GET /api/v1/admin/subscriptions` | `admin` | All subscriptions |
| `GET /api/v1/admin/pipeline/runs` | `admin` | Pipeline audit log |
| `GET /api/v1/admin/model/versions` | `admin` | All model versions + hyperparameters |
| `PUT /api/v1/admin/model/versions/{id}/activate` | `admin` | Activate a specific model version |

### 4.5 Internal Service Endpoints (API Key Only — Not JWT)

| Endpoint | Auth | Notes |
|---|---|---|
| `POST /internal/pipeline/run` | `X-Internal-Key` | Trigger post-game pipeline manually |
| `POST /internal/pipeline/retrain` | `X-Internal-Key` | Force full model retrain |
| `POST /internal/matches/result` | `X-Internal-Key` | Ingest a match result |
| `POST /internal/predictions/refresh` | `X-Internal-Key` | Regenerate all upcoming predictions |
| `GET /internal/health/deep` | `X-Internal-Key` | Full health check including DB and Redis |

**Internal endpoints are never exposed via the public API gateway.** They are accessible only on an internal network interface or via a firewall rule that restricts source IP to the pipeline's deployment environment.

### 4.6 FastAPI Dependency Injection for Auth

```python
# api/dependencies.py

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

bearer_scheme = HTTPBearer()

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(bearer_scheme)
) -> TokenPayload:
    """Validate JWT and return decoded payload."""
    try:
        payload = decode_jwt(credentials.credentials)
        return TokenPayload(**payload)
    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "TOKEN_EXPIRED", "message": "Access token expired. Please refresh."}
        )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "TOKEN_INVALID", "message": "Invalid or malformed token."}
        )

def require_pro(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
    """Enforce pro tier access."""
    if user.tier not in ("pro", "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "TIER_INSUFFICIENT",
                "message": "This feature requires a Pro subscription.",
                "upgrade_url": "/api/v1/subscriptions/checkout"
            }
        )
    return user

def require_admin(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
    """Enforce admin role."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "ADMIN_REQUIRED", "message": "Access denied."}
        )
    return user

def require_internal(x_internal_key: str = Header(None)) -> None:
    """Validate service account API key."""
    if x_internal_key != settings.INTERNAL_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "INTERNAL_KEY_INVALID", "message": "Access denied."}
        )
```

---

## 5. Row-Level Security — Subscription Tier Enforcement

Row-Level Security is enforced at **two layers**: the PostgreSQL database (structural safety net) and the FastAPI application layer (primary control with user-friendly error messages). Both must be consistent.

### 5.1 PostgreSQL Row-Level Security Policies

```sql
-- ─────────────────────────────────────────────────────────────
-- Enable RLS on sensitive tables
-- ─────────────────────────────────────────────────────────────
ALTER TABLE users                   ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_subscriptions      ENABLE ROW LEVEL SECURITY;
ALTER TABLE match_predictions       ENABLE ROW LEVEL SECURITY;

-- ─────────────────────────────────────────────────────────────
-- USERS TABLE: each user sees only their own row
-- ─────────────────────────────────────────────────────────────
CREATE POLICY user_isolation ON users
    USING (id = current_setting('app.current_user_id', true)::INTEGER);

-- Admins can read all rows
CREATE POLICY admin_read_all_users ON users
    FOR SELECT
    USING (current_setting('app.current_user_role', true) = 'admin');

-- ─────────────────────────────────────────────────────────────
-- USER_SUBSCRIPTIONS: users see only their own subscription
-- ─────────────────────────────────────────────────────────────
CREATE POLICY subscription_isolation ON user_subscriptions
    USING (user_id = current_setting('app.current_user_id', true)::INTEGER);

CREATE POLICY admin_read_all_subscriptions ON user_subscriptions
    FOR SELECT
    USING (current_setting('app.current_user_role', true) = 'admin');

-- ─────────────────────────────────────────────────────────────
-- MATCH_PREDICTIONS: knockout-stage rows hidden from free users
-- ─────────────────────────────────────────────────────────────
-- Free users cannot query predictions for knockout-stage matches
CREATE POLICY free_tier_prediction_access ON match_predictions
    FOR SELECT
    USING (
        -- Always allow if user is pro or admin
        current_setting('app.current_user_tier', true) IN ('pro', 'admin')
        OR
        -- Allow free users only for group-stage matches
        (current_setting('app.current_user_tier', true) = 'free'
         AND (
             SELECT m.round FROM matches m
             WHERE m.id = match_predictions.match_id
         ) = 'group'
        )
        OR
        -- Service account always allowed (for pipeline writes)
        current_setting('app.current_user_role', true) = 'service_account'
    );
```

**Setting session-level RLS variables in FastAPI (via SQLAlchemy):**
```python
# api/db/session.py

async def get_db_session(user: TokenPayload) -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        # Set PostgreSQL session variables for RLS policies
        await session.execute(
            text("SELECT set_config('app.current_user_id',   :uid,  true)"),
            {"uid": str(user.user_id)}
        )
        await session.execute(
            text("SELECT set_config('app.current_user_role', :role, true)"),
            {"role": user.role}
        )
        await session.execute(
            text("SELECT set_config('app.current_user_tier', :tier, true)"),
            {"tier": user.tier}
        )
        try:
            yield session
        finally:
            await session.close()
```

### 5.2 Application-Layer Field-Level Security

RLS controls which **rows** a user can see. Field-level security controls which **columns** within a row are returned. This is enforced in the FastAPI response schemas.

```python
# api/schemas/predictions.py

class MatchPredictionFreeResponse(BaseModel):
    """Returned for free_user tier — feature importances stripped."""
    match_id: int
    home_team: str
    away_team: str
    match_date: datetime
    round: str
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    predicted_outcome: str
    confidence_score: float
    model_version: str
    last_updated_at: datetime
    # feature_importances deliberately excluded

class MatchPredictionProResponse(MatchPredictionFreeResponse):
    """Returned for pro_user and admin — includes full payload."""
    feature_importances: dict[str, float]   # top 10 features + weights
    head_to_head: HeadToHeadStats
    calibration_note: str | None            # e.g. "Low confidence — no H2H history"

def get_prediction_schema(user: TokenPayload) -> type[BaseModel]:
    """Return the appropriate schema for the user's tier."""
    if user.tier in ("pro", "admin"):
        return MatchPredictionProResponse
    return MatchPredictionFreeResponse
```

### 5.3 Tier Feature Flag Enforcement

Feature flags are stored in `subscription_tiers.feature_flags` (JSONB) and embedded in the JWT. Application code reads from the JWT claim — not the database — on every request.

```python
# api/dependencies.py

def require_feature(feature_name: str):
    """Factory: returns a dependency that enforces a specific feature flag."""
    def _check(user: TokenPayload = Depends(get_current_user)) -> TokenPayload:
        if not user.tier_features.get(feature_name, False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "FEATURE_NOT_AVAILABLE",
                    "feature": feature_name,
                    "message": f"'{feature_name}' is not available on your current plan.",
                    "upgrade_url": "/api/v1/subscriptions/checkout"
                }
            )
        return user
    return _check

# Usage in router:
@router.get("/predictions/export")
async def export_predictions(
    user: TokenPayload = Depends(require_feature("export_csv")),
    db: AsyncSession = Depends(get_db_session_from_token),
):
    ...
```

### 5.4 Stripe Webhook → Tier Synchronization

When a user's subscription changes (payment failure, cancellation, upgrade), Stripe sends a webhook. The tier_features embedded in their JWT become stale until their next refresh. To minimize the lag:

- On **subscription cancelled / payment failed**: write a `tier_revocation` entry to Redis with key `revoked:user:{user_id}` and TTL = 60 min. All API requests for this user check this key and downgrade claims in real time, even before the JWT expires.
- On **upgrade to Pro**: no active revocation needed — the user simply refreshes their token and gets the updated `tier_features` in the new JWT.

```python
# api/routers/webhooks.py

@router.post("/webhooks/stripe")
async def stripe_webhook(request: Request, stripe_signature: str = Header(alias="Stripe-Signature")):
    body = await request.body()

    try:
        event = stripe.Webhook.construct_event(body, stripe_signature, settings.STRIPE_WEBHOOK_SECRET)
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail={"code": "WEBHOOK_SIGNATURE_INVALID"})

    if event["type"] in ("customer.subscription.deleted", "invoice.payment_failed"):
        user_id = get_user_id_from_stripe_customer(event["data"]["object"]["customer"])
        await redis_client.setex(f"revoked:user:{user_id}", 3600, "downgrade")

    return {"received": True}
```

---

## 6. Rate Limiting Rules

Rate limiting uses **slowapi** (FastAPI-compatible, backed by Redis). Limits are per-IP for anonymous, per-user-ID for authenticated.

### 6.1 Limit Table

| Endpoint Group | anonymous | free_user | pro_user | admin | service_account |
|---|---|---|---|---|---|
| Auth — `POST /auth/login` | 5/min, 20/hr | N/A | N/A | N/A | N/A |
| Auth — `POST /auth/register` | 3/hr, 10/day | N/A | N/A | N/A | N/A |
| Auth — `POST /auth/refresh` | 10/hr | 30/hr | 30/hr | Unlimited | N/A |
| Auth — `POST /auth/change-password` | N/A | 3/hr | 3/hr | 3/hr | N/A |
| Read — match predictions | 10/min | 60/min | 120/min | Unlimited | Unlimited |
| Read — accuracy tracker | 5/min | 30/min | 60/min | Unlimited | Unlimited |
| Read — exports (CSV) | N/A | N/A | 5/hr | Unlimited | N/A |
| Read — bracket simulator | N/A | N/A | 10/hr | Unlimited | N/A |
| Admin endpoints | N/A | N/A | N/A | 30/min | N/A |
| Internal endpoints | N/A | N/A | N/A | N/A | 30/min |
| Stripe webhooks | — | — | — | — | Unlimited (IP: Stripe only) |

### 6.2 Rate Limit Response

When a limit is exceeded, return:
```json
HTTP 429 Too Many Requests
Retry-After: 47
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1719392400

{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "You've made too many requests. Please wait 47 seconds before trying again.",
    "retry_after_seconds": 47,
    "limit": 60,
    "window": "60 seconds"
  }
}
```

### 6.3 Progressive Auth Lockout (Brute Force Protection)

Login failures trigger exponential lockout on the **account** (not just IP — prevents IP-rotation attacks):

| Consecutive failures | Action |
|---|---|
| 1–4 | Log only |
| 5 | 15-minute lockout; email alert to user |
| 6–9 | Lockout extended per attempt |
| 10+ | Account locked; email with unlock link; security team alerted |
| After unlock | MFA required for next 7 days regardless of MFA preference |

---

## 7. Standardized Error Response Contract

All API errors — regardless of origin (auth, validation, pipeline, ML) — return the same JSON envelope. This ensures the Streamlit frontend can handle errors uniformly.

### 7.1 Error Envelope Schema

```json
{
  "error": {
    "code":         "SNAKE_CASE_ERROR_CODE",
    "message":      "Human-readable description safe to display to users.",
    "detail":       "Optional: technical detail for logging (NOT shown in Streamlit UI).",
    "suggestion":   "Optional: what the user or developer can do next.",
    "http_status":  404,
    "request_id":   "req_7f3a1b2c4d5e6f7a",
    "timestamp":    "2026-06-26T14:30:00Z",
    "docs_url":     "https://docs.worldcup-platform.com/errors/PREDICTION_NOT_FOUND"
  }
}
```

**Rules:**
- `message` is safe for end-user display. Never include stack traces, SQL errors, or file paths.
- `detail` is logged server-side and returned only in `APP_ENV=development`. Stripped in production.
- `request_id` is generated per request (UUID4) and logged — enables cross-referencing logs with user-reported errors.
- `code` is always a SCREAMING_SNAKE_CASE string. Clients should match on `code`, never on `message` (messages can be updated for clarity).

### 7.2 Master Error Code Registry

| HTTP Status | Error Code | Trigger | User-Safe Message |
|---|---|---|---|
| 400 | `INVALID_REQUEST` | Malformed JSON body | "The request was malformed. Please check the request format." |
| 400 | `INVALID_DATE_FORMAT` | Date string not ISO 8601 | "Date must be in YYYY-MM-DD format." |
| 400 | `MATCH_NOT_YET_PLAYED` | Requesting stats for a future match | "This match hasn't been played yet. Predictions are available; live stats will appear after the final whistle." |
| 400 | `KNOCKOUT_DRAW_INVALID` | Requesting draw prediction for knockout match | "Draw outcomes don't apply to knockout rounds. Only Win or Loss probabilities are shown." |
| 401 | `TOKEN_MISSING` | No Authorization header | "Please log in to access this content." |
| 401 | `TOKEN_EXPIRED` | Access token expired | "Your session has expired. Please refresh or log in again." |
| 401 | `TOKEN_INVALID` | Malformed or tampered JWT | "Invalid session token. Please log in again." |
| 401 | `REFRESH_TOKEN_EXPIRED` | Refresh token past 30 days | "Your session has expired. Please log in again." |
| 401 | `REFRESH_TOKEN_REUSE` | Stolen token reuse detected | "A security event was detected on your account. Please log in again." |
| 401 | `MFA_REQUIRED` | Admin accessing endpoint without completed MFA | "Multi-factor authentication is required for this action." |
| 403 | `TIER_INSUFFICIENT` | Free user accessing Pro feature | "This feature requires a Pro subscription." |
| 403 | `FEATURE_NOT_AVAILABLE` | Specific feature flag off | "This feature is not available on your current plan." |
| 403 | `ADMIN_REQUIRED` | Non-admin accessing admin endpoint | "Access denied." |
| 403 | `INTERNAL_KEY_INVALID` | Bad or missing X-Internal-Key | "Access denied." |
| 403 | `OWN_RECORD_ONLY` | User requesting another user's data | "Access denied." |
| 404 | `MATCH_NOT_FOUND` | Invalid match_id | "No match found with this ID." |
| 404 | `PREDICTION_NOT_FOUND` | Match exists but no prediction generated yet | "No prediction is available for this match yet. The model may still be processing." |
| 404 | `TEAM_NOT_FOUND` | Invalid team_id | "No team found with this ID." |
| 404 | `USER_NOT_FOUND` | Invalid user_id (admin endpoint) | "No user found with this ID." |
| 404 | `MODEL_VERSION_NOT_FOUND` | Invalid model version | "No model version found with this ID." |
| 409 | `EMAIL_ALREADY_REGISTERED` | Duplicate registration | "This email address is already registered. Try logging in." |
| 409 | `USERNAME_TAKEN` | Duplicate username | "This username is already taken. Please choose another." |
| 409 | `SUBSCRIPTION_ALREADY_ACTIVE` | Upgrading when already Pro | "You already have an active Pro subscription." |
| 422 | `VALIDATION_ERROR` | Pydantic validation failure | "One or more fields failed validation." (with field-level details) |
| 429 | `RATE_LIMIT_EXCEEDED` | Too many requests | "Too many requests. Please wait before trying again." |
| 500 | `INTERNAL_SERVER_ERROR` | Unhandled exception | "Something went wrong on our end. Our team has been notified." |
| 500 | `DATABASE_UNAVAILABLE` | Postgres connection failure | "Our database is temporarily unavailable. Please try again shortly." |
| 500 | `CACHE_UNAVAILABLE` | Redis connection failure | "Cache layer unavailable. Predictions may be slower than usual." (non-fatal, falls through to DB) |
| 500 | `MODEL_LOAD_FAILED` | joblib.load() failure | "The prediction model is temporarily unavailable. Please try again shortly." |
| 502 | `UPSTREAM_API_ERROR` | Football data API returned unexpected response | "Our data source returned an unexpected response. We'll retry automatically." |
| 503 | `PIPELINE_PROCESSING` | Pipeline currently running — data temporarily locked | "Predictions are being updated after the latest match result. Please check back in a few minutes." |
| 503 | `SERVICE_UNAVAILABLE` | Deployment in progress / maintenance | "The service is temporarily undergoing maintenance. It will be back shortly." |

---

## 8. Error Handling Guide — API Layer

### 8.1 Global Exception Handler (FastAPI)

```python
# api/main.py

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uuid, logging

logger = logging.getLogger(__name__)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "One or more fields failed validation.",
                "fields": [
                    {
                        "field": " → ".join(str(loc) for loc in e["loc"]),
                        "issue": e["msg"],
                    }
                    for e in exc.errors()
                ],
                "http_status": 422,
                "request_id": str(uuid.uuid4()),
                "timestamp": utc_now_iso(),
            }
        },
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    request_id = str(uuid.uuid4())
    logger.exception(f"Unhandled exception [request_id={request_id}]", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "Something went wrong on our end. Our team has been notified.",
                "http_status": 500,
                "request_id": request_id,
                "timestamp": utc_now_iso(),
            }
        },
    )
```

### 8.2 Auth Error Flows

**Expired access token (401 TOKEN_EXPIRED):**
- Streamlit catches 401 from any API call
- Automatically calls `POST /auth/refresh` with the stored httpOnly cookie
- If refresh succeeds: retry the original request with the new access token (transparent to the user)
- If refresh fails (expired, reused, revoked): clear `st.session_state`, show the login page with message "Your session has expired. Please log in again."

**Stolen token reuse (401 REFRESH_TOKEN_REUSE):**
- All sessions for the affected user are invalidated server-side (delete all rows from `refresh_tokens` for this user_id)
- Return `REFRESH_TOKEN_REUSE` error to the current caller
- Log a `SECURITY_EVENT` entry to the audit log (see Section 15) with full context
- Send security alert email to the user
- Alert the security team channel (Slack webhook or email)

---

## 9. Error Handling Guide — Data Pipeline Layer

The data pipeline runs asynchronously. Errors here don't directly surface to users via HTTP — they affect prediction freshness. The Streamlit UI must communicate pipeline health clearly without alarming users unnecessarily.

### 9.1 Pipeline Error Categories

| Category | Description | Severity | Auto-recovery |
|---|---|---|---|
| `RESULT_NOT_YET_AVAILABLE` | Match completed but API hasn't published result | Low | ✅ Retry on next poll cycle |
| `UPSTREAM_API_TIMEOUT` | Football data API didn't respond within 10s | Medium | ✅ Retry with exponential backoff (tenacity) |
| `UPSTREAM_API_RATE_LIMITED` | 429 from football-data.org | Medium | ✅ Respect Retry-After header |
| `UPSTREAM_API_AUTH_FAILED` | 401/403 from football data API | High | ❌ Alert required — key may be revoked |
| `UPSTREAM_API_UNEXPECTED_RESPONSE` | Response schema doesn't match Pydantic model | High | ❌ Alert required — API schema may have changed |
| `DATABASE_WRITE_FAILED` | Postgres error during result/prediction insert | Critical | ❌ Alert required — manual investigation |
| `MODEL_RETRAIN_FAILED` | scikit-learn training raised an exception | High | ⚠️ Keep serving previous model; alert |
| `MODEL_ACCURACY_BELOW_THRESHOLD` | New model accuracy < `MODEL_MIN_ACCURACY_THRESHOLD` | High | ⚠️ Keep previous model; alert |
| `MODEL_ARTIFACT_WRITE_FAILED` | S3 or local disk write for .joblib failed | Critical | ❌ Alert required |
| `PREDICTION_GENERATION_FAILED` | predict() call raised exception on upcoming matches | Critical | ❌ Alert required — predictions stale |

### 9.2 Pipeline Error Response Strategy

```python
# ingestion/pipeline.py

async def run_post_game_pipeline(match_id: int) -> PipelineRunResult:
    run = PipelineRun(trigger="post-match", match_id=match_id)
    try:
        # Step 1: Ingest result
        result = await fetch_match_result(match_id)
        if result is None:
            log.warning(f"[{match_id}] Result not yet published. Will retry next poll.")
            return run.finish(status="pending", error="RESULT_NOT_YET_AVAILABLE")

        await save_match_result(result)

        # Step 2: Retrain model
        new_model = retrain_model()
        if new_model.accuracy < settings.MODEL_MIN_ACCURACY_THRESHOLD:
            log.error(f"New model accuracy {new_model.accuracy:.3f} below threshold. Keeping previous model.")
            alert_team("MODEL_ACCURACY_BELOW_THRESHOLD", details=new_model)
            return run.finish(status="partial", error="MODEL_ACCURACY_BELOW_THRESHOLD")

        # Step 3: Save and activate model
        save_model_artifact(new_model)
        activate_model_version(new_model.version_id)

        # Step 4: Regenerate predictions
        updated_count = regenerate_upcoming_predictions()

        # Step 5: Invalidate Redis cache
        await invalidate_prediction_cache()

        # Step 6: Update last_updated Redis key
        await redis_client.set("pipeline:last_updated", utc_now_iso())

        return run.finish(status="success", predictions_updated=updated_count)

    except httpx.TimeoutException:
        log.error("Football data API timed out.")
        alert_team("UPSTREAM_API_TIMEOUT")
        return run.finish(status="failed", error="UPSTREAM_API_TIMEOUT")

    except Exception as e:
        log.exception("Unexpected pipeline failure.")
        alert_team("PIPELINE_UNEXPECTED_FAILURE", error=str(e))
        return run.finish(status="failed", error="INTERNAL_ERROR")

    finally:
        await save_pipeline_run(run)   # Always write to pipeline_runs table
```

### 9.3 UI Behaviour During Pipeline Processing

When the pipeline is running, a brief window exists where match results are ingested but predictions are not yet updated. The API returns `503 PIPELINE_PROCESSING` for prediction endpoints during this window. Streamlit handles this gracefully:

```python
# app/pages/1_Upcoming_Matches.py

try:
    predictions = api.get_upcoming_predictions()
    last_updated = api.get_last_updated_at()
    st.caption(f"🟢 Predictions current as of {last_updated}")
    render_predictions(predictions)

except APIError as e:
    if e.code == "PIPELINE_PROCESSING":
        st.info("⏳ Predictions are being refreshed after the latest match result. Check back in a minute or two.")
    elif e.code == "PREDICTION_NOT_FOUND":
        st.warning("⚠️ No predictions available yet for this match.")
    elif e.code == "DATABASE_UNAVAILABLE":
        st.error("🔴 Our data service is temporarily unavailable. Please try again shortly.")
    else:
        st.error(f"Something went wrong ({e.code}). Please refresh the page.")
```

### 9.4 Stale Prediction Warning

If the pipeline hasn't run within `PIPELINE_POLL_INTERVAL_MINUTES * 3` (45 minutes by default) since a match was completed, the UI displays a staleness warning:

```
⚠️  Predictions may be out of date — last updated 52 minutes ago.
    A match result may still be processing.
```

This is determined by comparing `pipeline:last_updated` (Redis key) against the latest completed match timestamp.

---

## 10. Error Handling Guide — ML Model Layer

### 10.1 Cold Start (No Trained Model Available)

Occurs at initial deployment before the historical ETL has run and the first model has been trained.

```
Detection: No row in model_versions where is_active = TRUE.

API behaviour:
  GET /api/v1/matches/upcoming → returns matches list WITHOUT prediction fields.
  GET /api/v1/matches/{id}/prediction → 404 PREDICTION_NOT_FOUND with
    message: "Our model is still being trained on historical data. Predictions
              will be available shortly."

UI behaviour:
  Show matches list with "Predictions coming soon" placeholder badges.
  Do NOT show empty probability bars — empty UI is worse than a clear message.

Resolution:
  Triggered automatically when ml/train.py completes and
  POST /internal/pipeline/retrain returns 200.
```

### 10.2 Model Load Failure

```python
# ml/predict.py

_model_cache = None

def load_active_model() -> Pipeline:
    global _model_cache
    if _model_cache is not None:
        return _model_cache  # In-memory cache — don't reload on every prediction call

    try:
        model_version = get_active_model_version_from_db()
        _model_cache = joblib.load(model_version.artifact_path)
        return _model_cache
    except FileNotFoundError:
        log.critical(f"Model artifact not found at {model_version.artifact_path}")
        raise ModelLoadError("MODEL_ARTIFACT_MISSING")
    except Exception as e:
        log.critical(f"Failed to load model: {e}")
        raise ModelLoadError("MODEL_LOAD_FAILED")
```

When `ModelLoadError` is raised, the FastAPI exception handler returns `500 MODEL_LOAD_FAILED`. An alert is fired immediately.

### 10.3 Prediction Failure for a Specific Match

```python
# ml/predict.py

def predict_match(match_features: dict) -> PredictionResult:
    try:
        model = load_active_model()
        X = build_feature_vector(match_features)
        proba = model.predict_proba(X)[0]
        classes = model.classes_
        return PredictionResult(
            home_win_prob=float(proba[classes.tolist().index("home_win")]),
            draw_prob=float(proba[classes.tolist().index("draw")]),
            away_win_prob=float(proba[classes.tolist().index("away_win")]),
        )
    except KeyError as e:
        # A required feature is missing for this match
        log.warning(f"Missing feature {e} for match. Returning low-confidence prediction.")
        return PredictionResult(
            home_win_prob=0.40, draw_prob=0.25, away_win_prob=0.35,
            confidence_note="Incomplete data — treat this prediction with caution.",
            is_low_confidence=True
        )
    except Exception as e:
        log.error(f"Prediction failed: {e}")
        raise PredictionError("PREDICTION_COMPUTATION_FAILED")
```

**Low-confidence predictions** (missing features, no H2H history, first-time 48-team matchup) are served with a `calibration_note` field that the UI renders as a yellow warning banner: *"Limited historical data for this matchup — wider uncertainty than usual."*

### 10.4 Knockout Round Draw Handling

The model is trained on all World Cup matches including group-stage draws. In knockout rounds, a "draw" outcome is logically impossible in the final prediction (teams play ET and pens until a winner is decided).

```python
def postprocess_knockout_prediction(result: PredictionResult) -> PredictionResult:
    """
    In knockout rounds, redistribute draw probability proportionally
    between home and away win, based on the ratio of their raw probabilities.
    """
    total_non_draw = result.home_win_prob + result.away_win_prob
    if total_non_draw == 0:
        # Edge case: model returned 0/0/1 draw. Split 50/50.
        return PredictionResult(home_win_prob=0.5, draw_prob=0.0, away_win_prob=0.5)

    scale = 1.0 / total_non_draw  # Normalize
    return PredictionResult(
        home_win_prob=round(result.home_win_prob * scale, 4),
        draw_prob=0.0,
        away_win_prob=round(result.away_win_prob * scale, 4),
        calibration_note="Draw probability redistributed — knockout round."
    )
```

---

## 11. Error Handling Guide — External API Layer

### 11.1 Football Data API Failure Modes

| HTTP Status from API | Our Code | Handling |
|---|---|---|
| 200 but empty `matches` array | `RESULT_NOT_YET_AVAILABLE` | Retry in 15 min; do not retrain |
| 400 Bad Request | `UPSTREAM_API_BAD_REQUEST` | Log request params, alert team — likely a URL/param bug |
| 401 Unauthorized | `UPSTREAM_API_AUTH_FAILED` | Alert team immediately — API key may have expired |
| 403 Forbidden | `UPSTREAM_API_FORBIDDEN` | Alert team — plan limit may be reached |
| 404 Not Found | `UPSTREAM_RESOURCE_NOT_FOUND` | Log and skip — match may not exist in this API's data |
| 429 Too Many Requests | `UPSTREAM_RATE_LIMITED` | Honour `Retry-After` header. Switch to failover API (API-Football) if sustained |
| 500/503 from API | `UPSTREAM_SERVER_ERROR` | Retry with exponential backoff (3 attempts, max 32s). Then failover |
| httpx.TimeoutException | `UPSTREAM_TIMEOUT` | Timeout = 10s. 3 retries, then failover |
| Response schema mismatch | `UPSTREAM_SCHEMA_CHANGED` | Alert team — API may have updated their response format. Manual fix required |

### 11.2 Failover to API-Football

When `football-data.org` returns sustained errors (3+ consecutive failures within 15 minutes), the pipeline automatically switches to the `API-Football` failover:

```python
# ingestion/live_poller.py

PROVIDERS = [
    FootballDataOrgProvider(api_key=settings.FOOTBALL_DATA_API_KEY),
    APIFootballProvider(api_key=settings.API_FOOTBALL_KEY),
]

async def fetch_with_failover(match_id: int) -> MatchResult | None:
    for provider in PROVIDERS:
        if await provider.is_healthy():
            try:
                return await provider.fetch_result(match_id)
            except UpstreamError as e:
                log.warning(f"Provider {provider.name} failed: {e}. Trying next.")
    log.error("All providers failed. Cannot ingest result.")
    alert_team("ALL_UPSTREAM_PROVIDERS_FAILED", match_id=match_id)
    return None
```

### 11.3 Stripe Webhook Errors

| Failure | Handling |
|---|---|
| Signature verification fails | Return `400`. Log with request body hash. Do not process event. |
| Event type not handled | Return `200` (Stripe requires 200 for unhandled events to avoid retries). Log the type. |
| DB write fails during tier update | Return `500`. Stripe will retry the webhook up to 72 hours. Use idempotency key (`event.id`) to prevent duplicate processing on retry. |
| User not found for Stripe customer | Log `ORPHANED_STRIPE_CUSTOMER` alert. Manual reconciliation required. |

---

## 12. Edge Cases — Data Platform Specific

### 12.1 No Head-to-Head History Between Two Teams

**Scenario:** The model is asked to predict a match between two teams that have never met in a World Cup (e.g., a team making its tournament debut).

**Handling:**
- `h2h_home_wins`, `h2h_away_wins`, `h2h_draws` features are set to `0` (not NULL — scikit-learn doesn't handle NaN well without explicit imputation).
- A `missing_features_flag = 1` feature is added to signal this to the model.
- The prediction is served with `is_low_confidence = true` and `calibration_note = "No head-to-head World Cup history between these teams. Prediction relies on FIFA rankings and current-form data only."`.
- The UI renders a yellow information banner, not a red warning — low confidence is informative, not an error.

### 12.2 Request for Stats on a Future Match

**Scenario:** A user requests `match_stats` (possession, shots, cards) for a match that hasn't been played yet.

```
Detection: match.status == "scheduled" AND request is for match_stats endpoint.

Response: 400 MATCH_NOT_YET_PLAYED
{
  "error": {
    "code": "MATCH_NOT_YET_PLAYED",
    "message": "This match hasn't been played yet. Stats will appear here after the final whistle.",
    "suggestion": "Looking for the prediction instead? Try GET /matches/{id}/prediction",
    "http_status": 400
  }
}

UI behaviour: Show the prediction panel (always available for future matches).
             Show a "Stats available after match" placeholder where stats would appear.
             Never show empty charts or zero-filled stat rows — explicit "not yet" > confusing zeros.
```

### 12.3 Request for Prediction on a Completed Match

**Scenario:** User requests a prediction for a match that's already been played.

```
Handling: Return the prediction that was made BEFORE the match (the original prediction).
          Include the actual outcome alongside it.
          This is the Accuracy Tracker data — valuable, not an error.

Response:
{
  "match_id": 42,
  "status": "completed",
  "prediction": {
    "predicted_outcome": "home_win",
    "home_win_prob": 0.61,
    "draw_prob": 0.21,
    "away_win_prob": 0.18,
    "model_version": "v1.2.0"
  },
  "actual_outcome": "away_win",
  "prediction_correct": false,
  "probability_of_actual": 0.18
}

Note: Never regenerate a prediction for a completed match.
      The historical prediction is immutable once the match result is recorded.
```

### 12.4 Empty Dataset at Initial Deployment (No Historical Data Loaded)

```
Detection: SELECT COUNT(*) FROM matches = 0.

API behaviour:
  ALL prediction endpoints return 503 with code "DATA_NOT_LOADED":
  "Historical match data has not yet been loaded. The platform will be
   available shortly."

Admin panel: Shows a banner "⚠️ Historical ETL has not been run."
             with a "Trigger ETL" button (admin only).

Pipeline scheduler: Paused until historical data is confirmed loaded.
                    Prevents pipeline from running on an empty dataset.
```

### 12.5 Match Result Ingested But Teams Don't Match Expected Bracket Position

**Scenario (2026-specific):** The new 48-team Round of 32 creates scenarios where the third-place qualifier slot is determined by points/goal difference across multiple groups. The pipeline may receive a result before the bracket is fully resolved.

```
Handling:
1. Ingest the match result into `matches` as normal.
2. Before triggering retraining: verify that both teams' bracket positions
   are confirmed (i.e., both team_id values are non-NULL in the upcoming
   scheduled matches).
3. If a team_id is still a placeholder (TBD): defer retraining.
   Log "BRACKET_POSITION_UNRESOLVED" warning.
   Set a retry job for 30 minutes.
4. Once both positions are confirmed: run the full pipeline.

Prediction note for "TBD" matchups:
  If a future match has home_team_id = NULL or away_team_id = NULL (not yet
  determined), the API returns the match in the schedule with
  predicted_outcome = null and a note:
  "Teams for this fixture are not yet confirmed. Prediction will appear
   once the bracket position is resolved."
```

### 12.6 Pipeline Runs During a Live Match

**Scenario:** APScheduler fires the 15-minute poll while a match is in progress (status = "live").

```
Handling:
1. Poller fetches match status. If status = "IN_PLAY" or "HALF_TIME":
   - Do NOT ingest partial scores as a result.
   - Do NOT retrain the model.
   - Do NOT overwrite pending predictions.
   - Log "MATCH_IN_PROGRESS — skipping pipeline run."
   - Schedule a single retry for 15 minutes later.
2. Only process a result when status = "FINISHED".

Failure mode to guard against:
  Some football APIs briefly return "FINISHED" with a wrong score before
  correcting it. Guard: wait for the result to be stable across 2 consecutive
  API polls (15 min apart) before treating it as final and triggering retraining.
```

### 12.7 All Remaining Matches Have One Team (Deep Knockout Run)

**Scenario:** A team reaches the final, meaning every remaining match prediction involves them. If their historical data is sparse (e.g., a small footballing nation making a historic run), feature sparsity compounds across all remaining predictions.

```
Handling:
  Flag all remaining predictions involving this team with is_low_confidence = true
  if their historical_matches_count < 5 in the training set.

  Recommendation surface:
  Show a contextual banner in the Match Detail view:
  "ℹ️ [Team] has limited World Cup history in our dataset. This prediction
     is based primarily on FIFA ranking and current-tournament form."
```

### 12.8 Model Predicts Identical Probability for All Three Outcomes

**Scenario:** Model returns `[0.333, 0.334, 0.333]` — effectively a coin flip. Can happen for evenly matched teams with no H2H history.

```
Detection: max(home_win_prob, draw_prob, away_win_prob) < 0.40

Handling:
  confidence_score = max probability value.
  If confidence_score < 0.40:
    predicted_outcome = "toss-up"   ← not "home_win"/"draw"/"away_win"
    calibration_note = "Our model sees this as nearly even. No confident prediction."
  
  UI: Show a three-way probability bar with "Too close to call" label instead
      of a winner badge. This is honest and educational — better than
      confidently picking the wrong team.
```

### 12.9 User Requests a Match from a Tournament That Doesn't Exist

```
Detection: tournament_id not in tournaments table, or year > current year + 1.

Response: 404 TOURNAMENT_NOT_FOUND
  "No tournament found for this year. The platform currently covers
   the 2026 FIFA World Cup. Data for future tournaments will be added
   when available."
```

### 12.10 Concurrent Stripe Webhook Events for the Same User

**Scenario:** A user cancels and immediately re-subscribes. Two webhook events arrive out of order (subscription.deleted arrives after subscription.created).

```
Handling:
  All Stripe webhook events are idempotent — use event.id as the
  idempotency key. Check if event.id already exists in stripe_events_log
  table before processing.

  For subscription state transitions, use event.created timestamp to
  determine the latest valid state, not arrival order:
  - Keep the subscription status that corresponds to the event with
    the HIGHEST event.created timestamp.
  - Do not apply an older event on top of a newer one.

  Table: stripe_events_log (event_id VARCHAR UNIQUE, processed_at TIMESTAMP)
```

---

## 13. Security Headers & Transport

### 13.1 Required HTTP Response Headers

```python
# api/middleware/security_headers.py

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["X-Content-Type-Options"]    = "nosniff"
        response.headers["X-Frame-Options"]           = "DENY"
        response.headers["X-XSS-Protection"]          = "1; mode=block"
        response.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"]        = "geolocation=(), microphone=(), camera=()"
        response.headers["Content-Security-Policy"]   = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "   # Streamlit requires this
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "connect-src 'self' https://api.stripe.com"
        )
        return response
```

### 13.2 Transport Security

- **TLS 1.2 minimum, TLS 1.3 preferred.** TLS 1.0 and 1.1 disabled at the load balancer.
- **HTTPS enforced everywhere.** `Strict-Transport-Security` header with 1-year max-age.
- **Internal service communication** (Streamlit → FastAPI) uses HTTPS even within the same deployment environment.
- **Database connections** use SSL mode `require` (`sslmode=require` in the SQLAlchemy connection string).
- **Redis connections** use TLS if provided by the managed Redis service.

---

## 14. Secrets & Credential Management

### 14.1 Secret Classification

| Secret | Rotation frequency | Storage |
|---|---|---|
| `SECRET_KEY` (JWT signing) | On every deployment (MVP); 90 days (prod) | Env var only |
| `DATABASE_URL` | On breach or 90 days | Env var only |
| `FOOTBALL_DATA_API_KEY` | On breach or API provider rotation schedule | Env var only |
| `API_FOOTBALL_KEY` | On breach | Env var only |
| `STRIPE_SECRET_KEY` | On breach | Env var only |
| `STRIPE_WEBHOOK_SECRET` | When Stripe endpoint changes | Env var only |
| `INTERNAL_API_KEY` | On every deployment | Env var only |
| `AWS_SECRET_ACCESS_KEY` | 90 days | Env var only (use IAM roles in production) |
| Stripe `live` keys | Before going live | Env var; never in staging |
| Stripe `test` keys | No rotation needed | Env var; never in production |

### 14.2 Hard Rules

- **Zero secrets in source code.** Pre-commit hook (`detect-secrets` or `truffleHog`) blocks commits containing API keys, passwords, or connection strings.
- **Zero secrets in logs.** Log formatters scrub known patterns (key=, password=, token=, authorization=). Use structured logging with explicit field allowlists.
- **Zero secrets in error responses.** Never include database connection strings, file paths, or internal hostnames in API error responses served to users.
- **`.env` files are gitignored** at the project root. `.env.example` contains only placeholder values and is committed.
- **Production secrets live only in the deployment platform's secret store** (Railway, Render, or AWS SSM Parameter Store). Engineers cannot see production secrets after initial setup.

---

## 15. Audit Logging

Every security-relevant action is written to an `audit_log` table. This is append-only — no UPDATE or DELETE is permitted, enforced via a PostgreSQL trigger.

### 15.1 Audit Log Schema

```sql
CREATE TABLE audit_log (
    id              BIGSERIAL PRIMARY KEY,
    event_type      VARCHAR(80)     NOT NULL,   -- See event types below
    actor_id        INTEGER,                    -- user_id (NULL for service account or anonymous)
    actor_role      VARCHAR(30),
    actor_ip        VARCHAR(45),                -- IPv4 or IPv6
    target_type     VARCHAR(50),                -- "user", "match", "model_version", etc.
    target_id       VARCHAR(100),               -- ID of the affected resource
    event_detail    JSONB,                      -- Additional context (old value, new value, etc.)
    request_id      VARCHAR(36),                -- Matches request_id in error envelope
    created_at      TIMESTAMP       NOT NULL DEFAULT NOW()
);

-- Prevent modification of audit records
CREATE RULE no_update_audit AS ON UPDATE TO audit_log DO INSTEAD NOTHING;
CREATE RULE no_delete_audit AS ON DELETE TO audit_log DO INSTEAD NOTHING;

CREATE INDEX idx_audit_actor ON audit_log(actor_id, created_at DESC);
CREATE INDEX idx_audit_event ON audit_log(event_type, created_at DESC);
```

### 15.2 Events Logged

| Event Type | Trigger | Key Detail Fields |
|---|---|---|
| `USER_REGISTERED` | New account created | `email`, `method` (password/oauth) |
| `USER_LOGIN_SUCCESS` | Successful login | `method`, `ip` |
| `USER_LOGIN_FAILURE` | Failed login attempt | `email_attempted`, `reason` |
| `USER_ACCOUNT_LOCKED` | 10 consecutive failures | `email`, `ip`, `failure_count` |
| `TOKEN_REFRESH` | Access token refreshed | N/A |
| `TOKEN_REUSE_DETECTED` | Stolen refresh token reuse | `user_id`, `ip` — **CRITICAL** |
| `MFA_ENABLED` | User enables TOTP | N/A |
| `MFA_DISABLED` | User disables TOTP | N/A |
| `MFA_FAILURE` | Bad TOTP code | `attempt_count` |
| `PASSWORD_CHANGED` | Password updated | N/A (never log passwords) |
| `SUBSCRIPTION_UPGRADED` | Free → Pro | `old_tier`, `new_tier`, `stripe_event_id` |
| `SUBSCRIPTION_DOWNGRADED` | Pro → Free (cancellation/expiry) | `reason`, `stripe_event_id` |
| `TIER_MANUAL_OVERRIDE` | Admin changes a user's tier | `admin_id`, `old_tier`, `new_tier` |
| `ADMIN_USER_VIEWED` | Admin reads a user record | `admin_id`, `viewed_user_id` |
| `MODEL_ACTIVATED` | New model version activated | `version_tag`, `accuracy` |
| `PIPELINE_RUN_STARTED` | Pipeline triggered | `trigger`, `match_id` |
| `PIPELINE_RUN_COMPLETED` | Pipeline finished | `status`, `predictions_updated` |
| `SECURITY_HEADER_VIOLATION` | CSP violation report | `violated_directive`, `blocked_uri` |
| `STRIPE_WEBHOOK_INVALID_SIGNATURE` | Bad Stripe webhook | `ip` |

---

## 16. Incident Response Playbook

### 16.1 Severity Levels

| Level | Definition | Response SLA |
|---|---|---|
| **P1 — Critical** | User data breach, stolen credentials, production DB down, ALL predictions unavailable | 15 min first response; 1 hr containment target |
| **P2 — High** | Prediction pipeline broken for >1 hour, Auth system down, Stripe integration broken | 1 hr first response |
| **P3 — Medium** | Single prediction endpoint returning errors, Redis cache down, External API failing | 4 hr first response |
| **P4 — Low** | Stale predictions (<24hr), minor UI errors, non-critical logging failures | Next business day |

### 16.2 P1 Playbook: Suspected Data Breach

```
1. DETECT
   Trigger: TOKEN_REUSE_DETECTED audit events > 5 unique users in 10 min
            OR external security report
            OR anomalous DB access patterns

2. CONTAIN (within 15 min)
   a. Revoke ALL active JWT refresh tokens:
      UPDATE refresh_tokens SET revoked_at = NOW() WHERE revoked_at IS NULL;
   b. Force re-authentication for all users.
   c. Rotate SECRET_KEY → all existing JWTs immediately invalid.
   d. Take API read-only (disable write endpoints) if DB compromise suspected.

3. INVESTIGATE (within 1 hr)
   a. Pull audit_log for last 24 hr, filter on suspicious event types.
   b. Check pipeline_runs for unauthorized trigger events.
   c. Review access logs for unusual IP patterns or large data exports.
   d. Determine scope: which users affected, which data exposed.

4. NOTIFY (within 72 hr of confirmed breach, per GDPR / applicable law)
   a. Notify affected users by email.
   b. Notify relevant Data Protection Authority if EU users are affected.
   c. Prepare breach report.

5. REMEDIATE
   a. Rotate all secrets (DB password, API keys, Stripe keys, JWT secret).
   b. Patch the vulnerability that allowed the breach.
   c. Audit all user records for unauthorized changes.
   d. Re-enable write endpoints after confirming remediation.

6. POST-MORTEM (within 1 week)
   Document: timeline, root cause, impact scope, remediation steps,
   process changes to prevent recurrence.
```

### 16.3 P2 Playbook: Prediction Pipeline Broken

```
1. DETECT
   Trigger: pipeline:last_updated Redis key > 45 min old AND
            a match has completed in that window.

2. DIAGNOSE
   a. Check pipeline_runs table for latest entry and error field.
   b. Check external API health (football-data.org status page).
   c. Check model_versions for is_active = TRUE record.
   d. Check S3/local artifacts directory for .joblib file.

3. MITIGATE
   If upstream API down → switch to failover API, run POST /internal/pipeline/run.
   If model artifact missing → run POST /internal/pipeline/retrain.
   If DB write failed → investigate Postgres logs, restore from backup if needed.

4. USER COMMUNICATION
   Set Redis key "pipeline:status_message" to a user-facing message.
   Streamlit banner: "⚠️ Predictions are temporarily delayed. We're working on it."

5. RESOLVE
   Confirm pipeline_runs shows a successful run.
   Confirm pipeline:last_updated is current.
   Clear "pipeline:status_message" key.
   Remove Streamlit banner.
```

---

*End of Security and Access Control Document v1.0*

*All code examples are illustrative. Security configurations must be reviewed by a qualified security engineer before production deployment. This document should be treated as a living document and reviewed after each major version release.*
