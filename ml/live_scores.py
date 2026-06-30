"""
Live score fetcher for FIFA World Cup 2026.
Uses worldcup26.ir free public API (no key required) to pull real match results.
Falls back gracefully to local fixture data if the API is unavailable.
"""

import json
import datetime
import os
from pathlib import Path
from typing import Optional, Dict, Any

DATA_DIR = Path(__file__).parent.parent / "data"

# Team name mappings between API names and our internal names
TEAM_NAME_MAP = {
    "USA": "USA",
    "United States": "USA",
    "Korea Republic": "South Korea",
    "South Korea": "South Korea",
    "IR Iran": "Iran",
    "Czechia": "Czechia",
    "Czech Republic": "Czechia",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "DR Congo": "DR Congo",
    "Congo DR": "DR Congo",
    "Democratic Republic of the Congo": "DR Congo",
    "Ivory Coast": "Ivory Coast",
    "Cote d'Ivoire": "Ivory Coast",
    "Cape Verde": "Cabo Verde",
    "Curaçao": "Curacao",
    "Curaao": "Curacao",
}

_live_cache: Optional[Dict] = None
_live_cache_ts: Optional[datetime.datetime] = None
CACHE_TTL_SECONDS = 15  # check more frequently for live scores (15s)

ROUND_TYPE_MAP = {
    "group": "Group Stage",
    "group_stage": "Group Stage",
    "r32": "Round of 32",
    "round_of_32": "Round of 32",
    "r16": "Round of 16",
    "round_of_16": "Round of 16",
    "qf": "Quarterfinal",
    "quarterfinal": "Quarterfinal",
    "sf": "Semifinal",
    "semifinal": "Semifinal",
    "third_place": "3rd Place",
    "final": "Final",
}

STADIUM_UTC_OFFSETS_JUNE = {
    "1": -6,   # Mexico City
    "2": -6,   # Guadalajara
    "3": -6,   # Monterrey
    "4": -5,   # Dallas
    "5": -5,   # Houston
    "6": -5,   # Kansas City
    "7": -4,   # Atlanta
    "8": -4,   # Miami
    "9": -4,   # Boston
    "10": -4,  # Philadelphia
    "11": -4,  # New York/New Jersey
    "12": -4,  # Toronto
    "13": -7,  # Vancouver
    "14": -7,  # Seattle
    "15": -7,  # San Francisco Bay Area
    "16": -7,  # Los Angeles
}


def _normalize_team(name: str) -> str:
    if not name:
        return name
    name = name.replace("\ufffd", "")
    return TEAM_NAME_MAP.get(name, name)


def _to_int(val, default=0) -> int:
    if val is None or str(val).lower() in ("null", "none", ""):
        return default
    try:
        return int(float(str(val)))
    except Exception:
        return default


def _score_or_none(val):
    if val is None or str(val).lower() in ("null", "none", "", "nan"):
        return None
    try:
        return int(float(str(val)))
    except Exception:
        return None


def _provider_date(local_date: str) -> str:
    raw_date = str(local_date or "").strip()
    if not raw_date:
        return ""
    if "T" in raw_date:
        return raw_date.split("T")[0]
    if " " in raw_date:
        raw_date = raw_date.split(" ")[0]
    if "/" in raw_date:
        parts = raw_date.split("/")
        if len(parts) == 3:
            return f"{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}"
    return raw_date[:10]


def _round_from_match(match: dict, default: str = "Group Stage") -> str:
    raw_type = str(match.get("type") or match.get("round") or "").strip().lower()
    if raw_type in ROUND_TYPE_MAP:
        return ROUND_TYPE_MAP[raw_type]

    raw_group = str(match.get("group") or "").strip()
    if raw_group and raw_group.upper() not in {"NAN", "NONE", "NULL"}:
        if raw_group.upper() == "R32":
            return "Round of 32"
        if len(raw_group) <= 2:
            return "Group Stage"

    return default


def _normalize_live_clock(match: dict, kick_off_utc: str = "") -> dict:
    """
    Build a display-safe live clock.

    Some feeds return time_elapsed='live' with stale kickoff metadata. In that
    case, do not estimate into impossible values like 90+93; show LIVE until a
    numeric minute or explicit HT/FT-style status is provided.
    """
    import datetime as dt

    elapsed_raw = str(match.get("time_elapsed") or match.get("minute") or "").strip().lower()
    status_raw = str(match.get("status") or "").strip().lower()

    if elapsed_raw in {"ht", "half-time", "halftime", "half time"} or status_raw in {"ht", "half-time", "halftime"}:
        return {"minute": "HT", "elapsed_mins": 45, "elapsed_secs": 0, "clock_is_ticking": False}

    if elapsed_raw in {"finished", "ft", "full_time", "full time"}:
        return {"minute": "FT", "elapsed_mins": 90, "elapsed_secs": 0, "clock_is_ticking": False}

    elapsed_mins = None
    elapsed_secs = 0
    cleaned = elapsed_raw.replace("'", "").strip()
    try:
        if cleaned and cleaned not in {"live", "ongoing", "in_progress", "null", "none"}:
            elapsed_mins = int(float(cleaned))
    except Exception:
        elapsed_mins = None

    # Only estimate from kickoff when the feed gives an actual kickoff instant.
    if elapsed_mins is None and kick_off_utc:
        try:
            kick_off = dt.datetime.fromisoformat(kick_off_utc.replace("Z", "+00:00"))
            now = dt.datetime.now(dt.timezone.utc)
            if now >= kick_off:
                elapsed_total = (now - kick_off).total_seconds()
                estimated = int(elapsed_total // 60)
                if 0 <= estimated <= 120:
                    elapsed_mins = estimated
                    elapsed_secs = int(elapsed_total % 60)
        except Exception:
            elapsed_mins = None

    # Public fallback sometimes sends time_elapsed='live' plus only a stadium-local
    # kickoff. Use that for a capped, display-safe estimate.
    if elapsed_mins is None and match.get("local_date") and match.get("stadium_id"):
        try:
            local_dt = dt.datetime.strptime(str(match.get("local_date")), "%m/%d/%Y %H:%M")
            offset = STADIUM_UTC_OFFSETS_JUNE.get(str(match.get("stadium_id")))
            if offset is not None:
                kick_off = (local_dt - dt.timedelta(hours=offset)).replace(tzinfo=dt.timezone.utc)
                now = dt.datetime.now(dt.timezone.utc)
                if now >= kick_off:
                    elapsed_total = (now - kick_off).total_seconds()
                    estimated = int(elapsed_total // 60)
                    if 0 <= estimated <= 130:
                        elapsed_mins = estimated
                        elapsed_secs = int(elapsed_total % 60)
        except Exception:
            elapsed_mins = None

    if elapsed_mins is None:
        return {"minute": "LIVE", "elapsed_mins": 0, "elapsed_secs": 0, "clock_is_ticking": False}

    # Treat normal half-time window as HT when the source has no stronger signal.
    if 45 <= elapsed_mins <= 60:
        return {"minute": "HT", "elapsed_mins": 45, "elapsed_secs": 0, "clock_is_ticking": False}

    if elapsed_mins > 60:
        elapsed_mins -= 15

    if elapsed_mins > 105:
        return {"minute": "LIVE", "elapsed_mins": 0, "elapsed_secs": 0, "clock_is_ticking": False}

    if elapsed_mins >= 90:
        extra = min(elapsed_mins - 90, 15)
        return {
            "minute": f"90+{extra}'",
            "elapsed_mins": 90 + extra,
            "elapsed_secs": elapsed_secs,
            "clock_is_ticking": extra < 15,
        }

    return {
        "minute": f"{elapsed_mins}'",
        "elapsed_mins": elapsed_mins,
        "elapsed_secs": elapsed_secs,
        "clock_is_ticking": True,
    }


def _parse_scorers_string(scorers_str) -> list:
    if not scorers_str or str(scorers_str).lower() == "null":
        return []
    s = str(scorers_str).strip()
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    
    # Clean up common unicode replacement character artifacts
    # First apply specific corrections using Unicode escapes
    s = s.replace("Demb\ufffdl\ufffd", "Dembélé")
    s = s.replace("D\ufffdsir\ufffd Dou\ufffd", "Désiré Doué")
    s = s.replace("Mbapp\ufffd", "Mbappé")
    s = s.replace("Isma\ufffdla", "Ismaïla")
    s = s.replace("Le\ufffdo", "Leão")
    s = s.replace("G\ufffdler", "Güler")
    s = s.replace("\ufffdstig\ufffdrd", "Østigård")
    s = s.replace("Kessi\ufffd", "Kessié")
    s = s.replace("Rub\ufffdn", "Rubén")
    s = s.replace("Ch\ufffdvez", "Chávez")
    s = s.replace("J\ufffdnior", "Júnior")
    s = s.replace("Vin\ufffdcius", "Vinícius")
    s = s.replace("San\ufffd", "Sané")
    s = s.replace("D\ufffdsir\ufffd", "Désiré")
    s = s.replace("Dou\ufffd", "Doué")
    s = s.replace("\ufffdlex", "Álex")
    
    # Fallback general replacement character cleanups
    s = s.replace("\ufffd", "")
    
    # Transliteration cleanup for known mangled Farsi names if any
    s = s.replace("Markvs Hlmgrn Pdrsn", "Marcus Holmgren Pedersen")
    s = s.replace("Kvdi Khakpv", "Cody Gakpo")
    s = s.replace("Dniz Avndav", "Deniz Undav")
    s = s.replace("Svfian Rhimi", "Soufiane Rahimi")
    s = s.replace("Asmaail Saibari", "Ismael Saibari")
    s = s.replace("Ali Avlvan", "Ali Alwan")
    s = s.replace("Ali Jast", "Ali Jasim")
    s = s.replace("Dnil Mvnvz", "Daniel Muñoz")
    s = s.replace("Lviiz Diaz", "Luis Díaz")
    s = s.replace("Khamintvn Kampaz", "Jaminton Campaz")
    s = s.replace("Abas Bk Fiz Allh Af", "Abbosbek Fayzullaev")
    s = s.replace("Jvhan Mnzambi", "Jovan Mnzambi")
    s = s.replace("Prvmis Divid", "Promise David")
    s = s.replace("Karim Alaibgvvich", "Karim Alai")
    s = s.replace("Taplv Maskv", "Thapelo Maseko")
    s = s.replace("Jvlian Kviinvnz", "Julián Quiñones")
    s = s.replace("lvaro Fidalgo", "Álvaro Fidalgo")
    s = s.replace("Alvaro Fidalgo", "Álvaro Fidalgo")
    s = s.replace("Nilsvn Angvlv", "Nilson Angulo")
    s = s.replace("Gvnzalv Plata", "Gonzalo Plata")
    s = s.replace("Hazm Mstvri", "Hazem Mastouri")
    s = s.replace("Alis Skhiri", "Ellyes Skhiri")
    s = s.replace("Ian Fn Hkh", "Jan Paul van Hecke")
    s = s.replace("Baris Alpr Ailmaz", "Barış Alper Yılmaz")
    s = s.replace("Kan Aihan", "Kaan Ayhan")
    s = s.replace("Paph Gviih", "Pape Gueye")
    s = s.replace("Ailman Andiaih", "Iliman Ndiaye")
    
    import re
    # Match double quoted names
    parts = re.findall(r'"([^"]*)"', s)
    if not parts:
        parts = [p.strip().strip('"').strip("'") for p in s.split(",") if p.strip()]
    return [p.replace("\\'", "'").strip() for p in parts if p]



def fetch_live_matches(force_refresh: bool = False) -> Optional[Dict[str, Any]]:
    """
    Fetch current match data from API-Football using user's key.
    Falls back to worldcup26.ir games API if API-Football fails or has no access.
    Falls back to local live_scores.json if both are offline.
    """
    global _live_cache, _live_cache_ts
    now = datetime.datetime.utcnow()

    # 0. Check local live scores config first to support local debugging/mocking
    live_path = DATA_DIR / "live_scores.json"
    if live_path.exists():
        try:
            with open(live_path, encoding="utf-8") as f:
                data = json.load(f)
                if data and (data.get("matches") or data.get("games")):
                    _live_cache = data
                    _live_cache_ts = now
                    print("[live_scores] Loaded mock/local matches from live_scores.json.")
                    return data
        except Exception as err:
            print(f"[live_scores] Failed to read live_scores.json: {err}")

    if (
        not force_refresh
        and _live_cache is not None
        and _live_cache_ts is not None
        and (now - _live_cache_ts).total_seconds() < CACHE_TTL_SECONDS
    ):
        return _live_cache

    # 1. Try API-Football first when a key is configured in the host environment.
    api_key = os.getenv("APISPORTS_KEY") or os.getenv("API_FOOTBALL_KEY")
    if api_key:
        try:
            import urllib.request
            url = "https://v3.football.api-sports.io/fixtures?league=1&season=2026"
            req = urllib.request.Request(
                url,
                headers={
                    "x-apisports-key": api_key,
                    "User-Agent": "Mozilla/5.0 WorldCupPredictor/1.0",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            # Check if the API returned a plan restriction error
            errors = data.get("errors", {})
            if errors and isinstance(errors, dict) and "plan" in errors:
                print(f"[live_scores] API-Football plan error: {errors['plan']}. Falling back to worldcup26.ir API.")
                raise Exception("API-Football Plan Error")

            if data.get("response"):
                _live_cache = data
                _live_cache_ts = now
                print("[live_scores] Successfully updated scores from API-Football.")
                return data
        except Exception as e:
            print(f"[live_scores] API-Football fetch failed/restricted: {e}.")
    else:
        print("[live_scores] APISPORTS_KEY/API_FOOTBALL_KEY not configured; using public fallback.")

    # 2. Try worldcup26.ir public API as the primary fallback
    try:
        import urllib.request
        url = "https://worldcup26.ir/get/games"
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 WorldCupPredictor/1.0",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
        _live_cache = data
        _live_cache_ts = now
        print("[live_scores] Successfully updated scores from fallback worldcup26.ir API.")
        return data
    except Exception as e:
        print(f"[live_scores] Fallback public API fetch failed: {e}.")

    # 3. Fall back to local live scores config if both networks fail
    live_path = DATA_DIR / "live_scores.json"
    if live_path.exists():
        try:
            with open(live_path, encoding="utf-8") as f:
                data = json.load(f)
                _live_cache = data
                _live_cache_ts = now
                return data
        except Exception as err:
            print(f"[live_scores] Failed to read live_scores.json: {err}")

    return None


def get_completed_live_matches() -> list:
    """
    Return list of completed matches from the live API with normalized team names.
    Each item: {date, home_team, away_team, home_score, away_score, status}
    """
    data = fetch_live_matches()
    if data is None:
        return []

    results = []
    
    # ── CASE 1: API-Football Format ───────────────────────────────────────────
    if isinstance(data, dict) and "response" in data:
        matches = data.get("response", [])
        for m in matches:
            try:
                fixture = m.get("fixture", {})
                status_raw = str(fixture.get("status", {}).get("short", "")).upper()
                long_status = str(fixture.get("status", {}).get("long", "")).upper()
                
                is_completed = (
                    status_raw in ("FT", "AET", "PEN", "FINISHED", "COMPLETED", "FULL_TIME")
                    or "FINISHED" in long_status
                )
                if not is_completed:
                    continue

                teams = m.get("teams", {})
                home = _normalize_team(teams.get("home", {}).get("name", ""))
                away = _normalize_team(teams.get("away", {}).get("name", ""))
                if not home or not away:
                    continue

                goals = m.get("goals", {})
                home_score = _to_int(goals.get("home"))
                away_score = _to_int(goals.get("away"))

                raw_date = fixture.get("date", "")
                date_str = str(raw_date)[:10]

                home_scorers = []
                away_scorers = []
                events = m.get("events", [])
                for ev in events:
                    if ev.get("type") == "Goal":
                        player = ev.get("player", {}).get("name", "")
                        time = ev.get("time", {}).get("elapsed", "")
                        detail = f"{player} {time}'"
                        if ev.get("team", {}).get("name") == teams.get("home", {}).get("name"):
                            home_scorers.append(detail)
                        else:
                            away_scorers.append(detail)

                results.append({
                    "source_match_id": str(m.get("fixture", {}).get("id") or ""),
                    "date": date_str,
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_penalty_score": _score_or_none(m.get("score", {}).get("penalty", {}).get("home")),
                    "away_penalty_score": _score_or_none(m.get("score", {}).get("penalty", {}).get("away")),
                    "status": "completed",
                    "round": _round_from_match(m),
                    "group": None,
                    "scorers": {
                        "home": home_scorers,
                        "away": away_scorers
                    }
                })
            except Exception as e:
                print(f"[live_scores] Failed to parse completed API-Football match: {e}")
                continue
        return results

    # ── CASE 2: list directly or other formats (worldcup26.ir or matches) ──────
    if isinstance(data, list):
        matches = data
    else:
        matches = data.get("games", data.get("matches", data.get("data", [])))

    for m in matches:
        try:
            status_raw = str(m.get("status", "")).upper()
            finished_raw = str(m.get("finished", "")).upper()
            elapsed_raw = str(m.get("time_elapsed", "")).lower()

            is_completed = (
                status_raw in ("FT", "AET", "PEN", "FINISHED", "COMPLETED", "FULL_TIME")
                or finished_raw == "TRUE"
                or elapsed_raw == "finished"
            )
            if not is_completed:
                continue

            h_raw = m.get("home_team")
            home_name = h_raw.get("name") if isinstance(h_raw, dict) else h_raw
            home = _normalize_team(m.get("home_team_name_en") or home_name or "")

            a_raw = m.get("away_team")
            away_name = a_raw.get("name") if isinstance(a_raw, dict) else a_raw
            away = _normalize_team(m.get("away_team_name_en") or away_name or "")
            
            if not home or not away:
                continue

            home_score = _to_int(m.get("home_score"))
            away_score = _to_int(m.get("away_score"))

            raw_date = m.get("date") or m.get("match_date") or m.get("datetime") or m.get("local_date", "")
            raw_date = str(raw_date).strip()
            if "T" in raw_date:
                raw_date = raw_date.split("T")[0]
            if " " in raw_date:
                raw_date = raw_date.split(" ")[0]
            if "/" in raw_date:
                parts = raw_date.split("/")
                if len(parts) == 3:
                    raw_date = f"{parts[2]}-{parts[0]}-{parts[1]}"
            date_str = raw_date[:10]
            
            home_scorers = _parse_scorers_string(m.get("home_scorers"))
            away_scorers = _parse_scorers_string(m.get("away_scorers"))
            
            results.append({
                "source_match_id": str(m.get("id") or ""),
                "date": date_str,
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "home_penalty_score": _score_or_none(m.get("home_penalty_score")),
                "away_penalty_score": _score_or_none(m.get("away_penalty_score")),
                "status": "completed",
                "round": _round_from_match(m),
                "group": m.get("group"),
                "home_team_label": m.get("home_team_label"),
                "away_team_label": m.get("away_team_label"),
                "scorers": {
                    "home": home_scorers,
                    "away": away_scorers
                }
            })
        except Exception as e:
            print(f"[live_scores] Failed to parse completed match: {e}")
            continue

    return results


def get_live_matches_data() -> list:
    """
    Return list of live/ongoing matches from the live scores config/API.
    """
    import datetime as dt
    data = fetch_live_matches()
    if data is None:
        return []
    
    results = []

    # ── CASE 1: API-Football Format ───────────────────────────────────────────
    if isinstance(data, dict) and "response" in data:
        matches = data.get("response", [])
        for m in matches:
            try:
                fixture = m.get("fixture", {})
                status_raw = str(fixture.get("status", {}).get("short", "")).upper()
                
                is_live = status_raw in ("1H", "2H", "HT", "ET", "P", "LIVE", "IN_PROGRESS", "ONGOING")
                if not is_live:
                    continue

                teams = m.get("teams", {})
                home = _normalize_team(teams.get("home", {}).get("name", ""))
                away = _normalize_team(teams.get("away", {}).get("name", ""))
                if not home or not away:
                    continue

                goals = m.get("goals", {})
                home_score = _to_int(goals.get("home"))
                away_score = _to_int(goals.get("away"))

                # Parse scorers from events
                scorers = {}
                events = m.get("events", [])
                if events:
                    home_list = []
                    away_list = []
                    for ev in events:
                        if ev.get("type") == "Goal":
                            player = ev.get("player", {}).get("name", "")
                            time = ev.get("time", {}).get("elapsed", "")
                            detail = f"{player} {time}'"
                            if ev.get("team", {}).get("name") == teams.get("home", {}).get("name"):
                                home_list.append(detail)
                            else:
                                away_list.append(detail)
                    scorers = {
                        home: home_list,
                        away: away_list
                    }

                elapsed_mins = _to_int(fixture.get("status", {}).get("elapsed", 0))
                clock = _normalize_live_clock({"time_elapsed": elapsed_mins, "status": status_raw})
                kick_off_utc = fixture.get("date", "")

                results.append({
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "status": "live",
                    "round": _round_from_match(m),
                    "group": None,
                    "minute": clock["minute"],
                    "elapsed_mins": clock["elapsed_mins"],
                    "elapsed_secs": clock["elapsed_secs"],
                    "clock_is_ticking": clock["clock_is_ticking"],
                    "kick_off_utc": kick_off_utc,
                    "scorers": scorers
                })
            except Exception as e:
                print(f"[live_scores] Failed to parse live API-Football match: {e}")
                continue
        return results

    # ── CASE 2: list directly or other formats ────────────────────────────────
    if isinstance(data, list):
        matches = data
    else:
        matches = data.get("games", data.get("matches", data.get("data", [])))

    for m in matches:
        try:
            status_raw = str(m.get("status", "")).upper()
            finished_raw = str(m.get("finished", "")).upper()
            elapsed_raw = str(m.get("time_elapsed", "")).lower()

            is_live = (
                status_raw in ("LIVE", "ONGOING", "IN_PROGRESS")
                or (finished_raw == "FALSE" and elapsed_raw not in ("notstarted", "", "null"))
            )
            if not is_live:
                continue
            
            h_raw = m.get("home_team")
            home_name = h_raw.get("name") if isinstance(h_raw, dict) else h_raw
            home = _normalize_team(m.get("home_team_name_en") or home_name or "")

            a_raw = m.get("away_team")
            away_name = a_raw.get("name") if isinstance(a_raw, dict) else a_raw
            away = _normalize_team(m.get("away_team_name_en") or away_name or "")
            
            if not home or not away:
                continue
                
            home_score = _to_int(m.get("home_score"))
            away_score = _to_int(m.get("away_score"))

            # Scorers
            scorers = m.get("scorers", {})
            if not scorers or not isinstance(scorers, dict):
                scorers = {}
            if "home_scorers" in m or "away_scorers" in m:
                home_list = _parse_scorers_string(m.get("home_scorers"))
                away_list = _parse_scorers_string(m.get("away_scorers"))
                if home_list or away_list:
                    scorers = {
                        home: home_list,
                        away: away_list
                    }

            kick_off_utc = m.get("kick_off_utc", "")
            clock = _normalize_live_clock(m, kick_off_utc=kick_off_utc)
            
            results.append({
                "source_match_id": m.get("id"),
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "status": "live",
                "round": _round_from_match(m),
                "group": m.get("group"),
                "minute": clock["minute"],
                "elapsed_mins": clock["elapsed_mins"],
                "elapsed_secs": clock["elapsed_secs"],
                "clock_is_ticking": clock["clock_is_ticking"],
                "kick_off_utc": kick_off_utc,
                "scorers": scorers
            })
        except Exception as e:
            print(f"[live_scores] Failed to parse live match: {e}")
            continue
    return results


def get_provider_knockout_fixtures(round_type: str = "r32") -> list:
    """Return provider knockout fixtures, including not-started games."""
    data = fetch_live_matches()
    if data is None:
        return []

    if isinstance(data, list):
        matches = data
    elif isinstance(data, dict):
        matches = data.get("games", data.get("matches", data.get("data", [])))
    else:
        return []

    wanted_round = ROUND_TYPE_MAP.get(round_type.lower(), ROUND_TYPE_MAP.get(round_type, "Round of 32"))
    results = []
    for m in matches:
        try:
            if _round_from_match(m, default="") != wanted_round:
                continue

            h_raw = m.get("home_team")
            home_name = h_raw.get("name") if isinstance(h_raw, dict) else h_raw
            home = _normalize_team(m.get("home_team_name_en") or home_name or "")

            a_raw = m.get("away_team")
            away_name = a_raw.get("name") if isinstance(a_raw, dict) else a_raw
            away = _normalize_team(m.get("away_team_name_en") or away_name or "")
            if not home or not away:
                continue

            finished_raw = str(m.get("finished", "")).upper()
            elapsed_raw = str(m.get("time_elapsed", "")).lower()
            status_raw = str(m.get("status", "")).upper()
            if finished_raw == "TRUE" or elapsed_raw == "finished":
                status = "completed"
            elif status_raw in ("LIVE", "ONGOING", "IN_PROGRESS") or (
                finished_raw == "FALSE" and elapsed_raw not in ("notstarted", "", "null")
            ):
                status = "live"
            else:
                status = "scheduled"

            home_score = _score_or_none(m.get("home_score"))
            away_score = _score_or_none(m.get("away_score"))
            home_penalty_score = _score_or_none(m.get("home_penalty_score"))
            away_penalty_score = _score_or_none(m.get("away_penalty_score"))
            if status == "scheduled":
                home_score = None
                away_score = None
                home_penalty_score = None
                away_penalty_score = None
            else:
                home_score = home_score or 0
                away_score = away_score or 0

            scorers = {}
            home_list = _parse_scorers_string(m.get("home_scorers"))
            away_list = _parse_scorers_string(m.get("away_scorers"))
            if home_list or away_list:
                scorers = {home: home_list, away: away_list}

            if status == "completed":
                clock = {"minute": None, "elapsed_mins": 0, "elapsed_secs": 0, "clock_is_ticking": False}
            elif status == "live":
                clock = _normalize_live_clock(m, kick_off_utc=m.get("kick_off_utc", ""))
            else:
                clock = {"minute": None, "elapsed_mins": 0, "elapsed_secs": 0, "clock_is_ticking": False}

            results.append({
                "source_match_id": str(m.get("id") or ""),
                "round": wanted_round,
                "date": _provider_date(m.get("local_date") or m.get("date") or ""),
                "home_team": home,
                "away_team": away,
                "home_score": home_score,
                "away_score": away_score,
                "home_penalty_score": home_penalty_score,
                "away_penalty_score": away_penalty_score,
                "status": status,
                "group": m.get("group"),
                "home_team_label": m.get("home_team_label"),
                "away_team_label": m.get("away_team_label"),
                "minute": clock["minute"],
                "elapsed_mins": clock["elapsed_mins"],
                "elapsed_secs": clock["elapsed_secs"],
                "clock_is_ticking": clock["clock_is_ticking"],
                "kick_off_utc": m.get("kick_off_utc", ""),
                "scorers": scorers,
                "sort_key": str(m.get("local_date") or ""),
            })
        except Exception as e:
            print(f"[live_scores] Failed to parse provider knockout fixture: {e}")

    return sorted(results, key=lambda x: (x["date"], x["sort_key"], x["source_match_id"]))


def sync_live_results_to_fixtures(fixtures_df, wc_df=None) -> tuple:
    """
    Merge live API results and ongoing live scores into the fixtures DataFrame.
    Returns (updated_fixtures_df, changed: bool)
    """
    updated = fixtures_df.copy()
    for col in [
        "scorers", "minute", "elapsed_mins", "elapsed_secs", "kick_off_utc",
        "clock_is_ticking", "source_match_id", "home_team_label", "away_team_label",
        "home_penalty_score", "away_penalty_score"
    ]:
        if col not in updated.columns:
            updated[col] = None
    changed = False
    newly_completed = []

    # 0. Sync provider knockout bracket slots before applying live/completed deltas.
    provider_r32 = get_provider_knockout_fixtures("r32")
    if provider_r32:
        r32_indices = list(updated[updated["round"] == "Round of 32"].sort_values("match_id").index)
        for slot_idx, provider in zip(r32_indices, provider_r32):
            idx = slot_idx
            was_completed = str(updated.at[idx, "status"]) == "completed"
            fields = [
                "round", "date", "home_team", "away_team", "status", "home_score",
                "away_score", "home_penalty_score", "away_penalty_score",
                "group", "minute", "elapsed_mins", "elapsed_secs",
                "kick_off_utc", "clock_is_ticking", "source_match_id",
                "home_team_label", "away_team_label"
            ]
            for key in fields:
                new_val = provider.get(key)
                old_val = updated.at[idx, key] if key in updated.columns else None
                old_cmp = "" if old_val is None or str(old_val).lower() == "nan" else str(old_val)
                new_cmp = "" if new_val is None or str(new_val).lower() == "nan" else str(new_val)
                if old_cmp != new_cmp:
                    updated.at[idx, key] = new_val
                    changed = True
            scorers_json = json.dumps(provider.get("scorers", {}))
            old_scorers = updated.at[idx, "scorers"]
            old_scorers_cmp = old_scorers if isinstance(old_scorers, str) else json.dumps(old_scorers or {})
            if old_scorers_cmp != scorers_json:
                updated.at[idx, "scorers"] = scorers_json
                changed = True
            if provider.get("status") == "completed" and not was_completed:
                newly_completed.append(provider)

        duplicate_mask = updated["source_match_id"].notna() & (updated["source_match_id"].astype(str).str.lower() != "nan") & (updated["source_match_id"].astype(str) != "")
        duplicate_ids = updated.loc[duplicate_mask, "source_match_id"][updated.loc[duplicate_mask, "source_match_id"].duplicated()].unique()
        if len(duplicate_ids) > 0:
            drop_indices = []
            for source_id in duplicate_ids:
                dupes = updated[updated["source_match_id"].astype(str) == str(source_id)].sort_values("match_id")
                keep_idx = dupes.index[0]
                drop_indices.extend([idx for idx in dupes.index if idx != keep_idx])
            if drop_indices:
                updated = updated.drop(index=drop_indices).reset_index(drop=True)
                changed = True

    # 1. Sync completed live matches
    live_completed = get_completed_live_matches()
    for live in live_completed:
        source_id = str(live.get("source_match_id") or "")
        if source_id and "source_match_id" in updated.columns:
            mask = updated["source_match_id"].astype(str) == source_id
        else:
            mask = (
                (updated["home_team"] == live["home_team"]) &
                (updated["away_team"] == live["away_team"])
            )
        if not mask.any():
            mask = (
                (updated["home_team"] == live["away_team"]) &
                (updated["away_team"] == live["home_team"])
            )
            if mask.any():
                live["home_team"], live["away_team"] = live["away_team"], live["home_team"]
                live["home_score"], live["away_score"] = live["away_score"], live["home_score"]
                live["home_penalty_score"], live["away_penalty_score"] = (
                    live.get("away_penalty_score"),
                    live.get("home_penalty_score"),
                )

        if not mask.any():
            continue

        idx = updated[mask].index[0]
        existing = updated.at[idx, "status"]
        existing_date = str(updated.at[idx, "date"])
        existing_h_score = updated.at[idx, "home_score"]
        existing_a_score = updated.at[idx, "away_score"]
        existing_h_pen = updated.at[idx, "home_penalty_score"]
        existing_a_pen = updated.at[idx, "away_penalty_score"]
        existing_scorers = updated.at[idx, "scorers"] if "scorers" in updated.columns else None
        
        has_no_scorers = False
        import pandas as pd
        if "scorers" in updated.columns:
            has_no_scorers = (
                existing_scorers is None 
                or pd.isna(existing_scorers) 
                or not existing_scorers 
                or str(existing_scorers) in ("[]", "{}", '{"home": [], "away": []}')
            )

        # Update if not completed OR if date differs OR if scores don't match OR if it lacks scorers
        if (
            existing != "completed" 
            or existing_date != live["date"]
            or existing_h_score != live["home_score"]
            or existing_a_score != live["away_score"]
            or existing_h_pen != live.get("home_penalty_score")
            or existing_a_pen != live.get("away_penalty_score")
            or has_no_scorers
        ):
            updated.at[idx, "status"] = "completed"
            updated.at[idx, "home_score"] = live["home_score"]
            updated.at[idx, "away_score"] = live["away_score"]
            updated.at[idx, "home_penalty_score"] = live.get("home_penalty_score")
            updated.at[idx, "away_penalty_score"] = live.get("away_penalty_score")
            updated.at[idx, "date"] = live["date"]
            updated.at[idx, "minute"] = None
            updated.at[idx, "elapsed_mins"] = 0
            updated.at[idx, "elapsed_secs"] = 0
            updated.at[idx, "clock_is_ticking"] = False
            if "scorers" in updated.columns:
                # Store scorers dict directly (save_2026_fixtures will format/loads it)
                updated.at[idx, "scorers"] = live.get("scorers")
            changed = True
            if existing != "completed":
                newly_completed.append(live)
        elif existing == "completed":
            for key, val in {
                "minute": None,
                "elapsed_mins": 0,
                "elapsed_secs": 0,
                "clock_is_ticking": False,
            }.items():
                old_val = updated.at[idx, key]
                old_cmp = "" if old_val is None or str(old_val).lower() == "nan" else str(old_val)
                new_cmp = "" if val is None else str(val)
                if old_cmp != new_cmp:
                    updated.at[idx, key] = val
                    changed = True

    # 2. Sync ongoing live matches
    live_ongoing = get_live_matches_data()
    for live in live_ongoing:
        source_id = str(live.get("source_match_id") or "")
        if source_id and "source_match_id" in updated.columns:
            mask = updated["source_match_id"].astype(str) == source_id
        else:
            mask = (
                (updated["home_team"] == live["home_team"]) &
                (updated["away_team"] == live["away_team"])
            )
        if not mask.any():
            mask = (
                (updated["home_team"] == live["away_team"]) &
                (updated["away_team"] == live["home_team"])
            )
            if mask.any():
                live["home_team"], live["away_team"] = live["away_team"], live["home_team"]
                live["home_score"], live["away_score"] = live["away_score"], live["home_score"]

        if not mask.any():
            import pandas as pd
            next_id = int(pd.to_numeric(updated.get("match_id"), errors="coerce").max() or 9000) + 1
            new_row = {
                "match_id": next_id,
                "round": live.get("round", "Group Stage"),
                "date": str(datetime.date.today()),
                "home_team": live["home_team"],
                "away_team": live["away_team"],
                "status": "live",
                "home_score": live["home_score"],
                "away_score": live["away_score"],
                "group": live.get("group"),
                "scorers": json.dumps(live.get("scorers", {})),
                "minute": live["minute"],
                "elapsed_mins": live.get("elapsed_mins", 0),
                "elapsed_secs": live.get("elapsed_secs", 0),
                "kick_off_utc": live.get("kick_off_utc", ""),
                "clock_is_ticking": live.get("clock_is_ticking", False),
                "source_match_id": live.get("source_match_id"),
            }
            updated = pd.concat([updated, pd.DataFrame([new_row])], ignore_index=True)
            changed = True
            continue

        idx = updated[mask].index[0]
        existing = updated.at[idx, "status"]

        if existing in ("scheduled", "live"):
            for key in ["minute", "scorers", "elapsed_mins", "elapsed_secs", "kick_off_utc", "clock_is_ticking"]:
                if key not in updated.columns:
                    updated[key] = None
            
            # Check if values actually changed to avoid redundant saves
            if (
                updated.at[idx, "status"] != "live" or
                updated.at[idx, "home_score"] != live["home_score"] or
                updated.at[idx, "away_score"] != live["away_score"]
            ):
                updated.at[idx, "status"] = "live"
                if live.get("round"):
                    updated.at[idx, "round"] = live["round"]
                if live.get("group") is not None:
                    updated.at[idx, "group"] = live.get("group")
                updated.at[idx, "home_score"] = live["home_score"]
                updated.at[idx, "away_score"] = live["away_score"]
                updated.at[idx, "minute"] = live["minute"]
                updated.at[idx, "elapsed_mins"] = live.get("elapsed_mins", 0)
                updated.at[idx, "elapsed_secs"] = live.get("elapsed_secs", 0)
                updated.at[idx, "kick_off_utc"] = live.get("kick_off_utc", "")
                updated.at[idx, "clock_is_ticking"] = live.get("clock_is_ticking", False)
                updated.at[idx, "scorers"] = json.dumps(live["scorers"])
                changed = True
            else:
                # Update ticking minute/elapsed time in-memory, but DO NOT trigger changed=True
                if live.get("round"):
                    updated.at[idx, "round"] = live["round"]
                if live.get("group") is not None:
                    updated.at[idx, "group"] = live.get("group")
                updated.at[idx, "minute"] = live["minute"]
                updated.at[idx, "elapsed_mins"] = live.get("elapsed_mins", 0)
                updated.at[idx, "elapsed_secs"] = live.get("elapsed_secs", 0)
                updated.at[idx, "kick_off_utc"] = live.get("kick_off_utc", "")
                updated.at[idx, "clock_is_ticking"] = live.get("clock_is_ticking", False)
                updated.at[idx, "scorers"] = json.dumps(live["scorers"])

    if changed:
        from ml.data_loader import save_2026_fixtures, append_result_to_csv
        save_2026_fixtures(updated)
        
        # Only append to CSV and trigger model retraining if there are newly completed matches
        if newly_completed:
            for live in newly_completed:
                append_result_to_csv(
                    live["date"], live["home_team"], live["away_team"],
                    live["home_score"], live["away_score"]
                )
            try:
                from ml.train import retrain_active_model
                retrain_active_model()
            except Exception as e:
                print(f"[live_scores] [ERROR] Failed to auto-retrain model: {e}")
                
        # Clear Streamlit cache if running in streamlit context to refresh the UI
        try:
            import sys
            if "streamlit" in sys.modules:
                import streamlit as st
                st.cache_data.clear()
        except Exception as ce:
            print(f"[live_scores] Failed to clear streamlit cache: {ce}")

    return updated, changed
