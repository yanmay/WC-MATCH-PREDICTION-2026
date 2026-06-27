"""
Fetch Historical Stats — FIFA World Cup Prediction Platform
Uses user's API-Football key to fetch and update stats for 2022 World Cup matches.
Rate limits to 10 requests/minute.
"""

import urllib.request
import json
import time
import sys
from pathlib import Path

API_KEY = "6c502e6f8b71bcc41968fad0a016f22e"
DATA_DIR = Path(__file__).parent.parent / "data"
STATS_PATH = DATA_DIR / "historical_stats.json"

def fetch_api(path: str) -> dict:
    url = f"https://v3.football.api-sports.io/{path}"
    req = urllib.request.Request(url, headers={
        "x-apisports-key": API_KEY,
        "User-Agent": "Mozilla/5.0 WorldCupPredictor/1.0",
        "Accept": "application/json"
    })
    # Respect rate-limit (max 10 requests per minute)
    time.sleep(6.1)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())

def main():
    print("Fetching World Cup 2022 fixtures to retrieve API IDs...")
    try:
        data = fetch_api("fixtures?league=1&season=2022")
    except Exception as e:
        print(f"Error fetching 2022 fixtures: {e}")
        sys.exit(1)
        
    fixtures = data.get("response", [])
    if not fixtures:
        print("No fixtures returned. Please verify your API key and plan.")
        sys.exit(1)
        
    print(f"Found {len(fixtures)} fixtures for 2022 World Cup.")
    
    # Load existing stats database
    existing_stats = {}
    if STATS_PATH.exists():
        try:
            with open(STATS_PATH, encoding="utf-8") as f:
                existing_stats = json.load(f)
        except Exception:
            pass
            
    count = 0
    for idx, f in enumerate(fixtures, 1):
        fixture_id = str(f["fixture"]["id"])
        teams = f["teams"]
        home = teams["home"]["name"]
        away = teams["away"]["name"]
        
        # Check if we already have it
        if fixture_id in existing_stats and "stats" in existing_stats[fixture_id]:
            # skip fetching if already exists (avoids API quota waste)
            continue
            
        print(f"[{idx}/{len(fixtures)}] Fetching statistics for {home} vs {away} (ID: {fixture_id})...")
        try:
            stats_data = fetch_api(f"fixtures/statistics?fixture={fixture_id}")
            errors = stats_data.get("errors", {})
            if errors and isinstance(errors, dict) and "plan" in errors:
                print(f"API Plan restriction: {errors['plan']}")
                break
                
            response = stats_data.get("response", [])
            match_stats = {}
            for team_stats in response:
                team_name = team_stats["team"]["name"]
                stats_list = team_stats["statistics"]
                team_dict = {}
                for s in stats_list:
                    team_dict[s["type"]] = s["value"]
                match_stats[team_name] = team_dict
                
            if match_stats:
                existing_stats[fixture_id] = {
                    "home_team": home,
                    "away_team": away,
                    "date": f["fixture"]["date"][:10],
                    "stats": match_stats
                }
                with open(STATS_PATH, "w", encoding="utf-8") as out:
                    json.dump(existing_stats, out, indent=2)
                count += 1
        except Exception as err:
            print(f"Failed to fetch statistics for ID {fixture_id}: {err}")
            break
            
    print(f"Completed! Added/updated {count} match statistics.")

if __name__ == "__main__":
    main()
