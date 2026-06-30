import unittest
from unittest.mock import patch
import pandas as pd
import json
import os
from pathlib import Path

from ml.prediction_log import sync_results_from_fixtures, LOG_PATH

class PredictionLogTests(unittest.TestCase):
    def setUp(self):
        # Back up existing prediction log if it exists
        self.backup_path = Path(str(LOG_PATH) + ".bak")
        if LOG_PATH.exists():
            LOG_PATH.rename(self.backup_path)
            
    def tearDown(self):
        # Restore backed up prediction log
        if LOG_PATH.exists():
            LOG_PATH.unlink()
        if self.backup_path.exists():
            self.backup_path.rename(LOG_PATH)

    @patch("ml.predict.predict_match")
    @patch("ml.train.retrain_active_model")
    @patch("ml.data_loader.append_result_to_csv")
    def test_sync_resolves_penalties_and_corrects_teams(self, append_csv_mock, retrain_mock, predict_mock):
        # Mock predict_match to return a dummy prediction
        predict_mock.return_value = {
            "predicted_outcome": "away_win",
            "confidence": 0.65,
            "home_win_prob": 0.35,
            "draw_prob": 0.0,
            "away_win_prob": 0.65,
        }

        # Initialize mock prediction log
        dummy_log = {
            "3": {
                "match_id": 3,
                "home_team": "Germany",
                "away_team": "Senegal", # Wrong team
                "round": "Round of 32",
                "predicted_outcome": "home_win",
                "confidence": 0.8,
                "home_win_prob": 0.8,
                "draw_prob": 0.0,
                "away_win_prob": 0.2,
                "actual_outcome": None,
                "is_correct": None,
            },
            "4": {
                "match_id": 4,
                "home_team": "Netherlands",
                "away_team": "Morocco",
                "round": "Round of 32",
                "predicted_outcome": "home_win",
                "confidence": 0.7,
                "home_win_prob": 0.7,
                "draw_prob": 0.0,
                "away_win_prob": 0.3,
                "actual_outcome": "draw", # Incorrectly recorded as draw previously
                "is_correct": False,
            }
        }
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(dummy_log, f, indent=2)

        # Mock completed fixtures
        fixtures = pd.DataFrame([
            {
                "match_id": 3,
                "round": "Round of 32",
                "home_team": "Germany",
                "away_team": "Paraguay", # Correct team
                "status": "completed",
                "home_score": 1,
                "away_score": 1,
                "home_penalty_score": 3,
                "away_penalty_score": 4, # Away team (Paraguay) wins
                "date": "2026-06-29"
            },
            {
                "match_id": 4,
                "round": "Round of 32",
                "home_team": "Netherlands",
                "away_team": "Morocco",
                "status": "completed",
                "home_score": 1,
                "away_score": 1,
                "home_penalty_score": 2,
                "away_penalty_score": 3, # Away team (Morocco) wins
                "date": "2026-06-29"
            }
        ])

        # Run sync
        count = sync_results_from_fixtures(fixtures)
        self.assertEqual(count, 2)

        # Read updated log
        with open(LOG_PATH, encoding="utf-8") as f:
            updated = json.load(f)

        # Match 3 checks: should correct Senegal to Paraguay, re-predict (which returns away_win),
        # set actual to away_win, and mark as correct.
        m3 = updated["3"]
        self.assertEqual(m3["away_team"], "Paraguay")
        self.assertEqual(m3["predicted_outcome"], "away_win")
        self.assertEqual(m3["actual_outcome"], "away_win")
        self.assertTrue(m3["is_correct"])

        # Match 4 checks: should correct outcome from draw to away_win, and since predicted
        # outcome was home_win, is_correct should be False.
        m4 = updated["4"]
        self.assertEqual(m4["actual_outcome"], "away_win")
        self.assertFalse(m4["is_correct"])

if __name__ == "__main__":
    unittest.main()
