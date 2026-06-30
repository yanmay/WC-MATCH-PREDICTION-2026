import unittest
from unittest.mock import patch

import pandas as pd

from ml.live_scores import sync_live_results_to_fixtures


class LiveScoreSyncTests(unittest.TestCase):
    def base_fixtures(self):
        return pd.DataFrame([
            {
                "match_id": 1,
                "round": "Round of 32",
                "date": "2026-06-29",
                "home_team": "Germany",
                "away_team": "DR Congo",
                "status": "scheduled",
                "home_score": None,
                "away_score": None,
                "group": None,
                "source_match_id": None,
                "minute": None,
                "elapsed_mins": None,
                "elapsed_secs": None,
                "clock_is_ticking": None,
                "scorers": None,
            },
            {
                "match_id": 132,
                "round": "Round of 32",
                "date": "2026-06-30",
                "home_team": "Germany",
                "away_team": "Paraguay",
                "status": "live",
                "home_score": 0,
                "away_score": 1,
                "group": "R32",
                "source_match_id": "74",
                "minute": "44'",
                "elapsed_mins": 44,
                "elapsed_secs": 58,
                "clock_is_ticking": True,
                "scorers": "{}",
            },
        ])

    def provider_r32(self):
        return [{
            "source_match_id": "74",
            "round": "Round of 32",
            "date": "2026-06-29",
            "home_team": "Germany",
            "away_team": "Paraguay",
            "status": "completed",
            "home_score": 1,
            "away_score": 1,
            "group": "R32",
            "home_team_label": "Winner Group E",
            "away_team_label": "3rd Group A/B/C/D/F",
            "minute": None,
            "elapsed_mins": 0,
            "elapsed_secs": 0,
            "clock_is_ticking": False,
            "kick_off_utc": "",
            "scorers": {"Germany": ["Kai Havertz 54'"], "Paraguay": ["Khvliv Ansisv 42'"]},
        }]

    @patch("ml.train.retrain_active_model")
    @patch("ml.live_scores.get_live_matches_data", return_value=[])
    @patch("ml.live_scores.get_completed_live_matches", return_value=[])
    @patch("ml.live_scores.get_provider_knockout_fixtures")
    @patch("ml.data_loader.save_2026_fixtures")
    @patch("ml.data_loader.append_result_to_csv")
    def test_provider_r32_overwrites_stale_slot_and_dedupes_source_id(
        self,
        append_result,
        save_fixtures,
        provider_mock,
        completed_mock,
        live_mock,
        retrain_mock,
    ):
        provider_mock.return_value = self.provider_r32()

        updated, changed = sync_live_results_to_fixtures(self.base_fixtures())

        self.assertTrue(changed)
        source_rows = updated[updated["source_match_id"].astype(str) == "74"]
        self.assertEqual(len(source_rows), 1)
        row = source_rows.iloc[0]
        self.assertEqual(row["home_team"], "Germany")
        self.assertEqual(row["away_team"], "Paraguay")
        self.assertEqual(row["status"], "completed")
        self.assertTrue(pd.isna(row["minute"]) or row["minute"] is None)
        self.assertEqual(row["elapsed_mins"], 0)
        self.assertFalse(bool(row["clock_is_ticking"]))


if __name__ == "__main__":
    unittest.main()
