"""Tests for the combined prediction-report derivation (scripts/
build_prediction_report.py's logic, in simulation/reporting.py) -- all on
synthetic data shaped like representative_season_games.csv /
projected_playoff_bracket.csv / projected_final_record.csv."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.simulation import reporting


# ---- weekly results ------------------------------------------------------------


def _fake_games(rows: list[dict]) -> pd.DataFrame:
    defaults = {"game_id": "g", "margin": 3.0, "upset": False}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_weekly_results_reports_winners_own_pregame_probability_home_won():
    games = _fake_games(
        [{"week": 1, "home_team": "A", "away_team": "B", "home_win_prob": 0.7, "predicted_winner": "A", "winner": "A"}]
    )
    weekly = reporting.build_weekly_results(games)
    assert weekly.loc[0, "winner"] == "A"
    assert weekly.loc[0, "winner_pregame_win_prob_pct"] == 70.0


def test_weekly_results_flips_probability_when_the_away_team_actually_won():
    # A was favored (70%) but B actually won this specific representative season -- an upset.
    games = _fake_games(
        [{"week": 1, "home_team": "A", "away_team": "B", "home_win_prob": 0.7, "predicted_winner": "A", "winner": "B", "upset": True}]
    )
    weekly = reporting.build_weekly_results(games)
    assert weekly.loc[0, "winner"] == "B"
    assert weekly.loc[0, "winner_pregame_win_prob_pct"] == 30.0  # B's own (low) pregame probability
    assert weekly.loc[0, "upset"]


def test_weekly_results_columns_and_sort_order():
    games = _fake_games(
        [
            {"week": 2, "home_team": "C", "away_team": "D", "home_win_prob": 0.6, "predicted_winner": "C", "winner": "C"},
            {"week": 1, "home_team": "A", "away_team": "B", "home_win_prob": 0.6, "predicted_winner": "A", "winner": "A"},
        ]
    )
    weekly = reporting.build_weekly_results(games)
    assert list(weekly.columns) == [
        "week",
        "away_team",
        "home_team",
        "winner",
        "margin",
        "winner_pregame_win_prob_pct",
        "upset",
    ]
    assert weekly["week"].tolist() == [1, 2]  # sorted by week


# ---- playoff results --------------------------------------------------------------


def _fake_bracket(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_playoff_results_flags_an_upset_when_the_worse_seed_wins():
    bracket = _fake_bracket(
        [{"round": "Wild Card", "conference": "AFC", "home_team": "A", "away_team": "B", "home_seed": 2.0, "away_seed": 7.0, "winner": "B"}]
    )
    playoff = reporting.build_playoff_results(bracket)
    assert playoff.loc[0, "upset"] == True  # noqa: E712 -- comparing an actual bool, not identity


def test_playoff_results_no_upset_when_better_seed_wins():
    bracket = _fake_bracket(
        [{"round": "Wild Card", "conference": "AFC", "home_team": "A", "away_team": "B", "home_seed": 2.0, "away_seed": 7.0, "winner": "A"}]
    )
    playoff = reporting.build_playoff_results(bracket)
    assert playoff.loc[0, "upset"] == False  # noqa: E712


def test_playoff_results_super_bowl_upset_is_blank_not_false():
    bracket = _fake_bracket(
        [{"round": "Super Bowl", "conference": None, "home_team": "A", "away_team": "B", "home_seed": None, "away_seed": None, "winner": "B"}]
    )
    playoff = reporting.build_playoff_results(bracket)
    assert pd.isna(playoff.loc[0, "upset"])


def test_playoff_results_sorted_by_round_order_not_alphabetically():
    bracket = _fake_bracket(
        [
            {"round": "Conference Championship", "conference": "AFC", "home_team": "A", "away_team": "B", "home_seed": 1.0, "away_seed": 2.0, "winner": "A"},
            {"round": "Wild Card", "conference": "AFC", "home_team": "C", "away_team": "D", "home_seed": 2.0, "away_seed": 7.0, "winner": "C"},
        ]
    )
    playoff = reporting.build_playoff_results(bracket)
    assert playoff["round"].tolist() == ["Wild Card", "Conference Championship"]


# ---- final record --------------------------------------------------------------


def test_final_record_sorts_playoff_teams_first_then_by_wins():
    record = pd.DataFrame(
        [
            {"team": "A", "wins": 8, "losses": 9, "ties": 0, "division": "D", "conference": "C", "made_playoffs": False, "seed": None},
            {"team": "B", "wins": 10, "losses": 7, "ties": 0, "division": "D", "conference": "C", "made_playoffs": True, "seed": 5},
            {"team": "C", "wins": 12, "losses": 5, "ties": 0, "division": "D", "conference": "C", "made_playoffs": True, "seed": 1},
        ]
    )
    ordered = reporting.build_final_record(record)
    assert ordered["team"].tolist() == ["C", "B", "A"]  # playoff teams first (by wins), then non-playoff teams
