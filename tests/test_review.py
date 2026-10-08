"""Tests for the weekly prediction review: the rule-based upset explanation
(pure functions, synthetic data) and the review-CSV pipeline (synthetic
schedules/pbp/injuries/frozen-predictions, monkeypatched cache dirs)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.review import explain
from nfl_predictor.review import pipeline as review_pipeline
from nfl_predictor.review.explain import UpsetContext


# ---- explain_upset: factor selection and ranking ---------------------------


def _base_ctx(**overrides) -> UpsetContext:
    defaults = dict(favored_team="AAA", underdog_team="BBB", favored_win_prob=0.75, margin=-3)
    return UpsetContext(**{**defaults, **overrides})


def test_qb_change_is_cited_when_favored_team_changed_starters():
    ctx = _base_ctx(favored_qb_changed=True)
    note = explain.explain_upset(ctx)
    assert "different quarterback" in note
    assert "AAA" in note


def test_turnover_margin_is_cited_when_favored_team_lost_the_battle():
    ctx = _base_ctx(favored_turnover_margin=-2)
    note = explain.explain_upset(ctx)
    assert "turnover" in note.lower()
    assert "-2" in note


def test_turnover_margin_not_cited_as_a_cause_when_it_favored_the_favorite():
    # AAA actually won the turnover battle, so it can't be why AAA lost --
    # falls through to the "nothing stands out" fallback instead.
    ctx = _base_ctx(favored_turnover_margin=1)
    note = explain.explain_upset(ctx)
    assert "lost the turnover battle" not in note
    assert "No single factor stands out" in note


def test_epa_collapse_not_cited_when_rolling_average_is_nan():
    # A team's Week 1 game has no prior games this season to average, so
    # its rolling EPA comes back as float NaN (not None) from the pandas
    # lookup -- bool(NaN > threshold) is False, so a naive check lets it
    # through as a spurious "well below nan" factor.
    import math

    ctx = _base_ctx(favored_off_epa_actual=-0.20, favored_off_epa_rolling=math.nan)
    note = explain.explain_upset(ctx)
    assert "nan" not in note.lower()
    assert "well below" not in note


def test_epa_collapse_is_cited_when_offense_underperformed_form():
    ctx = _base_ctx(favored_off_epa_actual=-0.20, favored_off_epa_rolling=0.05)
    note = explain.explain_upset(ctx)
    assert "-0.20" in note
    assert "0.05" in note


def test_epa_collapse_not_cited_for_a_small_deviation():
    ctx = _base_ctx(favored_off_epa_actual=0.02, favored_off_epa_rolling=0.05)
    note = explain.explain_upset(ctx)
    assert "well below" not in note
    assert "No single factor stands out" in note


def test_key_injuries_cited_when_enough_starters_out():
    ctx = _base_ctx(favored_key_injuries_out=3)
    note = explain.explain_upset(ctx)
    assert "3 key starters" in note


def test_key_injuries_not_cited_below_threshold():
    ctx = _base_ctx(favored_key_injuries_out=1)
    note = explain.explain_upset(ctx)
    assert "key starters" not in note


def test_close_game_falls_back_to_coin_flip_framing_when_no_factors_apply():
    ctx = _base_ctx(favored_win_prob=0.55, margin=-3)
    note = explain.explain_upset(ctx)
    assert "toss-up" in note


def test_unexplained_fallback_when_nothing_applies_and_it_was_not_close():
    ctx = _base_ctx(favored_win_prob=0.85, margin=-20)
    note = explain.explain_upset(ctx)
    assert "No single factor" in note


def test_qb_change_outranks_a_borderline_turnover_margin():
    # Both factors apply; QB change (severity 3.0) should be cited first.
    ctx = _base_ctx(favored_qb_changed=True, favored_turnover_margin=-1)
    note = explain.explain_upset(ctx)
    assert note.lower().startswith("aaa started a different quarterback")


def test_describe_correct_mentions_both_teams_and_probability():
    note = explain.describe_correct("AAA", "BBB", 0.812)
    assert "AAA" in note and "BBB" in note and "81%" in note


# ---- review pipeline: synthetic schedules/pbp/injuries/frozen predictions --


@pytest.fixture
def wired_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(review_pipeline, "DATA_DIR", tmp_path)
    monkeypatch.setattr(review_pipeline, "PBP_DIR", tmp_path / "pbp")
    (tmp_path / "pbp").mkdir()
    monkeypatch.setattr(review_pipeline, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(review_pipeline, "REVIEW_PATH", tmp_path / "season_review.csv")
    monkeypatch.setattr(review_pipeline, "GAME_PROBABILITIES_PATH", tmp_path / "game_win_probabilities.csv")
    return tmp_path


def _write_schedules(tmp_path, season, games):
    defaults = {"game_type": "REG", "home_qb_id": None, "away_qb_id": None}
    pd.DataFrame([{**defaults, "season": season, **g} for g in games]).to_parquet(tmp_path / "schedules.parquet", index=False)


def _write_frozen_predictions(tmp_path, rows):
    pd.DataFrame(rows).to_csv(tmp_path / "game_win_probabilities.csv", index=False)


def test_correct_pick_gets_a_correct_note(wired_pipeline):
    _write_schedules(
        wired_pipeline,
        2026,
        [{"game_id": "g1", "week": 1, "home_team": "AAA", "away_team": "BBB", "home_score": 27, "away_score": 10}],
    )
    _write_frozen_predictions(
        wired_pipeline, [{"week": 1, "game_id": "g1", "home_team": "AAA", "away_team": "BBB", "home_win_prob": 0.7}]
    )
    result = review_pipeline.build_weekly_review(2026)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["correct"]
    assert row["predicted_winner"] == "AAA"
    assert "Correctly predicted" in row["review_note"]


def test_incorrect_pick_gets_an_explanation(wired_pipeline):
    _write_schedules(
        wired_pipeline,
        2026,
        [{"game_id": "g1", "week": 1, "home_team": "AAA", "away_team": "BBB", "home_score": 10, "away_score": 27}],
    )
    _write_frozen_predictions(
        wired_pipeline, [{"week": 1, "game_id": "g1", "home_team": "AAA", "away_team": "BBB", "home_win_prob": 0.7}]
    )
    result = review_pipeline.build_weekly_review(2026)
    assert len(result) == 1
    row = result.iloc[0]
    assert not row["correct"]
    assert row["predicted_winner"] == "AAA"
    assert row["actual_winner"] == "BBB"
    assert row["review_note"]  # some explanation was generated


def test_turnovers_are_pulled_from_pbp_and_cited(wired_pipeline):
    _write_schedules(
        wired_pipeline,
        2026,
        [{"game_id": "g1", "week": 1, "home_team": "AAA", "away_team": "BBB", "home_score": 10, "away_score": 27}],
    )
    _write_frozen_predictions(
        wired_pipeline, [{"week": 1, "game_id": "g1", "home_team": "AAA", "away_team": "BBB", "home_win_prob": 0.7}]
    )
    # AAA (home, favored) threw 2 interceptions; BBB turned it over 0 times.
    pd.DataFrame(
        [
            {"game_id": "g1", "posteam": "AAA", "defteam": "BBB", "interception": 1, "fumble_lost": 0},
            {"game_id": "g1", "posteam": "AAA", "defteam": "BBB", "interception": 1, "fumble_lost": 0},
            {"game_id": "g1", "posteam": "BBB", "defteam": "AAA", "interception": 0, "fumble_lost": 0},
        ]
    ).to_parquet(wired_pipeline / "pbp" / "2026.parquet", index=False)

    result = review_pipeline.build_weekly_review(2026)
    note = result.iloc[0]["review_note"]
    assert "turnover margin: -2" in note.lower()


def test_already_reviewed_games_are_not_regraded(wired_pipeline):
    _write_schedules(
        wired_pipeline,
        2026,
        [{"game_id": "g1", "week": 1, "home_team": "AAA", "away_team": "BBB", "home_score": 27, "away_score": 10}],
    )
    _write_frozen_predictions(
        wired_pipeline, [{"week": 1, "game_id": "g1", "home_team": "AAA", "away_team": "BBB", "home_win_prob": 0.7}]
    )
    first = review_pipeline.build_weekly_review(2026)
    review_pipeline.append_review(first)

    second = review_pipeline.build_weekly_review(2026)
    assert second.empty


def test_append_review_is_idempotent_on_disk(wired_pipeline):
    _write_schedules(
        wired_pipeline,
        2026,
        [{"game_id": "g1", "week": 1, "home_team": "AAA", "away_team": "BBB", "home_score": 27, "away_score": 10}],
    )
    _write_frozen_predictions(
        wired_pipeline, [{"week": 1, "game_id": "g1", "home_team": "AAA", "away_team": "BBB", "home_win_prob": 0.7}]
    )
    new_rows = review_pipeline.build_weekly_review(2026)
    combined = review_pipeline.append_review(new_rows)
    assert len(combined) == 1
    # Appending the same (already-written) rows again shouldn't duplicate them.
    combined_again = review_pipeline.append_review(new_rows)
    assert len(combined_again) == 1


def test_missing_frozen_prediction_is_skipped_not_crashed(wired_pipeline):
    _write_schedules(
        wired_pipeline,
        2026,
        [{"game_id": "g1", "week": 1, "home_team": "AAA", "away_team": "BBB", "home_score": 27, "away_score": 10}],
    )
    # No game_win_probabilities.csv written at all.
    result = review_pipeline.build_weekly_review(2026)
    assert result.empty
