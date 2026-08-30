"""Tests for the Stage 1b offseason adjustment layer: coaching-change
detection, the draft pick-value curve, QB value-delta computation (on
synthetic data), and an integration test against real cached history."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.ratings import adjustment, offseason_features
from nfl_predictor.ratings.pipeline import run as run_elo


# ---- coaching changes -------------------------------------------------------


def _fake_schedules(rows: list[dict]) -> pd.DataFrame:
    defaults = {"game_type": "REG", "week": 1}
    return pd.DataFrame([{**defaults, **r} for r in rows])


def test_coaching_change_detected_when_coach_differs():
    schedules = _fake_schedules(
        [
            {"season": 2021, "home_team": "AAA", "away_team": "BBB", "home_coach": "Alice", "away_coach": "Bob"},
            {"season": 2022, "home_team": "AAA", "away_team": "BBB", "home_coach": "Carol", "away_coach": "Bob"},
        ]
    )
    changes = offseason_features.build_coaching_changes(schedules).set_index(["season", "team"])
    assert changes.loc[(2022, "AAA"), "coaching_change"] == 1.0
    assert changes.loc[(2022, "BBB"), "coaching_change"] == 0.0


def test_coaching_change_nan_with_no_prior_season():
    schedules = _fake_schedules(
        [{"season": 2021, "home_team": "AAA", "away_team": "BBB", "home_coach": "Alice", "away_coach": "Bob"}]
    )
    changes = offseason_features.build_coaching_changes(schedules).set_index(["season", "team"])
    assert changes.loc[(2021, "AAA"), "coaching_change"] != changes.loc[(2021, "AAA"), "coaching_change"]  # NaN


# ---- draft pick value curve --------------------------------------------------


def test_draft_value_curve_decreases_with_pick_number():
    rng = np.random.default_rng(0)
    picks = np.arange(1, 225)
    # A plausible decaying value with a bit of noise, all from "mature" (2018) classes.
    w_av = 40 / np.log(picks + 2) + rng.normal(0, 1, size=len(picks))
    draft_picks = pd.DataFrame({"season": 2018, "pick": picks, "w_av": w_av})

    value_curve = offseason_features.fit_draft_value_curve(draft_picks, as_of_season=2025)
    assert value_curve(1) > value_curve(32) > value_curve(100) > value_curve(220)


def test_draft_value_curve_falls_back_to_zero_with_no_mature_classes():
    draft_picks = pd.DataFrame({"season": [2024], "pick": [1], "w_av": [50.0]})
    value_curve = offseason_features.fit_draft_value_curve(draft_picks, as_of_season=2025)
    assert value_curve(1) == 0.0


# ---- QB value delta (synthetic pbp) -----------------------------------------


def test_qb_value_delta_reflects_prior_season_performance(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "PBP_DIR", tmp_path)

    def make_pbp(season: int, rows: list[dict]) -> None:
        pd.DataFrame(rows).to_parquet(tmp_path / f"{season}.parquet", index=False)

    # 2021: QB "old" (200 dropbacks, weak) starts for TEAM; QB "new" (200 dropbacks, strong) starts elsewhere.
    make_pbp(
        2021,
        [{"passer_id": "old", "qb_dropback": 1, "qb_epa": -0.1}] * 200
        + [{"passer_id": "new", "qb_dropback": 1, "qb_epa": 0.3}] * 200,
    )
    make_pbp(2022, [{"passer_id": "new", "qb_dropback": 1, "qb_epa": 0.2}] * 200)

    schedules = _fake_schedules(
        [
            {"season": 2021, "home_team": "TEAM", "away_team": "OPP", "home_qb_id": "old", "away_qb_id": "zzz"},
            {"season": 2022, "home_team": "TEAM", "away_team": "OPP", "home_qb_id": "new", "away_qb_id": "zzz"},
        ]
    )
    deltas = offseason_features.build_qb_value_deltas(schedules).set_index(["season", "team"])
    assert deltas.loc[(2022, "TEAM"), "qb_value_delta"] == pytest.approx(0.3 - (-0.1))


# ---- integration: real cached history ---------------------------------------


def test_offseason_pipeline_runs_on_real_history():
    game_log, _, _ = run_elo(start_season=2021)
    max_season = int(game_log["season"].max())
    features = offseason_features.build_offseason_features(2021, max_season + 1)
    assert not features.empty

    full_model, loso_df = adjustment.fit_with_loso_cv(features, game_log)

    assert full_model.n_rows > 0
    assert all(np.isfinite(v) for v in full_model.coefficients.values())
    assert not loso_df.empty
    assert loso_df["loso_predicted_adjustment"].notna().any()

    preds = adjustment.predict(full_model, loso_df)
    assert np.isfinite(preds).all()
