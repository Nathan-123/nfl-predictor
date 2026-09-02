"""Tests for the Stage 2 game outcome model: rolling-EPA leakage safety,
QB-continuity-flag correctness (both on synthetic data), and an integration
test of the real walk-forward backtest."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.gamemodel import features as F
from nfl_predictor.gamemodel.features import FEATURE_COLS, build_game_features
from nfl_predictor.gamemodel.model import summarize_backtest, walk_forward_backtest
from nfl_predictor.metrics import brier_score


# ---- rolling EPA leakage safety ---------------------------------------------


def test_rolling_epa_excludes_the_games_own_plays(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "PBP_DIR", tmp_path)

    def make_pbp(season: int, rows: list[dict]) -> None:
        defaults = {"pass": 1, "rush": 0}
        pd.DataFrame([{**defaults, **r} for r in rows]).to_parquet(tmp_path / f"{season}.parquet", index=False)

    # TEAM's game 1: bad (epa -1); game 2: great (epa +5, should NOT show up
    # in game 2's own rolling feature; only game 1's -1 should).
    make_pbp(
        2021,
        [
            {"game_id": "g1", "season": 2021, "week": 1, "posteam": "TEAM", "defteam": "OPP", "epa": -1.0},
            {"game_id": "g2", "season": 2021, "week": 2, "posteam": "TEAM", "defteam": "OPP", "epa": 5.0},
        ],
    )

    rolling = F.build_rolling_epa([2021]).set_index(["game_id", "team"])
    assert np.isnan(rolling.loc[("g1", "TEAM"), "off_epa_roll"])  # no prior game at all
    assert rolling.loc[("g2", "TEAM"), "off_epa_roll"] == pytest.approx(-1.0)  # only game 1, not game 2's own +5


def test_rolling_epa_does_not_leak_future_games(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "PBP_DIR", tmp_path)
    pd.DataFrame(
        [
            {"game_id": "g1", "season": 2021, "week": 1, "posteam": "TEAM", "defteam": "OPP", "epa": 0.0, "pass": 1, "rush": 0},
            {"game_id": "g2", "season": 2021, "week": 2, "posteam": "TEAM", "defteam": "OPP", "epa": 0.0, "pass": 1, "rush": 0},
            # A week-3 blowout that must not affect week 1 or 2's features.
            {"game_id": "g3", "season": 2021, "week": 3, "posteam": "TEAM", "defteam": "OPP", "epa": 99.0, "pass": 1, "rush": 0},
        ]
    ).to_parquet(tmp_path / "2021.parquet", index=False)

    rolling = F.build_rolling_epa([2021]).set_index(["game_id", "team"])
    assert rolling.loc[("g1", "TEAM"), "off_epa_roll"] != 99.0
    assert rolling.loc[("g2", "TEAM"), "off_epa_roll"] != 99.0
    assert np.isnan(rolling.loc[("g1", "TEAM"), "off_epa_roll"]) or rolling.loc[("g1", "TEAM"), "off_epa_roll"] < 10


# ---- QB continuity -----------------------------------------------------------


def _fake_schedules(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_qb_continuity_flags_a_change_between_consecutive_games():
    schedules = _fake_schedules(
        [
            {"game_id": "g1", "season": 2021, "week": 1, "home_team": "TEAM", "home_qb_id": "qb_a", "away_team": "OPP", "away_qb_id": "z"},
            {"game_id": "g2", "season": 2021, "week": 2, "home_team": "TEAM", "home_qb_id": "qb_b", "away_team": "OPP", "away_qb_id": "z"},
            {"game_id": "g3", "season": 2021, "week": 3, "home_team": "TEAM", "home_qb_id": "qb_b", "away_team": "OPP", "away_qb_id": "z"},
        ]
    )
    flags = F.build_qb_continuity_flags(schedules).set_index(["game_id", "team"])
    assert flags.loc[("g1", "TEAM"), "qb_changed"] != flags.loc[("g1", "TEAM"), "qb_changed"]  # NaN, no prior game
    assert flags.loc[("g2", "TEAM"), "qb_changed"] == 1.0  # qb_a -> qb_b
    assert flags.loc[("g3", "TEAM"), "qb_changed"] == 0.0  # qb_b -> qb_b


# ---- integration: real cached history ---------------------------------------


def test_backtest_runs_on_real_history_and_is_competitive_with_elo():
    features = build_game_features(2021, 2100)
    assert not features.empty
    assert set(FEATURE_COLS) <= set(features.columns)

    predictions = walk_forward_backtest(features)
    assert not predictions.empty

    report = summarize_backtest(predictions)
    assert report.n_games > 0
    assert np.isfinite(report.gbm_brier)
    assert report.gbm_brier < report.baseline_brier  # beats a 50/50 coinflip
    # Small-sample reality check, not a strict pass/fail: shouldn't be wildly
    # worse than Elo+adjustment (a large gap would suggest overfitting/a bug,
    # not just noise).
    assert report.gbm_brier < report.elo_brier + 0.02
