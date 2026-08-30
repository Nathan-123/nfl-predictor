"""Unit tests for the pure Elo math (elo.py) plus one integration test that
replays the real cached schedule history and sanity-checks the backtest.

The integration test only *reads* data/raw/schedules.parquet -- read-only
access is safe and does not need the isolation fixture in test_pipeline.py.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.ratings import elo
from nfl_predictor.ratings.pipeline import run as run_elo


# ---- expected_home_win_prob ----------------------------------------------


def test_equal_ratings_no_hfa_is_a_coinflip():
    assert elo.expected_home_win_prob(1500, 1500, hfa=0) == pytest.approx(0.5)


def test_home_field_advantage_favors_home_team():
    even = elo.expected_home_win_prob(1500, 1500, hfa=0)
    with_hfa = elo.expected_home_win_prob(1500, 1500, hfa=50)
    assert with_hfa > even


def test_expected_win_prob_monotonic_in_rating_gap():
    p_small_gap = elo.expected_home_win_prob(1550, 1500, hfa=0)
    p_big_gap = elo.expected_home_win_prob(1700, 1500, hfa=0)
    assert 0.5 < p_small_gap < p_big_gap < 1.0


def test_expected_win_prob_symmetric():
    p_home = elo.expected_home_win_prob(1600, 1500, hfa=0)
    p_away = elo.expected_home_win_prob(1500, 1600, hfa=0)
    assert p_home + p_away == pytest.approx(1.0)


# ---- mov_multiplier --------------------------------------------------------


def test_mov_multiplier_grows_with_margin():
    small = elo.mov_multiplier(margin=3, elo_diff=0)
    large = elo.mov_multiplier(margin=35, elo_diff=0)
    assert large > small


def test_mov_multiplier_dampened_for_big_favorites():
    # Same margin, but a much bigger pre-game favorite gets a smaller multiplier.
    underdog_win = elo.mov_multiplier(margin=14, elo_diff=0)
    favorite_blowout = elo.mov_multiplier(margin=14, elo_diff=400)
    assert favorite_blowout < underdog_win


# ---- update_ratings ---------------------------------------------------------


def test_update_ratings_is_zero_sum():
    new_home, new_away = elo.update_ratings(1500, 1500, home_score=24, away_score=10, hfa=0, k=20)
    assert (new_home - 1500) == pytest.approx(-(new_away - 1500))


def test_winner_rating_increases():
    new_home, new_away = elo.update_ratings(1500, 1500, home_score=24, away_score=10, hfa=0, k=20)
    assert new_home > 1500
    assert new_away < 1500


def test_upset_moves_ratings_more_than_expected_result():
    # Huge underdog (home, -300 gap) wins outright vs. a pick'em game -- the
    # upset should be a bigger rating swing than an even-money result.
    _, upset_away = elo.update_ratings(1200, 1500, home_score=20, away_score=17, hfa=0, k=20)
    _, coinflip_away = elo.update_ratings(1500, 1500, home_score=20, away_score=17, hfa=0, k=20)
    assert (1500 - upset_away) > (1500 - coinflip_away)


def test_tie_leaves_favorite_rating_lower_than_a_win_would():
    tie_home, _ = elo.update_ratings(1600, 1500, home_score=20, away_score=20, hfa=0, k=20)
    win_home, _ = elo.update_ratings(1600, 1500, home_score=21, away_score=20, hfa=0, k=20)
    assert tie_home < win_home


# ---- regress_to_mean --------------------------------------------------------


def test_regress_to_mean_pulls_toward_average():
    assert elo.regress_to_mean(1800, mean=1500, regress_frac=1 / 3) == pytest.approx(1700)
    assert elo.regress_to_mean(1200, mean=1500, regress_frac=1 / 3) == pytest.approx(1300)


def test_regress_to_mean_leaves_average_team_unchanged():
    assert elo.regress_to_mean(1500, mean=1500, regress_frac=1 / 3) == pytest.approx(1500)


# ---- hfa_from_home_win_rate -------------------------------------------------


def test_hfa_zero_at_fifty_percent_home_win_rate():
    assert elo.hfa_from_home_win_rate(0.5) == pytest.approx(0.0)


def test_hfa_positive_when_home_teams_win_more():
    assert elo.hfa_from_home_win_rate(0.58) > 0


# ---- integration: real cached schedule history ------------------------------


def test_pipeline_runs_on_real_history_and_beats_the_coinflip_baseline():
    game_log, current_ratings, summary = run_elo(start_season=2021)

    assert summary.n_games == len(game_log)
    assert summary.n_games > 1000  # multiple full 17-game seasons + playoffs
    assert summary.elo_brier < summary.baseline_brier
    assert summary.elo_log_loss < summary.baseline_log_loss

    # Ratings should have spread out from the 1500 starting point but stay
    # in a sane range -- no runaway blowup from a bug in the update math.
    assert current_ratings["elo_rating"].between(1000, 2000).all()
    assert len(current_ratings) == 32  # all NFL teams present

    assert game_log["pred_home_win_prob"].between(0, 1).all()
