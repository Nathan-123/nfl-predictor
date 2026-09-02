"""Tests for Stage 4: playoff bracket simulation. Covers bracket advancement,
divisional re-seeding, no-tie resampling, and neutral-site handling, all on
synthetic data, plus an integration check (folded into
test_season_simulation.py's internal-consistency test)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.simulation import playoffs


# ---- no-tie resampling -------------------------------------------------------


def test_resamples_on_an_exact_tie():
    rng = MagicMock()
    rng.normal.side_effect = [0.0, 5.0]  # first draw ties exactly, must resample
    ratings = {"HOME": 1500.0, "AWAY": 1500.0}

    winner = playoffs._play_single_elim_game(
        "HOME", "AWAY", ratings, hfa=0.0, k=20.0, margin_intercept=0.0, margin_slope=0.04, margin_std=13.0, rng=rng
    )
    assert winner == "HOME"
    assert rng.normal.call_count == 2


# ---- neutral-site handling (Super Bowl) --------------------------------------


def test_neutral_site_drops_intercept_and_hfa(monkeypatch):
    captured = {}

    def fake_update_ratings(elo_home, elo_away, home_score, away_score, hfa, k):
        captured["hfa"] = hfa
        return elo_home, elo_away

    monkeypatch.setattr(playoffs, "update_ratings", fake_update_ratings)

    class FakeRng:
        def normal(self, loc, scale):
            captured["loc"] = loc
            return 5.0  # nonzero, avoids the resample loop

    ratings = {"A": 1500.0, "B": 1500.0}
    playoffs._play_single_elim_game(
        "A", "B", ratings, hfa=55.0, k=20.0, margin_intercept=2.17, margin_slope=0.04, margin_std=13.0,
        rng=FakeRng(), neutral=True,
    )
    assert captured["loc"] == 0.0  # intercept dropped
    assert captured["hfa"] == 0.0  # hfa dropped


def test_non_neutral_game_keeps_intercept_and_hfa():
    captured = {}

    class FakeRng:
        def normal(self, loc, scale):
            captured["loc"] = loc
            return 5.0

    ratings = {"A": 1500.0, "B": 1500.0}
    playoffs._play_single_elim_game(
        "A", "B", ratings, hfa=55.0, k=20.0, margin_intercept=2.17, margin_slope=0.04, margin_std=13.0,
        rng=FakeRng(), neutral=False,
    )
    assert captured["loc"] == pytest.approx(2.17)


# ---- divisional re-seeding ----------------------------------------------------


def test_divisional_round_reseeds_by_worst_remaining_seed(monkeypatch):
    seeds = ["S1", "S2", "S3", "S4", "S5", "S6", "S7"]
    calls = []
    # Scripted upsets: every Wild Card underdog wins (7 over 2, 6 over 3, 5 over 4).
    scripted_winners = {
        frozenset(["S2", "S7"]): "S7",
        frozenset(["S3", "S6"]): "S6",
        frozenset(["S4", "S5"]): "S5",
        # Correct re-seed: #1 plays the worst survivor (S7); S5 (better
        # remaining seed than S6) hosts, not the original bracket slots.
        frozenset(["S1", "S7"]): "S1",
        frozenset(["S5", "S6"]): "S6",
        frozenset(["S1", "S6"]): "S1",
    }

    def fake_play(home, away, *args, **kwargs):
        calls.append((home, away))
        return scripted_winners[frozenset([home, away])]

    monkeypatch.setattr(playoffs, "_play_single_elim_game", fake_play)

    champion, divisional_teams, conf_championship_teams = playoffs._simulate_conference(
        seeds, ratings={}, hfa=0.0, k=0.0, margin_intercept=0.0, margin_slope=0.0, margin_std=0.0, rng=None
    )

    assert champion == "S1"
    assert ("S1", "S7") in calls  # #1 seed hosts the worst surviving seed
    assert ("S5", "S6") in calls  # better remaining seed (S5) hosts, not original bracket slot
    assert divisional_teams == {"S1", "S7", "S6", "S5"}
    assert conf_championship_teams == {"S1", "S6"}


# ---- end-to-end bracket: a hugely stronger team should nearly always win ----


def test_much_stronger_team_wins_the_bracket():
    afc_seeds = ["A1", "A2", "A3", "A4", "A5", "A6", "GOAT"]  # GOAT enters as the worst seed
    nfc_seeds = ["N1", "N2", "N3", "N4", "N5", "N6", "N7"]
    ratings = {t: 1500.0 for t in afc_seeds + nfc_seeds}
    ratings["GOAT"] = 4000.0  # an enormous, unrealistic gap to make the outcome ~certain

    result = playoffs.simulate_playoffs(
        {"AFC": afc_seeds, "NFC": nfc_seeds},
        ratings,
        hfa=55.0,
        k=20.0,
        margin_intercept=2.17,
        margin_slope=0.04,
        margin_std=13.0,
        rng=np.random.default_rng(42),
    )
    assert result.champion == "GOAT"
