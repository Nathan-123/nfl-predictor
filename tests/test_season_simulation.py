"""Tests for Stage 3: standings/tiebreak logic (synthetic data) and an
integration test of the real Monte Carlo season simulation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.ratings.adjustment import fit_adjusted_elo_pipeline, project_upcoming_season
from nfl_predictor.ratings.elo import DEFAULT_K
from nfl_predictor.simulation import season, standings


# ---- margin -> scoreline -----------------------------------------------------


@pytest.mark.parametrize(
    "margin,expected",
    [(10, (10, 0)), (-7, (0, 7)), (0, (0, 0))],
)
def test_margin_to_scores(margin, expected):
    assert season.margin_to_scores(margin) == expected


# ---- tiebreak logic (synthetic mini-league) ----------------------------------

DIVISIONS = {"A1": "DIV_A", "A2": "DIV_A", "B1": "DIV_B", "B2": "DIV_B"}
CONFERENCES = {"A1": "CONF", "A2": "CONF", "B1": "CONF", "B2": "CONF"}


def test_head_to_head_breaks_a_win_pct_tie():
    st = standings.new_standings(list(DIVISIONS))
    # A1 and A2 both finish 1-1, but A1 beat A2 head-to-head.
    standings.record_game(st, "A1", "A2", 24, 10, DIVISIONS, CONFERENCES)  # A1 beats A2
    standings.record_game(st, "B1", "A1", 24, 10, DIVISIONS, CONFERENCES)  # A1 loses to B1
    standings.record_game(st, "A2", "B2", 24, 10, DIVISIONS, CONFERENCES)  # A2 beats B2

    ranked = standings.rank_teams(["A1", "A2"], st)
    assert ranked == ["A1", "A2"]


def test_division_record_breaks_tie_with_no_head_to_head():
    st = standings.new_standings(list(DIVISIONS))
    # A1 and B1 never play each other; A1 has a better division record.
    standings.record_game(st, "A1", "A2", 24, 10, DIVISIONS, CONFERENCES)  # A1 beats division rival
    standings.record_game(st, "B1", "B2", 10, 24, DIVISIONS, CONFERENCES)  # B1 loses to division rival
    standings.record_game(st, "B1", "A2", 24, 10, DIVISIONS, CONFERENCES)  # B1's one win, non-division

    # Both A1 and B1 are 1-1 overall with no games against each other.
    ranked = standings.rank_teams(["A1", "B1"], st)
    assert ranked == ["A1", "B1"]


def test_seed_conference_orders_division_winners_before_wildcards():
    st = standings.new_standings(list(DIVISIONS))
    # A1 wins the most, then B1, then A2, then B2 loses out.
    standings.record_game(st, "A1", "A2", 30, 0, DIVISIONS, CONFERENCES)
    standings.record_game(st, "A1", "B1", 30, 0, DIVISIONS, CONFERENCES)
    standings.record_game(st, "B1", "B2", 30, 0, DIVISIONS, CONFERENCES)
    standings.record_game(st, "A2", "B2", 30, 0, DIVISIONS, CONFERENCES)

    seeds = standings.seed_conference(list(DIVISIONS), DIVISIONS, st)
    assert set(seeds) == set(DIVISIONS)  # everyone seeded, only 4 teams total
    assert seeds[0] == "A1"  # division A's best team, division A's winner
    # Division winners (one per division) must come before wildcards.
    division_winners = {"A1", "B1"}
    assert set(seeds[:2]) == division_winners


# ---- integration: real projected 2026 season ---------------------------------


def test_full_simulation_pipeline_is_internally_consistent():
    pipeline = fit_adjusted_elo_pipeline(2021)
    upcoming_season = pipeline.max_season + 1

    projected_adjustments = project_upcoming_season(pipeline.features, pipeline.full_model, upcoming_season)
    starting_ratings = season.project_starting_ratings(pipeline.adjusted_ratings, projected_adjustments)

    margin_intercept, margin_slope, margin_std = season.fit_margin_model(pipeline.adjusted_log)
    schedule = season.load_season_schedule(upcoming_season)
    divisions, conferences = season.load_team_division_conference()
    assert not schedule.empty
    assert len(divisions) == 32

    results = season.run_simulations(
        n_sims=200,
        schedule=schedule,
        starting_ratings=starting_ratings,
        hfa=pipeline.adjusted_summary.hfa_used,
        k=DEFAULT_K,
        margin_intercept=margin_intercept,
        margin_slope=margin_slope,
        margin_std=margin_std,
        divisions=divisions,
        conferences=conferences,
        seed=42,
    )

    summary = results.summary
    assert len(summary) == 32
    assert summary["playoff_prob"].between(0, 1).all()
    assert summary["playoff_prob"].sum() == pytest.approx(14.0)  # 7 teams x 2 conferences, every sim
    assert summary["division_prob"].sum() == pytest.approx(8.0)  # 1 per division x 8 divisions
    assert summary["one_seed_prob"].sum() == pytest.approx(2.0)  # 1 per conference
    assert (summary["division_prob"] <= summary["playoff_prob"] + 1e-9).all()
    assert (summary["mean_wins"] >= 0).all() and (summary["mean_wins"] <= 17).all()

    # Stage 4 playoff rounds: mass at each round matches that round's bracket size.
    assert summary["won_wildcard_prob"].sum() == pytest.approx(8.0)  # 4 survivors x 2 conferences
    assert summary["conf_championship_prob"].sum() == pytest.approx(4.0)  # 2 finalists x 2 conferences
    assert summary["super_bowl_prob"].sum() == pytest.approx(2.0)  # 1 champion x 2 conferences
    assert summary["champion_prob"].sum() == pytest.approx(1.0)  # exactly 1 Super Bowl winner, every sim
    assert (summary["won_wildcard_prob"] <= summary["playoff_prob"] + 1e-9).all()
    assert (summary["champion_prob"] <= summary["super_bowl_prob"] + 1e-9).all()
