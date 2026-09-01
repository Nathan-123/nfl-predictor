"""Tests for Stage 3: standings/tiebreak logic (synthetic data) and an
integration test of the real Monte Carlo season simulation."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.ratings.adjustment import fit_adjusted_elo_pipeline, project_upcoming_season
from nfl_predictor.ratings.elo import DEFAULT_K
from nfl_predictor.simulation import playoffs, season, standings


# ---- margin -> scoreline -----------------------------------------------------


@pytest.mark.parametrize(
    "margin,expected",
    [(10, (10, 0)), (-7, (0, 7)), (0, (0, 0))],
)
def test_margin_to_scores(margin, expected):
    assert season.margin_to_scores(margin) == expected


def test_simulate_one_season_tie_rate_is_realistic_not_naive_rounding():
    # An even matchup (predicted_margin=0) is the worst case for spurious
    # ties -- naively rounding the continuous margin draw puts ~2.9% of its
    # mass at exactly 0 (confirmed empirically), ~10x the real NFL rate
    # (~0.28% of 2021+ games). The redraw-once fix in simulate_one_season
    # should land close to that real rate, not the naive one.
    #
    # One single game per trial (not a repeated A-vs-B schedule): ratings
    # update after every game, so a persistent single matchup runs away from
    # the even starting point after a handful of games (whoever leads starts
    # winning more), which would understate ties for the wrong reason. Each
    # trial instead starts fresh at a true 1500-1500 coin flip.
    schedule = pd.DataFrame([{"home_team": "A", "away_team": "B", "week": 1, "game_id": "g1"}])
    rng = np.random.default_rng(0)
    n_trials = 3000
    ties = 0
    for _ in range(n_trials):
        standings_out, _ = season.simulate_one_season(
            schedule=schedule,
            starting_ratings={"A": 1500.0, "B": 1500.0},
            hfa=0.0,
            k=20.0,
            margin_intercept=0.0,
            margin_slope=0.04,
            margin_std=13.2,
            divisions={"A": "DIV", "B": "DIV"},
            conferences={"A": "CONF", "B": "CONF"},
            rng=rng,
        )
        ties += standings_out["A"].ties
    tie_rate = ties / n_trials
    assert tie_rate < 0.01  # well below the naive ~2.9%, in the real ~0.28% ballpark
    assert ties > 0  # but not literally impossible either


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


# ---- deterministic ("model's best single guess") record + bracket -----------


def test_simulate_one_season_deterministic_always_picks_the_favorite():
    schedule = pd.DataFrame(
        [
            {"home_team": "STRONG", "away_team": "WEAK", "week": w, "game_id": f"g{w}"}
            for w in range(1, 6)
        ]
    )
    starting_ratings = {"STRONG": 1700.0, "WEAK": 1300.0}
    standings_out, _ = season.simulate_one_season_deterministic(
        schedule=schedule,
        starting_ratings=starting_ratings,
        hfa=0.0,
        k=20.0,
        margin_intercept=0.0,
        margin_slope=0.05,
        divisions={"STRONG": "DIV", "WEAK": "DIV"},
        conferences={"STRONG": "CONF", "WEAK": "CONF"},
    )
    assert standings_out["STRONG"].wins == 5
    assert standings_out["WEAK"].losses == 5
    assert standings_out["STRONG"].wins + standings_out["STRONG"].losses + standings_out["STRONG"].ties == 5


def test_simulate_one_season_deterministic_is_reproducible():
    # No RNG at all -- two runs of identical inputs must match exactly.
    schedule = pd.DataFrame(
        [{"home_team": "A", "away_team": "B", "week": 1, "game_id": "g1"}]
    )
    kwargs = dict(
        schedule=schedule,
        starting_ratings={"A": 1550.0, "B": 1480.0},
        hfa=25.0,
        k=20.0,
        margin_intercept=1.0,
        margin_slope=0.04,
        divisions={"A": "DIV", "B": "DIV"},
        conferences={"A": "CONF", "B": "CONF"},
    )
    st1, ratings1 = season.simulate_one_season_deterministic(**kwargs)
    st2, ratings2 = season.simulate_one_season_deterministic(**kwargs)
    assert st1["A"].wins == st2["A"].wins
    assert ratings1 == ratings2


def test_deterministic_conference_favorite_wins_every_round():
    # 7 teams, strictly decreasing ratings so seed order == rating order.
    seeds = [f"T{i}" for i in range(1, 8)]
    ratings = {t: 1700.0 - 20.0 * i for i, t in enumerate(seeds)}
    games: list = []
    champion = playoffs._simulate_conference_deterministic(
        "CONF", seeds, ratings, hfa=0.0, k=20.0, margin_intercept=0.0, margin_slope=0.05, games=games
    )
    # 3 wild-card + 2 divisional + 1 conference championship = 6 games.
    assert len(games) == 6
    assert champion == "T1"  # the #1 seed, biggest favorite in every matchup it plays


def test_deterministic_bracket_produces_a_super_bowl_between_two_conferences():
    afc = [f"AFC{i}" for i in range(1, 8)]
    nfc = [f"NFC{i}" for i in range(1, 8)]
    ratings = {t: 1600.0 for t in afc + nfc}
    ratings["AFC1"] = 1900.0  # heavy favorite, should win it all
    games, champion = playoffs.simulate_playoffs_deterministic(
        seeds_by_conference={"AFC": afc, "NFC": nfc},
        ratings=ratings,
        hfa=0.0,
        k=20.0,
        margin_intercept=0.0,
        margin_slope=0.05,
    )
    assert len(games) == 13  # 6 per conference + 1 Super Bowl
    assert sum(g.round == "Super Bowl" for g in games) == 1
    assert champion == "AFC1"


# ---- representative simulation (real randomness, one chosen sim) -------------


def _toy_league_of_14():
    # run_simulations always runs a real playoff bracket, which needs exactly
    # 7 seeded teams per conference: seed_conference takes the top team from
    # EACH division as a division winner, then the best 3 non-winners as
    # wildcards -- so 4 divisions per conference (sized 2,2,2,1) sum to
    # 4 winners + 3 wildcards = 7, matching the real NFL's structure.
    teams = [f"T{i}" for i in range(1, 15)]
    div_sizes = [2, 2, 2, 1]
    divisions = {}
    for conf_idx in range(2):
        conf_teams = teams[conf_idx * 7 : conf_idx * 7 + 7]
        pos = 0
        for div_idx, size in enumerate(div_sizes):
            for t in conf_teams[pos : pos + size]:
                divisions[t] = f"DIV_{conf_idx}_{div_idx}"
            pos += size
    conferences = {t: ("CONF_A" if i < 7 else "CONF_B") for i, t in enumerate(teams)}
    starting_ratings = {t: 1500.0 + 5.0 * i for i, t in enumerate(teams)}
    # Round-robin-ish: just enough games (3 weeks) for every team to have a
    # nonzero, plausible record without needing a real 17-game schedule.
    schedule_rows = []
    game_id = 0
    for week in range(1, 4):
        shifted = teams[week:] + teams[:week]
        for i in range(0, 14, 2):
            game_id += 1
            schedule_rows.append(
                {"home_team": shifted[i], "away_team": shifted[i + 1], "week": week, "game_id": f"g{game_id}"}
            )
    return pd.DataFrame(schedule_rows), starting_ratings, divisions, conferences


def test_run_simulations_omits_regular_season_details_by_default():
    schedule, starting_ratings, divisions, conferences = _toy_league_of_14()
    results = season.run_simulations(
        n_sims=5,
        schedule=schedule,
        starting_ratings=starting_ratings,
        hfa=25.0,
        k=20.0,
        margin_intercept=1.0,
        margin_slope=0.04,
        margin_std=10.0,
        divisions=divisions,
        conferences=conferences,
        seed=1,
    )
    assert results.regular_season_details is None


def test_run_simulations_keeps_regular_season_details_when_requested():
    schedule, starting_ratings, divisions, conferences = _toy_league_of_14()
    results = season.run_simulations(
        n_sims=5,
        schedule=schedule,
        starting_ratings=starting_ratings,
        hfa=25.0,
        k=20.0,
        margin_intercept=1.0,
        margin_slope=0.04,
        margin_std=10.0,
        divisions=divisions,
        conferences=conferences,
        seed=1,
        keep_regular_season_details=True,
    )
    assert len(results.regular_season_details) == 5
    rec_standings, seeds_by_conference, final_ratings = results.regular_season_details[0]
    assert rec_standings["T1"].wins + rec_standings["T1"].losses + rec_standings["T1"].ties == 3
    assert len(seeds_by_conference["CONF_A"]) == 7
    assert set(final_ratings) == set(starting_ratings)
    # A win vector recomputed from the stored standings must match win_totals'
    # own row for the same sim -- the two shouldn't be able to silently drift.
    row = results.win_totals[(results.win_totals["sim"] == 0) & (results.win_totals["team"] == "T1")].iloc[0]
    assert rec_standings["T1"].wins + 0.5 * rec_standings["T1"].ties == row["wins"]


def test_simulate_playoffs_detailed_produces_a_full_bracket():
    afc = [f"AFC{i}" for i in range(1, 8)]
    nfc = [f"NFC{i}" for i in range(1, 8)]
    ratings = {t: 1600.0 for t in afc + nfc}
    games, champion = playoffs.simulate_playoffs_detailed(
        seeds_by_conference={"AFC": afc, "NFC": nfc},
        ratings=ratings,
        hfa=0.0,
        k=20.0,
        margin_intercept=0.0,
        margin_slope=0.05,
        margin_std=10.0,
        rng=np.random.default_rng(0),
    )
    assert len(games) == 13  # 6 per conference + 1 Super Bowl
    assert sum(g.round == "Super Bowl" for g in games) == 1
    assert champion in afc + nfc
    assert games[-1].winner == champion
    assert games[-1].home_seed is None and games[-1].away_seed is None  # Super Bowl has no "seed"


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


def test_deterministic_pipeline_produces_one_valid_record_and_bracket():
    pipeline = fit_adjusted_elo_pipeline(2021)
    upcoming_season = pipeline.max_season + 1

    projected_adjustments = project_upcoming_season(pipeline.features, pipeline.full_model, upcoming_season)
    starting_ratings = season.project_starting_ratings(pipeline.adjusted_ratings, projected_adjustments)
    margin_intercept, margin_slope, _ = season.fit_margin_model(pipeline.adjusted_log)
    schedule = season.load_season_schedule(upcoming_season)
    divisions, conferences = season.load_team_division_conference()

    det_standings, det_ratings = season.simulate_one_season_deterministic(
        schedule=schedule,
        starting_ratings=starting_ratings,
        hfa=pipeline.adjusted_summary.hfa_used,
        k=DEFAULT_K,
        margin_intercept=margin_intercept,
        margin_slope=margin_slope,
        divisions=divisions,
        conferences=conferences,
    )
    assert len(det_standings) == 32
    for rec in det_standings.values():
        assert rec.wins + rec.losses + rec.ties == 17  # every team plays a full 17-game slate

    conference_names = sorted(set(conferences.values()))
    seeds_by_conference = {
        conf: standings.seed_conference([t for t in det_standings if conferences[t] == conf], divisions, det_standings)
        for conf in conference_names
    }
    assert sum(len(s) for s in seeds_by_conference.values()) == 14  # 7 seeds x 2 conferences

    games, champion = playoffs.simulate_playoffs_deterministic(
        seeds_by_conference=seeds_by_conference,
        ratings=det_ratings,
        hfa=pipeline.adjusted_summary.hfa_used,
        k=DEFAULT_K,
        margin_intercept=margin_intercept,
        margin_slope=margin_slope,
    )
    assert len(games) == 13
    assert champion in det_standings
    assert games[-1].round == "Super Bowl"
    assert games[-1].winner == champion
