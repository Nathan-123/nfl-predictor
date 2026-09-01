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
from nfl_predictor.ratings import pipeline as ratings_pipeline
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
    # qb_value_delta is now era-normalized (see the era-normalization tests below):
    # with just these 2 qualified QBs, mean=0.1, sample std=sqrt(2)*0.1, so
    # z_new - z_old = ((0.3-0.1) - (-0.1-0.1)) / (sqrt(2)*0.1) = sqrt(2).
    assert deltas.loc[(2022, "TEAM"), "qb_value_delta"] == pytest.approx(2**0.5)


# ---- presumptive starter fallback (for a season with no real Week-1 data yet) --


def test_presumptive_starter_picks_higher_prior_season_volume():
    qb_values = pd.DataFrame(
        [
            {"season": 2021, "passer_id": "backup", "epa_per_dropback": 0.0, "n_dropbacks": 20},
            {"season": 2021, "passer_id": "starter", "epa_per_dropback": 0.0, "n_dropbacks": 400},
        ]
    )
    rosters = _fake_rosters(
        [
            {"season": 2022, "team": "TEAM", "player_id": "backup", "position": "QB"},
            {"season": 2022, "team": "TEAM", "player_id": "starter", "position": "QB"},
        ]
    )
    presumptive = offseason_features.build_presumptive_starters(rosters, qb_values).set_index(["season", "team"])
    assert presumptive.loc[(2022, "TEAM"), "qb_id"] == "starter"


def test_qb_value_delta_falls_back_to_presumptive_starter_for_unplayed_season(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "PBP_DIR", tmp_path)

    def make_pbp(season: int, rows: list[dict]) -> None:
        pd.DataFrame(rows).to_parquet(tmp_path / f"{season}.parquet", index=False)

    # 2021: real season, TEAM started "old". 2022 hasn't been played -- no
    # schedule row for it at all, only a roster listing two candidate QBs.
    make_pbp(
        2021,
        [{"passer_id": "old", "qb_dropback": 1, "qb_epa": -0.1}] * 200
        + [{"passer_id": "new", "qb_dropback": 1, "qb_epa": 0.3}] * 200
        + [{"passer_id": "third_stringer", "qb_dropback": 1, "qb_epa": 0.5}] * 5,  # too few dropbacks to matter
    )
    schedules = _fake_schedules(
        [{"season": 2021, "home_team": "TEAM", "away_team": "OPP", "home_qb_id": "old", "away_qb_id": "zzz"}]
    )
    rosters = _fake_rosters(
        [
            {"season": 2022, "team": "TEAM", "player_id": "new", "position": "QB"},
            {"season": 2022, "team": "TEAM", "player_id": "third_stringer", "position": "QB"},
        ]
    )
    deltas = offseason_features.build_qb_value_deltas(schedules, rosters).set_index(["season", "team"])
    # Same expected math as the real-Week-1-data test above: "new" (0.3) picked
    # over "third_stringer" (too few dropbacks to be qualified) as presumptive
    # starter, compared against "old" (-0.1), the real 2021 starter.
    assert deltas.loc[(2022, "TEAM"), "qb_value_delta"] == pytest.approx(2**0.5)


def test_qb_value_delta_prefers_real_data_over_presumptive_starter(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "PBP_DIR", tmp_path)

    def make_pbp(season: int, rows: list[dict]) -> None:
        pd.DataFrame(rows).to_parquet(tmp_path / f"{season}.parquet", index=False)

    make_pbp(
        2021,
        [{"passer_id": "old", "qb_dropback": 1, "qb_epa": -0.1}] * 200
        + [{"passer_id": "bench_qb", "qb_dropback": 1, "qb_epa": 0.9}] * 200,  # huge volume, but didn't start 2022
    )
    # 2022 WAS played, and "old" started again (real data) -- even though the
    # roster also lists a higher-volume "bench_qb", real data must win.
    schedules = _fake_schedules(
        [
            {"season": 2021, "home_team": "TEAM", "away_team": "OPP", "home_qb_id": "old", "away_qb_id": "zzz"},
            {"season": 2022, "home_team": "TEAM", "away_team": "OPP", "home_qb_id": "old", "away_qb_id": "zzz"},
        ]
    )
    rosters = _fake_rosters(
        [
            {"season": 2022, "team": "TEAM", "player_id": "old", "position": "QB"},
            {"season": 2022, "team": "TEAM", "player_id": "bench_qb", "position": "QB"},
        ]
    )
    deltas = offseason_features.build_qb_value_deltas(schedules, rosters).set_index(["season", "team"])
    assert deltas.loc[(2022, "TEAM"), "qb_value_delta"] == pytest.approx(0.0)  # same starter, real data confirms it


# ---- era normalization (synthetic pbp, two eras with different spreads) ----


def test_qb_era_normalization_is_scale_invariant(tmp_path, monkeypatch):
    """Same relative standing within the season's spread -> same epa_z,
    even though the two seasons' raw values differ by a straight 2x rescale
    (standing in for one era's efficiency baseline/spread being wider than
    another's -- confirmed real, see module docstring)."""
    monkeypatch.setattr(offseason_features, "PBP_DIR", tmp_path)

    def make_season(season: int, values: dict[str, float]) -> None:
        rows = []
        for qb_id, epa in values.items():
            rows.extend([{"passer_id": qb_id, "qb_dropback": 1, "qb_epa": epa}] * 100)
        pd.DataFrame(rows).to_parquet(tmp_path / f"{season}.parquet", index=False)

    make_season(2008, {"a": -0.1, "b": 0.0, "c": 0.1, "d": 0.2})  # narrow spread, "old era"
    make_season(2016, {"a": -0.2, "b": 0.0, "c": 0.2, "d": 0.4})  # 2x wider spread, "new era"

    values = offseason_features.compute_qb_value_by_season([2008, 2016]).set_index(["season", "passer_id"])
    for qb_id in ["a", "b", "c", "d"]:
        assert values.loc[(2008, qb_id), "epa_z"] == pytest.approx(values.loc[(2016, qb_id), "epa_z"], abs=1e-6)


# ---- relocated-franchise Elo continuity ---------------------------------------


def test_relocated_franchise_keeps_continuous_elo_rating(tmp_path, monkeypatch):
    """A franchise that wins big under one code and then appears under its
    relocated code (e.g. real STL -> LA, 2016) should carry its rating
    forward, not silently reset to the 1500 default as a "new" team."""
    monkeypatch.setattr(ratings_pipeline, "DATA_DIR", tmp_path)

    rows = [
        {
            "game_id": f"g{week}",
            "season": 2015,
            "week": week,
            "game_type": "REG",
            "gameday": f"2015-09-{week:02d}",
            "home_team": "STL",
            "away_team": "OPP",
            "home_score": 30,
            "away_score": 10,
        }
        for week in range(1, 6)
    ]
    rows.append(
        {
            "game_id": "g_check",
            "season": 2016,
            "week": 1,
            "game_type": "REG",
            "gameday": "2016-09-08",
            "home_team": "LA",
            "away_team": "OPP2",
            "home_score": 20,
            "away_score": 20,
        }
    )
    pd.DataFrame(rows).to_parquet(tmp_path / "schedules.parquet", index=False)

    game_log, _, _ = run_elo(start_season=2015)
    la_pregame = game_log[(game_log["season"] == 2016) & (game_log["home_team"] == "LA")]["pregame_elo_home"].iloc[0]
    assert la_pregame > 1500  # STL's win streak carried over instead of resetting


# ---- skill-position value delta (synthetic pbp + rosters) -------------------


def _fake_rosters(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"week": 1, **r} for r in rows])


def _make_pbp(tmp_path, season: int, rows: list[dict]) -> None:
    defaults = {"rush": 0, "pass": 0, "rusher_id": None, "receiver_id": None}
    pd.DataFrame(
        [{**defaults, **r} for r in rows], columns=["rusher_id", "receiver_id", "epa", "rush", "pass"]
    ).to_parquet(tmp_path / f"{season}.parquet", index=False)


def test_skill_value_delta_positive_when_team_gains_a_productive_player(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "PBP_DIR", tmp_path)

    # 2021: a handful of unproductive players (to give the season a real
    # population to z-score against) plus one clear "star" WR, on RIVAL.
    _make_pbp(
        tmp_path,
        2021,
        [{"receiver_id": "avg1", "epa": 0.0, "pass": 1}]
        + [{"receiver_id": "avg2", "epa": 0.0, "pass": 1}]
        + [{"receiver_id": "avg3", "epa": 0.0, "pass": 1}]
        + [{"receiver_id": "avg4", "epa": 0.0, "pass": 1}]
        + [{"receiver_id": "star", "epa": 2.0, "pass": 1}] * 20,
    )
    _make_pbp(tmp_path, 2022, [])  # only prior-season (2021) production matters

    rosters = _fake_rosters(
        [
            {"season": 2021, "team": "OTHER1", "player_id": "avg1", "position": "WR"},
            {"season": 2021, "team": "OTHER2", "player_id": "avg2", "position": "WR"},
            {"season": 2021, "team": "OTHER3", "player_id": "avg3", "position": "WR"},
            {"season": 2021, "team": "OTHER4", "player_id": "avg4", "position": "WR"},
            {"season": 2021, "team": "RIVAL", "player_id": "star", "position": "WR"},
            # TEAM's 2021 room was empty; TEAM trades for "star" ahead of 2022.
            {"season": 2022, "team": "TEAM", "player_id": "star", "position": "WR"},
        ]
    )
    deltas = offseason_features.build_skill_value_deltas(rosters).set_index(["season", "team"])
    delta = deltas.loc[(2022, "TEAM"), "skill_value_delta"]
    assert delta > 0  # picked up a well-above-average player, own prior production was zero
    # star's 2021 z-score, computed by hand against the 5-player population [0,0,0,0,40]:
    # mean=8, sample std=sqrt(((0-8)**2*4+(40-8)**2)/4)=~17.89, z=(40-8)/17.89=~1.789.
    assert delta == pytest.approx(1.7889, abs=0.01)


def test_skill_value_delta_negative_when_team_loses_a_productive_player(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "PBP_DIR", tmp_path)

    _make_pbp(
        tmp_path,
        2021,
        [{"rusher_id": "avg1", "epa": 0.0, "rush": 1}]
        + [{"rusher_id": "avg2", "epa": 0.0, "rush": 1}]
        + [{"rusher_id": "avg3", "epa": 0.0, "rush": 1}]
        + [{"rusher_id": "avg4", "epa": 0.0, "rush": 1}]
        + [{"rusher_id": "star", "epa": 2.0, "rush": 1}] * 20,
    )
    _make_pbp(tmp_path, 2022, [])

    rosters = _fake_rosters(
        [
            {"season": 2021, "team": "OTHER1", "player_id": "avg1", "position": "RB"},
            {"season": 2021, "team": "OTHER2", "player_id": "avg2", "position": "RB"},
            {"season": 2021, "team": "OTHER3", "player_id": "avg3", "position": "RB"},
            {"season": 2021, "team": "OTHER4", "player_id": "avg4", "position": "RB"},
            {"season": 2021, "team": "TEAM", "player_id": "star", "position": "RB"},
            # 2022: TEAM's room is an unproven replacement (star left in free agency).
            {"season": 2022, "team": "TEAM", "player_id": "replacement", "position": "RB"},
        ]
    )
    deltas = offseason_features.build_skill_value_deltas(rosters).set_index(["season", "team"])
    delta = deltas.loc[(2022, "TEAM"), "skill_value_delta"]
    assert delta < 0  # lost a well-above-average player, incoming replacement has no track record


# ---- special teams value delta (synthetic pbp + rosters) --------------------


def test_special_teams_value_combines_kicking_and_punting(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "PBP_DIR", tmp_path)
    defaults = {
        "kicker_player_id": None,
        "punter_player_id": None,
        "field_goal_attempt": 0,
        "extra_point_attempt": 0,
        "punt_attempt": 0,
    }
    pd.DataFrame(
        [{**defaults, **r} for r in
         [{"kicker_player_id": "kicker", "field_goal_attempt": 1, "epa": 1.0}] * 10
         + [{"punter_player_id": "punter", "punt_attempt": 1, "epa": 0.2}] * 10]
    ).to_parquet(tmp_path / "2021.parquet", index=False)

    values = offseason_features.compute_special_teams_value_by_season([2021]).set_index("player_id")[
        "special_teams_value"
    ]
    assert values.loc["kicker"] == pytest.approx(10.0)
    assert values.loc["punter"] == pytest.approx(2.0)


def test_special_teams_value_delta_positive_when_team_gains_a_productive_kicker(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "PBP_DIR", tmp_path)
    defaults = {
        "kicker_player_id": None,
        "punter_player_id": None,
        "field_goal_attempt": 0,
        "extra_point_attempt": 0,
        "punt_attempt": 0,
    }
    rows = [{"kicker_player_id": f"avg{i}", "field_goal_attempt": 1, "epa": 0.1} for i in range(1, 5)] + [
        {"kicker_player_id": "star", "field_goal_attempt": 1, "epa": 2.0}
    ] * 10
    pd.DataFrame([{**defaults, **r} for r in rows]).to_parquet(tmp_path / "2021.parquet", index=False)
    pd.DataFrame([{**defaults, "epa": 0.0}]).to_parquet(tmp_path / "2022.parquet", index=False)

    rosters = _fake_rosters(
        [
            {"season": 2021, "team": "OTHER1", "player_id": "avg1", "position": "K"},
            {"season": 2021, "team": "OTHER2", "player_id": "avg2", "position": "K"},
            {"season": 2021, "team": "OTHER3", "player_id": "avg3", "position": "K"},
            {"season": 2021, "team": "OTHER4", "player_id": "avg4", "position": "K"},
            {"season": 2021, "team": "RIVAL", "player_id": "star", "position": "K"},
            # TEAM's 2021 room was empty; TEAM signs "star" ahead of 2022.
            {"season": 2022, "team": "TEAM", "player_id": "star", "position": "K"},
        ]
    )
    deltas = offseason_features.build_special_teams_value_deltas(rosters).set_index(["season", "team"])
    assert deltas.loc[(2022, "TEAM"), "special_teams_value_delta"] > 0


# ---- defensive value delta (synthetic PFR def stats + rosters) --------------


def _fake_pfr_def_stats(tmp_path, rows: list[dict]) -> None:
    defaults = {"int": 0.0, "prss": 0.0, "rat": 90.0, "tgt": 0.0, "comb": 0.0}
    pd.DataFrame([{**defaults, **r} for r in rows]).to_parquet(tmp_path / "pfr_def_stats.parquet", index=False)


def _fake_snap_counts(tmp_path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(tmp_path / "snap_counts.parquet", index=False)


def _fake_player_ids(tmp_path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_parquet(tmp_path / "player_ids.parquet", index=False)


def test_secondary_pfr_crosswalk_fills_gaps_without_overriding_real_data(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "DATA_DIR", tmp_path)
    _fake_player_ids(
        tmp_path,
        [
            {"gsis_id": "p_missing", "pfr_id": "recovered_from_secondary"},
            # Should be ignored -- rosters already has a real pfr_id for this player.
            {"gsis_id": "p_has_real", "pfr_id": "wrong_should_not_be_used"},
        ],
    )
    rosters = _fake_rosters(
        [
            {"season": 2022, "team": "TEAM", "player_id": "p_missing", "pfr_id": None, "position": "DE"},
            {"season": 2022, "team": "TEAM", "player_id": "p_has_real", "pfr_id": "real_pfr_id", "position": "DE"},
        ]
    )
    filled = offseason_features._with_secondary_pfr_ids(rosters).set_index("player_id")["pfr_id"]
    assert filled.loc["p_missing"] == "recovered_from_secondary"
    assert filled.loc["p_has_real"] == "real_pfr_id"  # real rosters.parquet data always wins


def test_defense_value_delta_picks_up_player_only_findable_via_secondary_crosswalk(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "DATA_DIR", tmp_path)
    _fake_pfr_def_stats(
        tmp_path,
        [{"season": 2021, "pfr_id": f"avg{i}", "prss": 3, "rat": 90.0, "tgt": 20, "comb": 40} for i in range(1, 5)]
        + [{"season": 2021, "pfr_id": "star", "prss": 25, "int": 2, "rat": 90.0, "tgt": 20, "comb": 40}],
    )
    _fake_player_ids(tmp_path, [{"gsis_id": "p_star", "pfr_id": "star"}])
    rosters = _fake_rosters(
        [
            {"season": 2021, "team": "OTHER1", "player_id": "p_avg1", "pfr_id": "avg1", "position": "CB"},
            {"season": 2021, "team": "OTHER2", "player_id": "p_avg2", "pfr_id": "avg2", "position": "CB"},
            {"season": 2021, "team": "OTHER3", "player_id": "p_avg3", "pfr_id": "avg3", "position": "CB"},
            {"season": 2021, "team": "OTHER4", "player_id": "p_avg4", "pfr_id": "avg4", "position": "CB"},
            {"season": 2021, "team": "RIVAL", "player_id": "p_star", "pfr_id": None, "position": "DE"},
            # rosters.parquet's own pfr_id is missing for "p_star" -- only the
            # secondary crosswalk (player_ids.parquet) knows their pfr_id.
            {"season": 2022, "team": "TEAM", "player_id": "p_star", "pfr_id": None, "position": "DE"},
        ]
    )
    deltas = offseason_features.build_defense_value_deltas(rosters).set_index(["season", "team"])
    assert deltas.loc[(2022, "TEAM"), "defense_value_delta"] > 0  # not silently dropped for lack of a pfr_id


def test_defensive_value_zeroed_for_thin_snap_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "DATA_DIR", tmp_path)
    _fake_pfr_def_stats(
        tmp_path,
        [
            # Huge raw stats, but from a tiny number of snaps -- shouldn't be trusted.
            {"season": 2021, "pfr_id": "small_sample_stud", "prss": 30, "rat": 90.0, "tgt": 20, "comb": 40},
            {"season": 2021, "pfr_id": "regular", "prss": 5, "rat": 90.0, "tgt": 20, "comb": 40},
        ],
    )
    _fake_snap_counts(
        tmp_path,
        [
            {"season": 2021, "pfr_player_id": "small_sample_stud", "defense_snaps": 20},
            {"season": 2021, "pfr_player_id": "regular", "defense_snaps": 500},
            # "regular"'s snaps split across two games, must be summed to a season total.
            {"season": 2021, "pfr_player_id": "regular", "defense_snaps": 200},
        ],
    )
    values = offseason_features.compute_defensive_value_by_season([2021]).set_index("pfr_id")["defensive_value"]
    assert values.loc["small_sample_stud"] == 0.0  # below MIN_DEFENSIVE_SNAPS, zeroed regardless of raw stats
    assert values.loc["regular"] != 0.0  # 700 total snaps, well above threshold, trusted


def test_defensive_value_zeroed_when_no_snap_record_at_all(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "DATA_DIR", tmp_path)
    _fake_pfr_def_stats(tmp_path, [{"season": 2021, "pfr_id": "no_snap_record", "prss": 30, "rat": 90.0, "tgt": 20, "comb": 40}])
    _fake_snap_counts(tmp_path, [{"season": 2021, "pfr_player_id": "someone_else", "defense_snaps": 500}])
    values = offseason_features.compute_defensive_value_by_season([2021]).set_index("pfr_id")["defensive_value"]
    assert values.loc["no_snap_record"] == 0.0  # missing snap data treated as 0 snaps, not "trust the stat line"


def test_defensive_value_rewards_pressure_and_penalizes_bad_coverage(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "DATA_DIR", tmp_path)
    _fake_pfr_def_stats(
        tmp_path,
        [
            {"season": 2021, "pfr_id": "avg1", "prss": 5, "rat": 90.0, "tgt": 20, "comb": 40},
            {"season": 2021, "pfr_id": "avg2", "prss": 5, "rat": 95.0, "tgt": 20, "comb": 40},
            {"season": 2021, "pfr_id": "avg3", "prss": 5, "rat": 85.0, "tgt": 20, "comb": 40},
            {"season": 2021, "pfr_id": "rusher", "prss": 30, "rat": 90.0, "tgt": 20, "comb": 40},
            {"season": 2021, "pfr_id": "bad_cover", "prss": 5, "rat": 140.0, "tgt": 20, "comb": 40},
        ],
    )
    values = offseason_features.compute_defensive_value_by_season([2021]).set_index("pfr_id")["defensive_value"]
    assert values.loc["rusher"] > values.loc["avg1"]  # elite pass rush, average coverage
    assert values.loc["bad_cover"] < values.loc["avg1"]  # average pass rush, bad coverage


def test_defensive_value_ignores_rat_below_coverage_qualifier(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "DATA_DIR", tmp_path)
    _fake_pfr_def_stats(
        tmp_path,
        [
            {"season": 2021, "pfr_id": "avg1", "prss": 5, "rat": 80.0, "tgt": 20, "comb": 40},
            {"season": 2021, "pfr_id": "avg2", "prss": 5, "rat": 90.0, "tgt": 20, "comb": 40},
            {"season": 2021, "pfr_id": "avg3", "prss": 5, "rat": 100.0, "tgt": 20, "comb": 40},
            # Qualified (tgt >= MIN_COVERAGE_TARGETS) with a terrible rat -- should be penalized.
            {"season": 2021, "pfr_id": "bad_cover_qualified", "prss": 5, "rat": 158.3, "tgt": 20, "comb": 40},
            # Same terrible rat, but tiny sample -- shouldn't be penalized for it.
            {"season": 2021, "pfr_id": "bad_cover_small_sample", "prss": 5, "rat": 158.3, "tgt": 1, "comb": 40},
        ],
    )
    values = offseason_features.compute_defensive_value_by_season([2021]).set_index("pfr_id")["defensive_value"]
    assert values.loc["bad_cover_small_sample"] > values.loc["bad_cover_qualified"]


def test_defense_value_delta_positive_when_team_gains_a_productive_defender(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "DATA_DIR", tmp_path)
    _fake_pfr_def_stats(
        tmp_path,
        [{"season": 2021, "pfr_id": f"avg{i}", "prss": 3, "rat": 90.0, "tgt": 20, "comb": 40} for i in range(1, 5)]
        + [{"season": 2021, "pfr_id": "star", "prss": 25, "int": 2, "rat": 90.0, "tgt": 20, "comb": 40}],
    )
    rosters = _fake_rosters(
        [
            {"season": 2021, "team": "OTHER1", "player_id": "p_avg1", "pfr_id": "avg1", "position": "CB"},
            {"season": 2021, "team": "OTHER2", "player_id": "p_avg2", "pfr_id": "avg2", "position": "CB"},
            {"season": 2021, "team": "OTHER3", "player_id": "p_avg3", "pfr_id": "avg3", "position": "CB"},
            {"season": 2021, "team": "OTHER4", "player_id": "p_avg4", "pfr_id": "avg4", "position": "CB"},
            {"season": 2021, "team": "RIVAL", "player_id": "p_star", "pfr_id": "star", "position": "DE"},
            # TEAM's 2021 room was empty; TEAM trades for "star" ahead of 2022.
            {"season": 2022, "team": "TEAM", "player_id": "p_star", "pfr_id": "star", "position": "DE"},
        ]
    )
    deltas = offseason_features.build_defense_value_deltas(rosters).set_index(["season", "team"])
    assert deltas.loc[(2022, "TEAM"), "defense_value_delta"] > 0


def test_defense_value_delta_negative_when_team_loses_a_productive_defender(tmp_path, monkeypatch):
    monkeypatch.setattr(offseason_features, "DATA_DIR", tmp_path)
    _fake_pfr_def_stats(
        tmp_path,
        [{"season": 2021, "pfr_id": f"avg{i}", "prss": 3, "rat": 90.0, "tgt": 20, "comb": 40} for i in range(1, 5)]
        + [{"season": 2021, "pfr_id": "star", "prss": 25, "int": 2, "rat": 90.0, "tgt": 20, "comb": 40}],
    )
    rosters = _fake_rosters(
        [
            {"season": 2021, "team": "OTHER1", "player_id": "p_avg1", "pfr_id": "avg1", "position": "CB"},
            {"season": 2021, "team": "OTHER2", "player_id": "p_avg2", "pfr_id": "avg2", "position": "CB"},
            {"season": 2021, "team": "OTHER3", "player_id": "p_avg3", "pfr_id": "avg3", "position": "CB"},
            {"season": 2021, "team": "OTHER4", "player_id": "p_avg4", "pfr_id": "avg4", "position": "CB"},
            {"season": 2021, "team": "TEAM", "player_id": "p_star", "pfr_id": "star", "position": "DE"},
            # 2022: TEAM's room is an unproven replacement (star left in free agency).
            {"season": 2022, "team": "TEAM", "player_id": "p_replacement", "pfr_id": "replacement", "position": "DE"},
        ]
    )
    deltas = offseason_features.build_defense_value_deltas(rosters).set_index(["season", "team"])
    assert deltas.loc[(2022, "TEAM"), "defense_value_delta"] < 0


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


def test_widened_regression_window_yields_substantially_more_training_rows():
    """The whole point of regression_start_season: a lot more independent
    season-transitions to fit on than the production 2021+ window alone
    (128 rows/4 transitions) gives."""
    pipeline = adjustment.fit_adjusted_elo_pipeline(2021, regression_start_season=2007)

    assert pipeline.full_model.n_rows > 300  # well above the old 128
    assert pipeline.loso_df["season"].nunique() > 8  # well above the old 4
    assert all(np.isfinite(v) for v in pipeline.full_model.coefficients.values())
    # Production backtest window is untouched by widening the regression's training data.
    assert pipeline.adjusted_log["season"].min() == 2021
