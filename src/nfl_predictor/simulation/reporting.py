"""Derives the combined human-readable prediction report (scripts/
build_prediction_report.py) from run_season_simulation.py's representative-
simulation outputs: one real, randomly-drawn realistic season, so real
upsets show up naturally, and the weekly results, playoff bracket, and
final record are all mutually consistent (literally the same simulated
season), unlike picking each game's aggregate favorite independently. Pure
functions, no I/O; the script handles reading and writing.
"""

from __future__ import annotations

import pandas as pd

_ROUND_ORDER = {"Wild Card": 0, "Divisional": 1, "Conference Championship": 2, "Super Bowl": 3}


def build_weekly_results(games: pd.DataFrame) -> pd.DataFrame:
    """games: representative_season_games.csv's shape (week, game_id,
    home_team, away_team, winner, margin, home_win_prob, predicted_winner,
    upset). Returns the real result for every game, with the model's
    pregame confidence in whoever actually won (flipped from home_win_prob
    when the away team was favored), so upsets read as "predicted at 28%,
    won anyway" rather than a raw home/away number."""
    picks = games.copy()
    home_win_pct = (picks["home_win_prob"] * 100).round(1)
    away_win_pct = (100 - home_win_pct).round(1)
    picks["winner_pregame_win_prob_pct"] = home_win_pct.where(picks["winner"] == picks["home_team"], away_win_pct)
    picks["margin"] = picks["margin"].round(1)
    return picks[
        ["week", "away_team", "home_team", "winner", "margin", "winner_pregame_win_prob_pct", "upset"]
    ].sort_values(["week", "home_team"]).reset_index(drop=True)


def build_playoff_results(bracket: pd.DataFrame) -> pd.DataFrame:
    """bracket: projected_playoff_bracket.csv's shape (round, conference,
    home_team, away_team, home_seed, away_seed, winner). The same
    representative simulation's real bracket: home_team is always the
    better seed by construction, except the Super Bowl, which has no seeds
    (see simulation.playoffs). "upset" is left blank for the Super Bowl
    since there's no seed to judge it against."""
    bracket = bracket.copy()
    has_seeds = bracket["home_seed"].notna()
    upset = pd.Series(pd.NA, index=bracket.index, dtype="boolean")
    upset[has_seeds] = bracket.loc[has_seeds, "winner"] != bracket.loc[has_seeds, "home_team"]
    bracket["upset"] = upset

    bracket["_round_order"] = bracket["round"].map(_ROUND_ORDER)
    bracket = bracket.sort_values(["_round_order", "conference"])
    return bracket[
        ["conference", "round", "home_team", "home_seed", "away_team", "away_seed", "winner", "upset"]
    ].reset_index(drop=True)


def build_final_record(record: pd.DataFrame) -> pd.DataFrame:
    """record: projected_final_record.csv's shape (team, wins, losses, ties,
    division, conference, made_playoffs, seed). The same representative
    simulation's real final standings, already consistent with
    build_weekly_results' game-by-game results since it's literally the
    same season. Just reorders/sorts for the combined report."""
    return record.sort_values(["made_playoffs", "wins"], ascending=[False, False]).reset_index(drop=True)
