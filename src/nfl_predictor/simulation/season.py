"""Monte Carlo season simulation, driven by the Elo+adjustment engine (see
the plan doc for why not the Stage 2 GBM: a season sim is recursive --
each simulated outcome feeds the next game's pregame ratings -- which is
exactly what Elo is built for, while the GBM's best features (rolling EPA,
in-season QB continuity) only exist for real, already-played games.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_predictor.config import DATA_DIR
from nfl_predictor.ratings.elo import regress_to_mean, update_ratings
from nfl_predictor.simulation.standings import new_standings, record_game, seed_conference
from nfl_predictor.team_codes import canonicalize_teams


def fit_margin_model(elo_game_log: pd.DataFrame) -> tuple[float, float, float]:
    """OLS home_margin ~ intercept + slope * elo_diff on real history, plus
    the residual std -- the same approach as gamemodel.model's Elo-margin
    baseline, refit here on the full dataset (not per walk-forward fold)
    since this drives random scoreline sampling, not a backtest."""
    elo_diff = (elo_game_log["pregame_elo_home"] - elo_game_log["pregame_elo_away"]).to_numpy(dtype=float)
    margin = (elo_game_log["home_score"] - elo_game_log["away_score"]).to_numpy(dtype=float)
    A = np.vstack([np.ones_like(elo_diff), elo_diff]).T
    intercept, slope = np.linalg.lstsq(A, margin, rcond=None)[0]
    residual_std = float((margin - A @ [intercept, slope]).std())
    return float(intercept), float(slope), residual_std


def project_starting_ratings(
    current_ratings: pd.DataFrame, adjustments: dict[str, float], regress_frac: float = 1 / 3
) -> dict[str, float]:
    """Apply the same season-boundary treatment ratings.pipeline.run() uses
    internally (mean-reversion + fitted offseason adjustment) to get each
    team's rating at the start of the upcoming season."""
    return {
        row.team: regress_to_mean(row.elo_rating) + adjustments.get(row.team, 0.0)
        for row in current_ratings.itertuples(index=False)
    }


def load_team_division_conference() -> tuple[dict[str, str], dict[str, str]]:
    """team -> division, team -> conference, restricted to the 32 current
    team codes (schedules.parquet spans back to 2007 and, before
    canonicalizing, carries old codes for relocated franchises, e.g.
    OAK/SD/STL; team_desc.parquet carries those same old codes too)."""
    schedules = pd.read_parquet(DATA_DIR / "schedules.parquet")
    schedules = canonicalize_teams(schedules, ["home_team", "away_team"])
    current_teams = set(schedules["home_team"].unique()) | set(schedules["away_team"].unique())
    team_desc = pd.read_parquet(DATA_DIR / "team_desc.parquet")
    team_desc = team_desc[team_desc["team_abbr"].isin(current_teams)].drop_duplicates("team_abbr")
    divisions = dict(zip(team_desc["team_abbr"], team_desc["team_division"]))
    conferences = dict(zip(team_desc["team_abbr"], team_desc["team_conf"]))
    return divisions, conferences


def load_season_schedule(season: int) -> pd.DataFrame:
    schedules = pd.read_parquet(DATA_DIR / "schedules.parquet")
    season_games = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")]
    return season_games.sort_values(["week", "game_id"]).reset_index(drop=True)


def margin_to_scores(margin: int) -> tuple[int, int]:
    """Convert a signed point margin into a synthetic (home_score,
    away_score) pair -- elo.update_ratings only cares about the sign
    (who won) and magnitude (for the MOV multiplier), not the actual score
    level, so this is the simplest pair with the right difference."""
    return max(margin, 0), max(-margin, 0)


def simulate_one_season(
    schedule: pd.DataFrame,
    starting_ratings: dict[str, float],
    hfa: float,
    k: float,
    margin_intercept: float,
    margin_slope: float,
    margin_std: float,
    divisions: dict[str, str],
    conferences: dict[str, str],
    rng: np.random.Generator,
):
    ratings = dict(starting_ratings)
    standings = new_standings(list(ratings.keys()))

    for game in schedule.itertuples(index=False):
        home, away = game.home_team, game.away_team
        predicted_margin = margin_intercept + margin_slope * (ratings[home] - ratings[away])
        margin = int(round(rng.normal(predicted_margin, margin_std)))
        home_score, away_score = margin_to_scores(margin)

        record_game(standings, home, away, home_score, away_score, divisions, conferences)
        ratings[home], ratings[away] = update_ratings(ratings[home], ratings[away], home_score, away_score, hfa, k)

    return standings


@dataclass
class SimulationResults:
    win_totals: pd.DataFrame  # one row per (sim, team): wins (ties count as 0.5)
    summary: pd.DataFrame  # one row per team: aggregated probabilities


def run_simulations(
    n_sims: int,
    schedule: pd.DataFrame,
    starting_ratings: dict[str, float],
    hfa: float,
    k: float,
    margin_intercept: float,
    margin_slope: float,
    margin_std: float,
    divisions: dict[str, str],
    conferences: dict[str, str],
    seed: int | None = None,
) -> SimulationResults:
    rng = np.random.default_rng(seed)
    teams = list(starting_ratings.keys())
    conference_names = sorted(set(conferences.values()))

    win_rows = []
    playoff_counts = {t: 0 for t in teams}
    division_counts = {t: 0 for t in teams}
    one_seed_counts = {t: 0 for t in teams}

    for sim in range(n_sims):
        standings = simulate_one_season(
            schedule, starting_ratings, hfa, k, margin_intercept, margin_slope, margin_std, divisions, conferences, rng
        )
        for t in teams:
            rec = standings[t]
            win_rows.append({"sim": sim, "team": t, "wins": rec.wins + 0.5 * rec.ties})

        for conf in conference_names:
            conf_teams = [t for t in teams if conferences[t] == conf]
            seeds = seed_conference(conf_teams, divisions, standings)
            for t in seeds:
                playoff_counts[t] += 1
            for t in seeds[:4]:
                division_counts[t] += 1
            one_seed_counts[seeds[0]] += 1

    win_totals = pd.DataFrame(win_rows)
    grouped = win_totals.groupby("team")["wins"]
    summary = pd.DataFrame(
        {
            "team": grouped.mean().index,
            "mean_wins": grouped.mean().to_numpy(),
            "median_wins": grouped.median().to_numpy(),
            "wins_p10": grouped.quantile(0.10).to_numpy(),
            "wins_p90": grouped.quantile(0.90).to_numpy(),
            "playoff_prob": [playoff_counts[t] / n_sims for t in grouped.mean().index],
            "division_prob": [division_counts[t] / n_sims for t in grouped.mean().index],
            "one_seed_prob": [one_seed_counts[t] / n_sims for t in grouped.mean().index],
        }
    ).sort_values("mean_wins", ascending=False).reset_index(drop=True)

    return SimulationResults(win_totals=win_totals, summary=summary)
