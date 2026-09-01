#!/usr/bin/env python
"""CLI entrypoint for Stage 3 (regular season) + Stage 4 (playoffs): Monte
Carlo simulation of the upcoming NFL season through to a Super Bowl champion.

Example:
    python scripts/run_season_simulation.py --n-sims 10000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.config import DEFAULT_REGRESSION_START_SEASON, DEFAULT_START_SEASON, PROCESSED_DIR
from nfl_predictor.ratings.adjustment import fit_adjusted_elo_pipeline, project_upcoming_season
from nfl_predictor.ratings.elo import DEFAULT_K
from nfl_predictor.simulation.playoffs import simulate_playoffs_deterministic
from nfl_predictor.simulation.season import (
    fit_margin_model,
    load_season_schedule,
    load_team_division_conference,
    project_starting_ratings,
    run_simulations,
    simulate_one_season_deterministic,
)
from nfl_predictor.simulation.standings import seed_conference

WIN_TOTALS_PATH = PROCESSED_DIR / "season_simulation_win_totals.parquet"
SUMMARY_PATH = PROCESSED_DIR / "season_simulation_summary.parquet"
PROJECTED_RECORD_PATH = PROCESSED_DIR / "projected_final_record.csv"
PROJECTED_BRACKET_PATH = PROCESSED_DIR / "projected_playoff_bracket.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo simulate the upcoming NFL regular season.")
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--regression-start-season", type=int, default=DEFAULT_REGRESSION_START_SEASON)
    parser.add_argument("--n-sims", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Fitting Elo (Stage 1) + offseason adjustment (Stage 1b)...")
    pipeline = fit_adjusted_elo_pipeline(args.start_season, regression_start_season=args.regression_start_season)
    upcoming_season = pipeline.max_season + 1

    projected_adjustments = project_upcoming_season(pipeline.features, pipeline.full_model, upcoming_season)
    starting_ratings = project_starting_ratings(pipeline.adjusted_ratings, projected_adjustments)

    margin_intercept, margin_slope, margin_std = fit_margin_model(pipeline.adjusted_log)
    print(
        f"Margin model: intercept={margin_intercept:.2f}, slope={margin_slope:.4f}, "
        f"residual_std={margin_std:.2f}, hfa={pipeline.adjusted_summary.hfa_used:.1f}"
    )

    schedule = load_season_schedule(upcoming_season)
    if schedule.empty:
        raise SystemExit(f"No {upcoming_season} regular-season schedule found in the cache.")
    divisions, conferences = load_team_division_conference()

    print(f"\nRunning {args.n_sims} simulations of the {upcoming_season} season ({len(schedule)} games each)...")
    results = run_simulations(
        n_sims=args.n_sims,
        schedule=schedule,
        starting_ratings=starting_ratings,
        hfa=pipeline.adjusted_summary.hfa_used,
        k=DEFAULT_K,
        margin_intercept=margin_intercept,
        margin_slope=margin_slope,
        margin_std=margin_std,
        divisions=divisions,
        conferences=conferences,
        seed=args.seed,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results.win_totals.to_parquet(WIN_TOTALS_PATH, index=False)
    results.summary.to_parquet(SUMMARY_PATH, index=False)

    print(f"\n{upcoming_season} season simulation summary (sorted by mean projected wins):\n")
    display = results.summary.copy()
    for col in ["mean_wins", "median_wins", "wins_p10", "wins_p90"]:
        display[col] = display[col].round(1)
    pct_cols = [
        "playoff_prob",
        "division_prob",
        "one_seed_prob",
        "won_wildcard_prob",
        "conf_championship_prob",
        "super_bowl_prob",
        "champion_prob",
    ]
    for col in pct_cols:
        display[col] = (display[col] * 100).round(1)
    print(display.to_string(index=False))

    # ---- deterministic "model's best single guess" record + bracket --------
    # A complement to the Monte Carlo summary above: one single, non-random
    # run (each game goes to whoever the fitted margin model favors) gives
    # one coherent final record and one coherent bracket, instead of the
    # per-team/per-round probabilities the Monte Carlo output reports.
    det_standings, det_ratings = simulate_one_season_deterministic(
        schedule=schedule,
        starting_ratings=starting_ratings,
        hfa=pipeline.adjusted_summary.hfa_used,
        k=DEFAULT_K,
        margin_intercept=margin_intercept,
        margin_slope=margin_slope,
        divisions=divisions,
        conferences=conferences,
    )

    conference_names = sorted(set(conferences.values()))
    seeds_by_conference = {
        conf: seed_conference([t for t in det_standings if conferences[t] == conf], divisions, det_standings)
        for conf in conference_names
    }
    seed_lookup = {t: (conf, i + 1) for conf, seeds in seeds_by_conference.items() for i, t in enumerate(seeds)}

    record_rows = [
        {
            "team": team,
            "wins": rec.wins,
            "losses": rec.losses,
            "ties": rec.ties,
            "division": divisions[team],
            "conference": conferences[team],
            "made_playoffs": team in seed_lookup,
            "seed": seed_lookup.get(team, (None, None))[1],
        }
        for team, rec in det_standings.items()
    ]
    record_df = (
        pd.DataFrame(record_rows)
        .sort_values(["made_playoffs", "wins"], ascending=[False, False])
        .reset_index(drop=True)
    )
    record_df.to_csv(PROJECTED_RECORD_PATH, index=False)

    print(f"\n{upcoming_season} projected final record (single best-guess run, no randomness):\n")
    print(record_df.to_string(index=False))

    bracket_games, champion = simulate_playoffs_deterministic(
        seeds_by_conference=seeds_by_conference,
        ratings=det_ratings,
        hfa=pipeline.adjusted_summary.hfa_used,
        k=DEFAULT_K,
        margin_intercept=margin_intercept,
        margin_slope=margin_slope,
    )
    bracket_df = pd.DataFrame([asdict(g) for g in bracket_games])
    bracket_df.to_csv(PROJECTED_BRACKET_PATH, index=False)

    print(f"\n{upcoming_season} projected playoff bracket (same deterministic run):\n")
    print(bracket_df.to_string(index=False))
    print(f"\nProjected Super Bowl champion: {champion}")

    print(f"\nSaved: {PROJECTED_RECORD_PATH}")
    print(f"Saved: {PROJECTED_BRACKET_PATH}")


if __name__ == "__main__":
    main()
