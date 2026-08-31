#!/usr/bin/env python
"""CLI entrypoint for Stage 3: Monte Carlo full-season simulation.

Example:
    python scripts/run_season_simulation.py --n-sims 10000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.config import DEFAULT_START_SEASON, PROCESSED_DIR
from nfl_predictor.ratings.adjustment import fit_adjusted_elo_pipeline, project_upcoming_season
from nfl_predictor.ratings.elo import DEFAULT_K
from nfl_predictor.simulation.season import (
    fit_margin_model,
    load_season_schedule,
    load_team_division_conference,
    project_starting_ratings,
    run_simulations,
)

WIN_TOTALS_PATH = PROCESSED_DIR / "season_simulation_win_totals.parquet"
SUMMARY_PATH = PROCESSED_DIR / "season_simulation_summary.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo simulate the upcoming NFL regular season.")
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--n-sims", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Fitting Elo (Stage 1) + offseason adjustment (Stage 1b)...")
    pipeline = fit_adjusted_elo_pipeline(args.start_season)
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
    for col in ["playoff_prob", "division_prob", "one_seed_prob"]:
        display[col] = (display[col] * 100).round(1)
    print(display.to_string(index=False))


if __name__ == "__main__":
    main()
