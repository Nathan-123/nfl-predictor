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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.config import DEFAULT_REGRESSION_START_SEASON, DEFAULT_START_SEASON, PROCESSED_DIR
from nfl_predictor.ratings.adjustment import fit_adjusted_elo_pipeline, project_upcoming_season
from nfl_predictor.ratings.elo import DEFAULT_K
from nfl_predictor.simulation.playoffs import simulate_playoffs_detailed
from nfl_predictor.simulation.season import (
    fit_margin_model,
    load_season_schedule,
    load_team_division_conference,
    project_starting_ratings,
    run_simulations,
)

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
        keep_regular_season_details=True,
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

    # ---- one realistic representative simulation: record + bracket --------
    # A complement to the Monte Carlo summary above, but built from a REAL
    # simulated season (real random variance -- favorites do sometimes lose)
    # rather than a no-randomness "favorite always wins" run, which produces
    # unrealistic blowout records (16-1, 1-16) that don't match the summary's
    # own mean_wins. Picked from the n_sims already run above: the single
    # simulation whose per-team win total is closest (least squared error)
    # to the aggregate summary's median_wins -- i.e. the most "typical" of
    # the realistic seasons already drawn, not a fabricated extreme.
    target_wins = results.summary.set_index("team")["median_wins"]
    win_pivot = results.win_totals.pivot(index="sim", columns="team", values="wins")
    squared_error = ((win_pivot - target_wins) ** 2).sum(axis=1)
    representative_sim = int(squared_error.idxmin())
    rep_standings, rep_seeds_by_conference, rep_ratings = results.regular_season_details[representative_sim]

    seed_lookup = {t: i + 1 for seeds in rep_seeds_by_conference.values() for i, t in enumerate(seeds)}
    record_rows = [
        {
            "team": team,
            "wins": rec.wins,
            "losses": rec.losses,
            "ties": rec.ties,
            "division": divisions[team],
            "conference": conferences[team],
            "made_playoffs": team in seed_lookup,
            "seed": seed_lookup.get(team),
        }
        for team, rec in rep_standings.items()
    ]
    record_df = (
        pd.DataFrame(record_rows)
        .sort_values(["made_playoffs", "wins"], ascending=[False, False])
        .reset_index(drop=True)
    )
    record_df.to_csv(PROJECTED_RECORD_PATH, index=False)

    print(
        f"\n{upcoming_season} projected final record "
        f"(one representative simulation -- #{representative_sim} of {args.n_sims}, "
        "closest to the summary's median win totals):\n"
    )
    print(record_df.to_string(index=False))

    # A fresh random draw for the playoffs, seeded from that same simulation's
    # real final standings/ratings -- not a literal replay of sim #N's own
    # postseason (which wasn't retained to save memory), but the same
    # stochastic model applied to the same realistic regular season.
    rng_playoff = np.random.default_rng(None if args.seed is None else args.seed + 1)
    bracket_games, champion = simulate_playoffs_detailed(
        seeds_by_conference=rep_seeds_by_conference,
        ratings=dict(rep_ratings),
        hfa=pipeline.adjusted_summary.hfa_used,
        k=DEFAULT_K,
        margin_intercept=margin_intercept,
        margin_slope=margin_slope,
        margin_std=margin_std,
        rng=rng_playoff,
    )
    bracket_df = pd.DataFrame([asdict(g) for g in bracket_games])
    bracket_df.to_csv(PROJECTED_BRACKET_PATH, index=False)

    print(f"\n{upcoming_season} projected playoff bracket (from that same representative simulation):\n")
    print(bracket_df.to_string(index=False))
    print(f"\nProjected Super Bowl champion: {champion}")

    print(f"\nSaved: {PROJECTED_RECORD_PATH}")
    print(f"Saved: {PROJECTED_BRACKET_PATH}")


if __name__ == "__main__":
    main()
