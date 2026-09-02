#!/usr/bin/env python
"""CLI entrypoint for Stage 1b: fits the offseason adjustment layer on top
of the Stage 1 Elo engine and backtests it.

Example:
    python scripts/run_offseason_adjustment.py --start-season 2021
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.config import DEFAULT_REGRESSION_START_SEASON, DEFAULT_START_SEASON, PROCESSED_DIR
from nfl_predictor.ratings.adjustment import fit_adjusted_elo_pipeline, project_upcoming_season

ADJUSTED_GAME_LOG_PATH = PROCESSED_DIR / "elo_adjusted_game_log.parquet"
ADJUSTED_CURRENT_RATINGS_PATH = PROCESSED_DIR / "elo_adjusted_current_ratings.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and backtest the offseason Elo adjustment layer.")
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument(
        "--regression-start-season",
        type=int,
        default=DEFAULT_REGRESSION_START_SEASON,
        help="Fit the adjustment model on transitions from this season onward (wider than --start-season) "
        "for more independent training data; the production Elo engine/backtest still use --start-season.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Fitting Elo (Stage 1) + offseason adjustment (Stage 1b)...")
    pipeline = fit_adjusted_elo_pipeline(args.start_season, regression_start_season=args.regression_start_season)

    print(f"\nFitted on {pipeline.full_model.n_rows} team-season transitions ({sorted(pipeline.loso_df['season'].unique())}):")
    for name, coef in pipeline.full_model.coefficients.items():
        print(f"  {name:<28}{coef:+.3f}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pipeline.adjusted_log.to_parquet(ADJUSTED_GAME_LOG_PATH, index=False)
    pipeline.adjusted_ratings.to_parquet(ADJUSTED_CURRENT_RATINGS_PATH, index=False)

    b, a = pipeline.baseline_summary, pipeline.adjusted_summary
    print(f"\n{'':<20}{'Brier':>10}{'Log loss':>12}")
    print(f"{'Elo (Stage 1)':<20}{b.elo_brier:>10.4f}{b.elo_log_loss:>12.4f}")
    print(f"{'Elo + adjustment':<20}{a.elo_brier:>10.4f}{a.elo_log_loss:>12.4f}")
    print(f"{'Market':<20}{b.market_brier:>10.4f}{b.market_log_loss:>12.4f}")
    delta = b.elo_brier - a.elo_brier
    verdict = "improved" if delta > 0.0005 else "about the same" if abs(delta) <= 0.0005 else "got worse"
    print(f"\n(Small sample: {pipeline.full_model.n_rows} rows across {pipeline.loso_df['season'].nunique()} season transitions.")
    print(f" Brier {verdict} by {delta:+.4f}. Take the direction as informative, not conclusive.)")

    upcoming_season = pipeline.max_season + 1
    projected = project_upcoming_season(pipeline.features, pipeline.full_model, upcoming_season)
    if projected:
        preview = pd.DataFrame({"team": list(projected.keys()), "projected_adjustment": list(projected.values())})
        print(f"\nProjected {upcoming_season} preseason adjustment (Elo points, informational only --")
        print(" qb_value_delta uses a presumptive starter, by prior-season dropback volume, for")
        print(" any team without a real confirmed Week-1 starter yet):")
        print(preview.sort_values("projected_adjustment", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
