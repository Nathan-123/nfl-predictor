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

from nfl_predictor.config import DEFAULT_START_SEASON, PROCESSED_DIR
from nfl_predictor.ratings.adjustment import (
    FEATURE_COLS,
    fit_with_loso_cv,
    predict,
    season_adjustments_from_loso,
)
from nfl_predictor.ratings.offseason_features import build_offseason_features
from nfl_predictor.ratings.pipeline import run as run_elo

ADJUSTED_GAME_LOG_PATH = PROCESSED_DIR / "elo_adjusted_game_log.parquet"
ADJUSTED_CURRENT_RATINGS_PATH = PROCESSED_DIR / "elo_adjusted_current_ratings.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and backtest the offseason Elo adjustment layer.")
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Running baseline Elo (Stage 1)...")
    baseline_log, _, baseline_summary = run_elo(start_season=args.start_season)

    max_season = int(baseline_log["season"].max())
    features = build_offseason_features(args.start_season, max_season + 1)  # +1 to include the upcoming offseason

    print("Fitting offseason adjustment model (OLS + leave-one-season-out CV)...")
    full_model, loso_df = fit_with_loso_cv(features, baseline_log)

    print(f"\nFitted on {full_model.n_rows} team-season transitions ({sorted(loso_df['season'].unique())}):")
    for name, coef in full_model.coefficients.items():
        print(f"  {name:<22}{coef:+.3f}")

    season_adjustments = season_adjustments_from_loso(loso_df)

    print("\nRe-running Elo with LOSO-predicted adjustments applied...")
    adjusted_log, adjusted_ratings, adjusted_summary = run_elo(
        start_season=args.start_season, season_adjustments=season_adjustments
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    adjusted_log.to_parquet(ADJUSTED_GAME_LOG_PATH, index=False)
    adjusted_ratings.to_parquet(ADJUSTED_CURRENT_RATINGS_PATH, index=False)

    print(f"\n{'':<20}{'Brier':>10}{'Log loss':>12}")
    print(f"{'Elo (Stage 1)':<20}{baseline_summary.elo_brier:>10.4f}{baseline_summary.elo_log_loss:>12.4f}")
    print(f"{'Elo + adjustment':<20}{adjusted_summary.elo_brier:>10.4f}{adjusted_summary.elo_log_loss:>12.4f}")
    print(f"{'Market':<20}{baseline_summary.market_brier:>10.4f}{baseline_summary.market_log_loss:>12.4f}")
    delta = baseline_summary.elo_brier - adjusted_summary.elo_brier
    verdict = "improved" if delta > 0.0005 else "about the same" if abs(delta) <= 0.0005 else "got worse"
    print(f"\n(Small sample: {full_model.n_rows} rows across {loso_df['season'].nunique()} season transitions -- ")
    print(f" Brier {verdict} by {delta:+.4f}. Take the direction as informative, not conclusive.)")

    upcoming_season = max_season + 1
    preview = features[features["season"] == upcoming_season].copy()
    if not preview.empty:
        preview[FEATURE_COLS] = preview[FEATURE_COLS].fillna(0.0)
        preview["projected_adjustment"] = predict(full_model, preview)
        print(f"\nProjected {upcoming_season} preseason adjustment (Elo points, informational only --")
        print(" QB continuity can't be confirmed until Week 1 starters are known):")
        print(
            preview[["team", "coaching_change", "qb_value_delta", "draft_capital_added", "projected_adjustment"]]
            .sort_values("projected_adjustment", ascending=False)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
