#!/usr/bin/env python
"""CLI entrypoint for Stage 2: fits the XGBoost game outcome model and
backtests it against Elo+adjustment and the market.

Requires Stage 1b's data/processed/elo_adjusted_game_log.parquet to exist
(run scripts/run_offseason_adjustment.py first).

Example:
    python scripts/run_game_model.py --start-season 2021
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.config import DEFAULT_START_SEASON, PROCESSED_DIR
from nfl_predictor.gamemodel.features import build_game_features
from nfl_predictor.gamemodel.model import fit_feature_importance, summarize_backtest, walk_forward_backtest

PREDICTIONS_PATH = PROCESSED_DIR / "game_model_predictions.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit and backtest the Stage 2 game outcome model.")
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=2100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Building game features...")
    features = build_game_features(args.start_season, args.end_season)

    print("Running walk-forward-by-season backtest...")
    predictions = walk_forward_backtest(features)
    report = summarize_backtest(predictions)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(PREDICTIONS_PATH, index=False)

    print(f"\nTest seasons: {report.test_seasons} ({report.n_games} games)\n")
    print(f"{'':<20}{'Brier':>10}{'Log loss':>12}")
    print(f"{'GBM (Stage 2)':<20}{report.gbm_brier:>10.4f}{report.gbm_log_loss:>12.4f}")
    print(f"{'Elo + adjustment':<20}{report.elo_brier:>10.4f}{report.elo_log_loss:>12.4f}")
    print(f"{'Market':<20}{report.market_brier:>10.4f}{report.market_log_loss:>12.4f}")
    print(f"{'50/50':<20}{report.baseline_brier:>10.4f}{report.baseline_log_loss:>12.4f}")

    print(f"\n{'':<20}{'Margin MAE':>12}")
    print(f"{'GBM (Stage 2)':<20}{report.gbm_margin_mae:>12.3f}")
    print(f"{'Elo-implied':<20}{report.elo_margin_mae:>12.3f}")
    print(f"{'Market (spread)':<20}{report.market_margin_mae:>12.3f}")

    print("\nFeature importances (classifier, fit on all data):")
    print(fit_feature_importance(features).to_string())


if __name__ == "__main__":
    main()
