#!/usr/bin/env python
"""CLI entrypoint for the Elo team rating engine.

Examples:
    python scripts/run_elo.py --start-season 2021
    python scripts/run_elo.py --start-season 2021 --k 25 --hfa 55
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.config import DEFAULT_START_SEASON
from nfl_predictor.ratings.elo import DEFAULT_K
from nfl_predictor.ratings.pipeline import save
from nfl_predictor.ratings.pipeline import run as run_elo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Elo rating engine over cached schedules and backtest it.")
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=None)
    parser.add_argument("--k", type=float, default=DEFAULT_K, help="Elo K-factor (rating update speed).")
    parser.add_argument(
        "--hfa", type=float, default=None, help="Home-field-advantage Elo points. Default: estimate from data."
    )
    parser.add_argument(
        "--regress", type=float, default=0.4, help="Fraction of each team's rating pulled back to the mean each offseason."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    game_log, current_ratings, summary = run_elo(
        start_season=args.start_season,
        end_season=args.end_season,
        k=args.k,
        hfa=args.hfa,
        regress_frac=args.regress,
    )
    save(game_log, current_ratings)

    print(f"\nGames processed: {summary.n_games}  (seasons {args.start_season}-{args.end_season or 'latest'})")
    print(f"Home-field advantage used: {summary.hfa_used:.1f} Elo points")
    print()
    print(f"{'':<12}{'Brier':>10}{'Log loss':>12}")
    print(f"{'Elo':<12}{summary.elo_brier:>10.4f}{summary.elo_log_loss:>12.4f}")
    print(f"{'50/50':<12}{summary.baseline_brier:>10.4f}{summary.baseline_log_loss:>12.4f}")
    if summary.market_brier is not None:
        print(f"{'Market':<12}{summary.market_brier:>10.4f}{summary.market_log_loss:>12.4f}  ({summary.market_coverage} games w/ odds)")
    else:
        print("Market:      no moneyline data available in this range")

    print("\nCurrent top 10 teams by Elo rating:")
    print(current_ratings.head(10).to_string(index=False))
    print("\nCurrent bottom 5 teams by Elo rating:")
    print(current_ratings.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
