#!/usr/bin/env python
"""CLI entrypoint for the historical data pipeline.

Examples:
    python scripts/run_pipeline.py --start-season 2023 --end-season 2024
    python scripts/run_pipeline.py --start-season 2002
    python scripts/run_pipeline.py --datasets schedules,injuries,win_totals
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.config import DEFAULT_START_SEASON
from nfl_predictor.data.fetch import DATASETS
from nfl_predictor.data.pipeline import run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch and cache NFL historical data via nfl_data_py.")
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--end-season", type=int, default=date.today().year)
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help=f"Comma-separated subset of: {','.join(DATASETS.keys())}. Default: all.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = parse_args()
    dataset_names = args.datasets.split(",") if args.datasets else None

    results = run(args.start_season, args.end_season, dataset_names)

    name_w = max(len(r.name) for r in results) + 2
    print(f"\n{'dataset':<{name_w}}{'status':<10}{'rows':>10}  seasons")
    print("-" * (name_w + 40))
    for r in results:
        seasons = f"{r.seasons_requested[0]}-{r.seasons_requested[-1]}" if r.seasons_requested else "-"
        note = f" (clipped from {r.seasons_clipped_from})" if r.seasons_clipped_from else ""
        extra = f" [{r.error}]" if r.error else note
        print(f"{r.name:<{name_w}}{r.status:<10}{r.rows:>10}  {seasons}{extra}")

    failed = [r for r in results if r.status == "failed"]
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
