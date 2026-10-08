#!/usr/bin/env python
"""CLI entrypoint for the weekly prediction review: once a week's games are
final, grades the frozen pregame prediction for each one (right or wrong)
and, for the wrong ones, writes a short rule-based explanation. Appends the
graded rows to data/processed/season_review.csv.

Safe to re-run any time (idempotent -- a game already in the CSV is never
re-graded), so the natural way to use this is to just run it once a week
after that week's games wrap up:

    python scripts/run_weekly_review.py --season 2026

By default it also refetches first, so this week's real scores and stats
are actually in the cache to grade against: schedules and injuries across
their full history (--fetch-start-season through --season, since both are
single whole-history files, not partitioned per season like pbp is --
narrowing their range would overwrite every other cached season), and pbp
for just --season. Pass --skip-fetch to reuse whatever's already cached
(e.g. when re-running offline, or in a test).

IMPORTANT: nothing in this pipeline should ever regenerate
game_win_probabilities.csv (run_season_simulation.py's output) once the
season has started. That file is this script's source of truth for "what
was actually predicted before Week 1" -- regenerating it mid-season would
let each week's real results quietly reshape earlier weeks' "predictions,"
which defeats the point of grading them. Re-run run_season_simulation.py
during the offseason, freely; never during the season.

Likewise, season_review.csv itself is not read by anything in the
prediction pipeline today. It's meant to become an input to next season's
offseason-adjustment fit once a full season of real data exists to design
that feature from -- see review/pipeline.py's load_completed_season_review
docstring -- but never for the season it was generated from.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.config import DEFAULT_REGRESSION_START_SEASON, DEFAULT_START_SEASON, PARTITIONED_DATASETS
from nfl_predictor.data.pipeline import run as fetch_datasets
from nfl_predictor.review.pipeline import REVIEW_PATH, append_review, build_weekly_review

# Datasets this script needs fresh every week. schedules and injuries are
# each ONE file covering every season ever fetched (not partitioned like
# pbp is) -- fetching just --season for one of those would overwrite that
# whole-history file with only --season's rows, destroying every other
# cached season. So those always get fetched for the full history range
# (--fetch-start-season), never narrowed to just --season; only pbp
# (PARTITIONED_DATASETS, one file per season) is safe to narrow.
DATASETS_NEEDED = ["schedules", "pbp", "injuries"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade last week's predictions against what actually happened.")
    parser.add_argument("--season", type=int, default=date.today().year)
    parser.add_argument(
        "--weeks", type=str, default=None, help="Comma-separated week numbers to (re-)grade. Default: every completed, ungraded week."
    )
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON, help="Passed through for playoff-game grading.")
    parser.add_argument(
        "--regression-start-season", type=int, default=DEFAULT_REGRESSION_START_SEASON, help="Passed through for playoff-game grading."
    )
    parser.add_argument(
        "--fetch-start-season",
        type=int,
        default=DEFAULT_REGRESSION_START_SEASON,
        help="Earliest season to preserve when refetching whole-history datasets (schedules, injuries). "
        "Should match (or be earlier than) whatever range was used to originally populate data/raw/.",
    )
    parser.add_argument("--skip-fetch", action="store_true", help="Reuse whatever's already cached instead of refetching first.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weeks = [int(w) for w in args.weeks.split(",")] if args.weeks else None

    if not args.skip_fetch:
        whole_history = [d for d in DATASETS_NEEDED if d not in PARTITIONED_DATASETS]
        partitioned = [d for d in DATASETS_NEEDED if d in PARTITIONED_DATASETS]
        print(f"Fetching {', '.join(whole_history)} for {args.fetch_start_season}-{args.season}...")
        fetch_datasets(args.fetch_start_season, args.season, dataset_names=whole_history)
        print(f"Fetching {', '.join(partitioned)} for {args.season}...")
        fetch_datasets(args.season, args.season, dataset_names=partitioned)

    new_rows = build_weekly_review(
        args.season, weeks=weeks, start_season=args.start_season, regression_start_season=args.regression_start_season
    )
    if new_rows.empty:
        print("\nNothing new to grade (no completed, ungraded games found).")
        return

    combined = append_review(new_rows)

    correct = int(new_rows["correct"].sum())
    print(f"\nGraded {len(new_rows)} newly-completed game(s), {correct} correct, {len(new_rows) - correct} missed:\n")
    display = new_rows.copy()
    display["predicted_win_prob"] = (display["predicted_win_prob"] * 100).round(1)
    print(display[["week", "game_type", "home_team", "away_team", "predicted_winner", "correct", "review_note"]].to_string(index=False))

    print(f"\nSaved: {REVIEW_PATH} ({len(combined)} games total for season {args.season} across all graded weeks)")


if __name__ == "__main__":
    main()
