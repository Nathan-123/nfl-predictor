#!/usr/bin/env python
"""Builds one combined, human-readable prediction report from run_season_
simulation.py's representative-simulation outputs: the real result of every
regular-season game (week by week, with real upsets), the real playoff
bracket, and the real final regular-season record -- all in one CSV, all
from the SAME one randomly-drawn realistic season, so every section is
mutually consistent (unlike picking each game's aggregate favorite
independently, which both looks unrealistic and can disagree with itself
from one section to the next).

Doesn't run any new simulation -- purely reformats
data/processed/representative_season_games.csv, projected_playoff_bracket.csv,
and projected_final_record.csv (run scripts/run_season_simulation.py first
if those don't exist yet, or are stale relative to the data you want
reflected).

Example:
    python scripts/build_prediction_report.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.config import PROCESSED_DIR
from nfl_predictor.simulation.reporting import build_final_record, build_playoff_results, build_weekly_results

REPRESENTATIVE_GAMES_PATH = PROCESSED_DIR / "representative_season_games.csv"
PROJECTED_BRACKET_PATH = PROCESSED_DIR / "projected_playoff_bracket.csv"
PROJECTED_RECORD_PATH = PROCESSED_DIR / "projected_final_record.csv"
REPORT_PATH = PROCESSED_DIR / "prediction_report.csv"


def _write_section(writer: csv.writer, title: str, note: str, df: pd.DataFrame) -> None:
    writer.writerow([f"=== {title} ==="])
    if note:
        writer.writerow([note])
    writer.writerow(list(df.columns))
    for row in df.itertuples(index=False):
        writer.writerow(list(row))
    writer.writerow([])


def main() -> None:
    missing = [
        p for p in (REPRESENTATIVE_GAMES_PATH, PROJECTED_BRACKET_PATH, PROJECTED_RECORD_PATH) if not p.exists()
    ]
    if missing:
        raise SystemExit(
            f"Missing {[p.name for p in missing]} -- run scripts/run_season_simulation.py first."
        )

    games = pd.read_csv(REPRESENTATIVE_GAMES_PATH)
    bracket = pd.read_csv(PROJECTED_BRACKET_PATH)
    record = pd.read_csv(PROJECTED_RECORD_PATH)

    weekly = build_weekly_results(games)
    playoff = build_playoff_results(bracket)
    final_record = build_final_record(record)

    n_upsets = int(weekly["upset"].sum())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        _write_section(
            writer,
            "Weekly Game Results",
            f"One real, randomly-drawn realistic season ({n_upsets} of {len(weekly)} games were upsets vs. the "
            "model's pregame favorite) -- not a guarantee this exact season happens; see "
            "season_simulation_summary.csv for the full probability distribution across every simulation.",
            weekly,
        )
        _write_section(
            writer,
            "Playoff Bracket Results",
            "The same representative season's real playoff bracket -- home_team is always the better seed except "
            "the Super Bowl (no seed to judge an upset against, left blank).",
            playoff,
        )
        _write_section(
            writer,
            "Final Regular-Season Record",
            "The same representative season's real final standings -- consistent by construction with the weekly "
            "results above (literally the same simulated season, not re-derived).",
            final_record,
        )

    print(f"Wrote {REPORT_PATH}")
    print(f"  {len(weekly)} regular-season game results ({n_upsets} upsets)")
    print(f"  {len(playoff)} playoff games")
    print(f"  {len(final_record)} team records")


if __name__ == "__main__":
    main()
