"""Central configuration: paths and dataset defaults for the data pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.json"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MANUAL_DIR = ROOT_DIR / "data" / "manual"
PBP_DIR = DATA_DIR / "pbp"

# First season of the 17-game regular season (2021+). Earlier seasons used a
# 16-game schedule, which skews any rate/total stat computed across a full
# season, so this default avoids silently mixing schedule lengths.
DEFAULT_START_SEASON = 2021

# Datasets that raise an error (rather than returning an empty frame) when
# queried before their earliest available season. Determined empirically
# against nfl_data_py; update if the upstream source changes.
MIN_SEASON = {
    "injuries": 2009,
    "pfr_def_stats": 2018,
    "snap_counts": 2013,
}

# Datasets that always get fetched further back than a run's requested start
# season. draft_picks needs mature (4+ year old) classes to fit the
# pick-value curve (ratings/offseason_features.py), which the 2021+ default
# alone wouldn't leave old enough. schedules/pbp/rosters need the same
# widening for the offseason-adjustment regression, but that's specific to
# one caller, not every run, so they're widened with a one-off backfill
# instead (`run_pipeline.py --datasets schedules,rosters,pbp --start-season
# 2007`) rather than listed here.
EXTENDED_START_SEASON = {
    "draft_picks": 2005,
}

# Start season for the offseason-adjustment regression's training data,
# separate from DEFAULT_START_SEASON, which still governs the production
# Elo engine's current ratings/backtest window. Requires the one-off
# backfill above to have run at least once.
DEFAULT_REGRESSION_START_SEASON = 2007

# Datasets written as data/raw/<name>/<season>.parquet (one file per season)
# instead of a single data/raw/<name>.parquet. Use for datasets large enough
# that consumers usually want one season at a time: currently just play-by-
# play, at about 20MB/season vs under 1MB for everything else.
PARTITIONED_DATASETS = {"pbp"}
