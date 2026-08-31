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
# season -- keeping the default here avoids silently mixing schedule lengths.
DEFAULT_START_SEASON = 2021

# Datasets that raise an error (rather than returning an empty frame) when
# queried before their earliest available season. Determined empirically
# against nfl_data_py; update if the upstream source changes.
MIN_SEASON = {
    "injuries": 2009,
}

# Datasets that should ALWAYS be fetched further back than a run's requested
# start season, on every call, regardless of caller intent -- reserved for
# cases where that's intrinsically required, not just "wider is nicer".
# draft_picks needs mature (4+ year old) classes to fit an expected-value-
# by-draft-slot curve (ratings/offseason_features.py); the 17-game-era 2021+
# default alone doesn't leave any classes old enough for that.
#
# schedules/pbp/rosters are NOT here on purpose, even though the offseason-
# adjustment regression wants them widened back to 2007 -- unlike
# draft_picks, that's a need specific to one caller
# (ratings.adjustment.fit_adjusted_elo_pipeline's regression_start_season),
# not every pipeline run. Putting them here previously made an unrelated
# small-range smoke test silently balloon into a multi-minute full historical
# pbp refetch. The one-off backfill (`run_pipeline.py --datasets
# schedules,rosters,pbp --start-season 2007`) already widened the on-disk
# cache; that's sufficient without a standing floor.
EXTENDED_START_SEASON = {
    "draft_picks": 2005,
}

# Default start season for the offseason-adjustment regression's *training*
# data (season-transitions used to fit the model), kept separate from
# DEFAULT_START_SEASON which still governs the production Elo engine/current
# ratings/backtest window. A few seasons of Elo warm-up are discarded before
# the first trusted transition -- see fit_adjusted_elo_pipeline. Requires the
# one-off backfill above to have been run at least once.
DEFAULT_REGRESSION_START_SEASON = 2007

# Datasets written as data/raw/<name>/<season>.parquet (one file per season)
# instead of a single data/raw/<name>.parquet. Use for datasets large enough
# that consumers usually want one season at a time -- currently just play-by-
# play (~20MB/season vs <1MB for everything else).
PARTITIONED_DATASETS = {"pbp"}
