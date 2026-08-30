"""Central configuration: paths and dataset defaults for the data pipeline."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
MANIFEST_PATH = DATA_DIR / "manifest.json"

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

# Datasets written as data/raw/<name>/<season>.parquet (one file per season)
# instead of a single data/raw/<name>.parquet. Use for datasets large enough
# that consumers usually want one season at a time -- currently just play-by-
# play (~20MB/season vs <1MB for everything else).
PARTITIONED_DATASETS = {"pbp"}
