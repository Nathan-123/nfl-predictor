"""Network smoke test: pulls a small, cheap season range for every dataset
and checks each lands as a non-empty, sane-looking DataFrame. Requires
internet access (hits nfl_data_py's hosted data files).

All fetched output is redirected to a throwaway tmp dir for the duration of
this module (see isolated_data_dir below) -- these tests must never write
into the real data/raw/ cache.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.config import PARTITIONED_DATASETS
from nfl_predictor.data import pipeline

SMOKE_START, SMOKE_END = 2023, 2024

EXPECTED_COLUMNS = {
    "schedules": {"season", "week", "home_team", "away_team", "home_score", "away_score"},
    "team_desc": {"team_abbr", "team_name"},
    "seasonal_data": {"season", "player_id"},
    "weekly_data": {"season", "week", "player_id"},
    "rosters": {"season", "team", "player_id"},
    "injuries": {"season", "week", "team"},
    "depth_charts": {"season", "club_code"},
    "draft_picks": {"season", "team"},
    "combine": {"season", "player_name"},
    "win_totals": {"season", "abbr"},
    "pbp": {"season", "week", "game_id", "posteam", "defteam", "epa", "play_type"},
}

# nfl_data_py's win_totals source is explicitly flagged upstream as "in flux
# and may be out of date" -- it can legitimately return 0 rows for recent
# seasons. Everything else is expected to have data for SMOKE_START/END.
KNOWN_EMPTY_OK = {"win_totals"}


@pytest.fixture(scope="module", autouse=True)
def isolated_data_dir(tmp_path_factory):
    """Redirect pipeline.DATA_DIR/MANIFEST_PATH to a throwaway tmp dir for
    this whole module. pipeline.py did `from nfl_predictor.config import
    DATA_DIR`, so patching config.DATA_DIR alone would not affect the name
    already bound inside pipeline.py -- patch pipeline's own attributes.
    """
    tmp_dir = tmp_path_factory.mktemp("data_raw")
    original_data_dir, original_manifest_path = pipeline.DATA_DIR, pipeline.MANIFEST_PATH
    pipeline.DATA_DIR = tmp_dir
    pipeline.MANIFEST_PATH = tmp_dir / "manifest.json"
    try:
        yield
    finally:
        pipeline.DATA_DIR, pipeline.MANIFEST_PATH = original_data_dir, original_manifest_path


@pytest.fixture(scope="module")
def results():
    return {r.name: r for r in pipeline.run(SMOKE_START, SMOKE_END, list(EXPECTED_COLUMNS.keys()))}


def _read_dataset(name: str) -> pd.DataFrame:
    if name in PARTITIONED_DATASETS:
        parts = sorted((pipeline.DATA_DIR / name).glob("*.parquet"))
        return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    return pd.read_parquet(pipeline.DATA_DIR / f"{name}.parquet")


@pytest.mark.parametrize("name", EXPECTED_COLUMNS.keys())
def test_dataset_fetches_nonempty(results, name):
    result = results[name]
    if name in KNOWN_EMPTY_OK:
        assert result.status in ("ok", "empty"), f"{name}: {result.error}"
        return
    assert result.status == "ok", f"{name}: {result.error or 'no rows returned'}"
    assert result.rows > 0


@pytest.mark.parametrize("name", EXPECTED_COLUMNS.keys())
def test_dataset_has_expected_columns(results, name):
    result = results[name]
    if result.status != "ok":
        pytest.skip(f"{name} did not fetch successfully")
    df = _read_dataset(name)
    missing = EXPECTED_COLUMNS[name] - set(df.columns)
    assert not missing, f"{name} missing expected columns: {missing}"


def test_win_totals_has_data_for_a_known_good_range():
    # win_totals coverage is inconsistent for recent seasons (see KNOWN_EMPTY_OK
    # above); confirm the fetch path itself works against a range that reliably
    # has data upstream.
    [result] = pipeline.run(2018, 2019, ["win_totals"])
    assert result.status == "ok", result.error
    df = _read_dataset("win_totals")
    assert not (EXPECTED_COLUMNS["win_totals"] - set(df.columns))


def test_pbp_is_partitioned_by_season(results):
    assert results["pbp"].status == "ok"
    season_files = sorted(p.name for p in (pipeline.DATA_DIR / "pbp").glob("*.parquet"))
    assert season_files == ["2023.parquet", "2024.parquet"]


def test_schedules_game_count_is_plausible(results):
    df = _read_dataset("schedules")
    games_per_season = df.groupby("season").size()
    # Regular + postseason games; a full season is >= 272 (17 weeks x 16 games) regular games alone.
    assert (games_per_season >= 260).all(), games_per_season.to_dict()
