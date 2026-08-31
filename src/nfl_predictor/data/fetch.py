"""Thin wrappers around nfl_data_py, one per dataset.

Every fetch function takes a list of seasons and returns a DataFrame, even
when the underlying nfl_data_py call ignores the years argument (team_desc)
-- this keeps the pipeline's dispatch loop uniform.
"""

from __future__ import annotations

import logging

import nfl_data_py as nfl
import pandas as pd

log = logging.getLogger(__name__)


def fetch_schedules(years: list[int]) -> pd.DataFrame:
    return nfl.import_schedules(years)


def fetch_team_desc(years: list[int]) -> pd.DataFrame:
    return nfl.import_team_desc()


def fetch_seasonal_data(years: list[int]) -> pd.DataFrame:
    return nfl.import_seasonal_data(years)


def fetch_weekly_data(years: list[int]) -> pd.DataFrame:
    return nfl.import_weekly_data(years)


def fetch_rosters(years: list[int]) -> pd.DataFrame:
    df = nfl.import_seasonal_rosters(years)
    # jersey_number/draft_number come back as str for some historical seasons
    # and float for others (confirmed: 2007/2015 str, 2020/2025 float) --
    # pyarrow can't write a column that's genuinely mixed-type across rows,
    # so coerce to a single numeric dtype before it ever reaches Parquet.
    for col in ("jersey_number", "draft_number"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_injuries(years: list[int]) -> pd.DataFrame:
    return nfl.import_injuries(years)


def fetch_depth_charts(years: list[int]) -> pd.DataFrame:
    return nfl.import_depth_charts(years)


def fetch_draft_picks(years: list[int]) -> pd.DataFrame:
    return nfl.import_draft_picks(years)


def fetch_combine_data(years: list[int]) -> pd.DataFrame:
    return nfl.import_combine_data(years)


def fetch_win_totals(years: list[int]) -> pd.DataFrame:
    return nfl.import_win_totals(years)


def fetch_pbp(years: list[int]) -> pd.DataFrame:
    return nfl.import_pbp_data(years, downcast=True)


def fetch_pfr_def_stats(years: list[int]) -> pd.DataFrame:
    return nfl.import_seasonal_pfr("def", years)


# Registry consumed by the pipeline orchestrator. Keys double as the
# --datasets CLI values and the output parquet filenames.
DATASETS = {
    "schedules": fetch_schedules,
    "team_desc": fetch_team_desc,
    "seasonal_data": fetch_seasonal_data,
    "weekly_data": fetch_weekly_data,
    "rosters": fetch_rosters,
    "injuries": fetch_injuries,
    "depth_charts": fetch_depth_charts,
    "draft_picks": fetch_draft_picks,
    "combine": fetch_combine_data,
    "win_totals": fetch_win_totals,
    "pbp": fetch_pbp,
    "pfr_def_stats": fetch_pfr_def_stats,
}
