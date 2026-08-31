"""Canonical NFL team codes.

Different nfl_data_py sources use different, inconsistent abbreviations for
the same franchise, especially for historical seasons:
  - Relocations: the Rams (STL -> LA, 2016), Chargers (SD -> LAC, 2017), and
    Raiders (OAK -> LV, 2020) appear under both their old and new codes
    depending on the season.
  - rosters.parquet carries extra alternate short-codes for some historical
    seasons: ARZ, BLT, CLV, HST, SL.
  - draft_picks.parquet (PFR-sourced) uses its own style throughout,
    including separate old-city codes for the same three relocations.

Confirmed empirically that the current 2021+ production data only ever uses
the 32 standard codes -- this table only matters once historical (pre-2021)
data is in play, but canonicalizing unconditionally is harmless (a no-op)
either way.
"""

from __future__ import annotations

import pandas as pd

CANONICAL_TEAM = {
    # relocations / mid-history renames
    "STL": "LA",
    "SL": "LA",
    "LAR": "LA",
    "SD": "LAC",
    "SDG": "LAC",
    "OAK": "LV",
    "LVR": "LV",
    # alternate short codes seen in rosters.parquet historical seasons
    "ARZ": "ARI",
    "BLT": "BAL",
    "CLV": "CLE",
    "HST": "HOU",
    # PFR-style (draft_picks.parquet) vs. nflverse-style, non-relocation teams
    "GNB": "GB",
    "KAN": "KC",
    "NOR": "NO",
    "NWE": "NE",
    "SFO": "SF",
    "TAM": "TB",
}


def canonicalize_team(code: str) -> str:
    return CANONICAL_TEAM.get(code, code)


def canonicalize_teams(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Return a copy of df with the given team-code columns canonicalized."""
    df = df.copy()
    for col in columns:
        df[col] = df[col].replace(CANONICAL_TEAM)
    return df
