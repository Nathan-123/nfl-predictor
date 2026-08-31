"""Tests for the canonical team-code mapping used to reconcile relocation-
era and cross-source abbreviation inconsistencies (see team_codes.py)."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.team_codes import canonicalize_team, canonicalize_teams


def test_relocated_franchises_map_to_current_code():
    assert canonicalize_team("STL") == "LA"
    assert canonicalize_team("SL") == "LA"
    assert canonicalize_team("SD") == "LAC"
    assert canonicalize_team("OAK") == "LV"


def test_pfr_style_codes_map_to_nflverse_style():
    assert canonicalize_team("GNB") == "GB"
    assert canonicalize_team("NWE") == "NE"
    assert canonicalize_team("SDG") == "LAC"


def test_roster_alternate_codes_map_to_standard():
    assert canonicalize_team("ARZ") == "ARI"
    assert canonicalize_team("BLT") == "BAL"
    assert canonicalize_team("CLV") == "CLE"
    assert canonicalize_team("HST") == "HOU"


def test_already_standard_code_is_unchanged():
    for code in ["ARI", "GB", "LA", "LAC", "LV", "KC"]:
        assert canonicalize_team(code) == code


def test_canonicalize_teams_applies_to_a_dataframe_column():
    df = pd.DataFrame({"home_team": ["STL", "ARI"], "away_team": ["SD", "GB"]})
    out = canonicalize_teams(df, ["home_team", "away_team"])
    assert out["home_team"].tolist() == ["LA", "ARI"]
    assert out["away_team"].tolist() == ["LAC", "GB"]
