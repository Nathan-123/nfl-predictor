"""Preseason signals for each (season, team): did the head coach change,
how does the incoming starting QB compare to who actually played last year,
how much draft capital was added, and (2026 only so far) the market's
preseason win total. See ratings/adjustment.py for how these get combined
into an Elo starting-rating adjustment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_predictor.config import DATA_DIR, MANUAL_DIR, PBP_DIR

MIN_DROPBACKS = 100  # below this, a QB-season's EPA/dropback is too noisy to trust
DRAFT_CURVE_MATURITY_YEARS = 4  # only fit the pick-value curve on classes at least this old

# draft_picks.team comes from PFR and uses different abbreviations than every
# other dataset (schedules, rosters, ...), which follow nflverse's style --
# normalize to nflverse codes so team keys match across the whole pipeline.
PFR_TO_NFLVERSE_TEAM = {
    "GNB": "GB",
    "KAN": "KC",
    "LAR": "LA",
    "LVR": "LV",
    "NOR": "NO",
    "NWE": "NE",
    "SFO": "SF",
    "TAM": "TB",
}


def _week1_rows(schedules: pd.DataFrame) -> pd.DataFrame:
    return schedules[(schedules["game_type"] == "REG") & (schedules["week"] == 1)]


def build_coach_by_team_season(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, team) with that team's Week-1 head coach."""
    week1 = _week1_rows(schedules)
    home = week1[["season", "home_team", "home_coach"]].rename(columns={"home_team": "team", "home_coach": "coach"})
    away = week1[["season", "away_team", "away_coach"]].rename(columns={"away_team": "team", "away_coach": "coach"})
    return pd.concat([home, away], ignore_index=True).dropna(subset=["coach"])


def build_coaching_changes(schedules: pd.DataFrame) -> pd.DataFrame:
    """season, team, coaching_change (1.0 if the Week-1 HC differs from the
    prior season's Week-1 HC, 0.0 if same, NaN if no prior-season data)."""
    coaches = build_coach_by_team_season(schedules)
    prior = coaches.copy()
    prior["season"] = prior["season"] + 1
    prior = prior.rename(columns={"coach": "prior_coach"})
    merged = coaches.merge(prior, on=["season", "team"], how="left")
    merged["coaching_change"] = np.where(
        merged["prior_coach"].isna(), np.nan, (merged["coach"] != merged["prior_coach"]).astype(float)
    )
    return merged[["season", "team", "coaching_change"]]


def build_week1_starters(schedules: pd.DataFrame) -> pd.DataFrame:
    """season, team, qb_id for that season's Week-1 starting QB."""
    week1 = _week1_rows(schedules)
    home = week1[["season", "home_team", "home_qb_id"]].rename(columns={"home_team": "team", "home_qb_id": "qb_id"})
    away = week1[["season", "away_team", "away_qb_id"]].rename(columns={"away_team": "team", "away_qb_id": "qb_id"})
    return pd.concat([home, away], ignore_index=True).dropna(subset=["qb_id"])


def compute_qb_value_by_season(seasons: list[int]) -> pd.DataFrame:
    """season, passer_id, epa_per_dropback, n_dropbacks -- computed directly
    from cached play-by-play so this works for any season we've fetched pbp
    for, independent of nfl_data_py's seasonal_data/weekly_data lag."""
    frames = []
    for season in seasons:
        path = PBP_DIR / f"{season}.parquet"
        if not path.exists():
            continue
        pbp = pd.read_parquet(path, columns=["passer_id", "qb_dropback", "qb_epa"])
        pbp = pbp[(pbp["qb_dropback"] == 1) & pbp["passer_id"].notna() & pbp["qb_epa"].notna()]
        agg = pbp.groupby("passer_id")["qb_epa"].agg(epa_per_dropback="mean", n_dropbacks="count").reset_index()
        agg["season"] = season
        frames.append(agg)
    if not frames:
        return pd.DataFrame(columns=["season", "passer_id", "epa_per_dropback", "n_dropbacks"])
    return pd.concat(frames, ignore_index=True)


def _replacement_level_by_season(qb_values: pd.DataFrame) -> dict[int, float]:
    """25th percentile of EPA/dropback among qualified (>= MIN_DROPBACKS) QBs
    that season -- a stand-in for "an unproven/backup-level starter"."""
    qualified = qb_values[qb_values["n_dropbacks"] >= MIN_DROPBACKS]
    levels = qualified.groupby("season")["epa_per_dropback"].quantile(0.25).to_dict()
    fallback = qualified["epa_per_dropback"].quantile(0.25) if not qualified.empty else 0.0
    return levels, fallback


def build_qb_value_deltas(schedules: pd.DataFrame) -> pd.DataFrame:
    """season, team, qb_value_delta: how much better/worse this season's
    Week-1 starter looks (by their OWN prior-season EPA/dropback, replacement
    level if unproven) than last season's actual starter's actual prior-
    season production. Zero if the starter didn't change."""
    starters = build_week1_starters(schedules)
    seasons = sorted(schedules["season"].unique())
    qb_values = compute_qb_value_by_season(seasons)
    replacement_by_season, fallback_level = _replacement_level_by_season(qb_values)

    qualified = qb_values[qb_values["n_dropbacks"] >= MIN_DROPBACKS].set_index(["passer_id", "season"])[
        "epa_per_dropback"
    ]

    def value_of(qb_id: str | None, as_of_season: int) -> float:
        if qb_id is not None and (qb_id, as_of_season) in qualified.index:
            return qualified.loc[(qb_id, as_of_season)]
        return replacement_by_season.get(as_of_season, fallback_level)

    prior_starters = starters.copy()
    prior_starters["season"] = prior_starters["season"] + 1
    prior_starters = prior_starters.rename(columns={"qb_id": "prior_qb_id"})
    merged = starters.merge(prior_starters, on=["season", "team"], how="inner")  # need a prior season to compare to

    rows = []
    for row in merged.itertuples(index=False):
        prior_season = row.season - 1
        new_value = value_of(row.qb_id, prior_season)
        old_value = value_of(row.prior_qb_id, prior_season)
        rows.append({"season": row.season, "team": row.team, "qb_value_delta": new_value - old_value})
    return pd.DataFrame(rows)


def fit_draft_value_curve(draft_picks: pd.DataFrame, as_of_season: int):
    """Fit expected w_av ~ a + b*log(pick) on mature draft classes (season <=
    as_of_season - DRAFT_CURVE_MATURITY_YEARS), so it can be applied to a
    current draft class whose players haven't accrued career value yet.
    Returns a function pick_number -> expected value (clamped >= 0)."""
    mature = draft_picks[draft_picks["season"] <= as_of_season - DRAFT_CURVE_MATURITY_YEARS]
    mature = mature.dropna(subset=["pick", "w_av"])
    if len(mature) < 20:
        return lambda pick: 0.0

    x = np.log(mature["pick"].to_numpy(dtype=float))
    y = mature["w_av"].to_numpy(dtype=float)
    A = np.vstack([np.ones_like(x), x]).T
    intercept, slope = np.linalg.lstsq(A, y, rcond=None)[0]

    def value_of_pick(pick: float) -> float:
        return max(0.0, intercept + slope * np.log(pick))

    return value_of_pick


def build_draft_capital_added(draft_picks: pd.DataFrame) -> pd.DataFrame:
    """season, team, draft_capital_added: sum of expected value (via the
    pick-value curve) over a team's picks in that season's draft class."""
    draft_picks = draft_picks.copy()
    draft_picks["team"] = draft_picks["team"].replace(PFR_TO_NFLVERSE_TEAM)
    rows = []
    for season in sorted(draft_picks["season"].unique()):
        value_curve = fit_draft_value_curve(draft_picks, as_of_season=season)
        season_picks = draft_picks[draft_picks["season"] == season].dropna(subset=["pick", "team"])
        added = season_picks.assign(pick_value=season_picks["pick"].apply(value_curve)).groupby("team")["pick_value"].sum()
        rows.extend({"season": season, "team": team, "draft_capital_added": value} for team, value in added.items())
    return pd.DataFrame(rows)


def load_preseason_win_totals() -> pd.DataFrame:
    """season, team, preseason_win_total -- from the manually-curated CSV.
    Sparse (only seasons we've sourced so far); see data/manual/README-ish
    comment at the top of the CSV for coverage."""
    path = MANUAL_DIR / "preseason_win_totals.csv"
    if not path.exists():
        return pd.DataFrame(columns=["season", "team", "preseason_win_total"])
    df = pd.read_csv(path)
    return df.rename(columns={"team_abbr": "team", "win_total": "preseason_win_total"})[
        ["season", "team", "preseason_win_total"]
    ]


def build_offseason_features(start_season: int, end_season: int) -> pd.DataFrame:
    """season, team, coaching_change, qb_value_delta, draft_capital_added,
    preseason_win_total for every team-season transition in
    [start_season, end_season] (each row needs a prior season to compare
    against, so the earliest usable season is start_season + 1)."""
    schedules = pd.read_parquet(DATA_DIR / "schedules.parquet")
    schedules = schedules[(schedules["season"] >= start_season) & (schedules["season"] <= end_season)]
    draft_picks = pd.read_parquet(DATA_DIR / "draft_picks.parquet")

    features = build_coaching_changes(schedules)
    features = features.merge(build_qb_value_deltas(schedules), on=["season", "team"], how="left")
    features = features.merge(build_draft_capital_added(draft_picks), on=["season", "team"], how="left")
    features["draft_capital_added"] = features["draft_capital_added"].fillna(0.0)  # no picks that year = 0 added
    features = features.merge(load_preseason_win_totals(), on=["season", "team"], how="left")
    return features[features["season"] > start_season].reset_index(drop=True)
