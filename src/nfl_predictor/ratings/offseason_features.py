"""Preseason signals for each (season, team): did the head coach change,
how does the incoming starting QB compare to who actually played last year,
how much draft capital was added, how does the incoming RB/WR/TE room's
prior production compare to what the team actually had last year, and the
market's own preseason win total (covers 2007-2026). See
ratings/adjustment.py for how these get combined into an Elo starting-
rating adjustment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nfl_predictor.config import DATA_DIR, MANUAL_DIR, PBP_DIR
from nfl_predictor.team_codes import canonicalize_teams

MIN_DROPBACKS = 100  # below this, a QB-season's EPA/dropback is too noisy to trust
DRAFT_CURVE_MATURITY_YEARS = 4  # only fit the pick-value curve on classes at least this old
SKILL_POSITIONS = {"RB", "WR", "TE"}
DEFENSE_POSITIONS = {"DB", "DL", "LB", "CB", "DE", "OLB", "DT", "ILB", "MLB", "NT", "SS", "FS", "S"}
MIN_COVERAGE_TARGETS = 10  # below this, a defender's opponent-passer-rating-allowed is too noisy to trust
MIN_DEFENSIVE_SNAPS = 100  # below this, a defender's whole season line is too thin a sample to trust at all
SPECIAL_TEAMS_POSITIONS = {"K", "P"}  # returners aren't tracked as their own roster position


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
    """season, passer_id, epa_per_dropback, n_dropbacks, epa_z. Computed
    directly from cached play-by-play, so it works for any season we've
    fetched pbp for, without depending on nfl_data_py's seasonal_data/
    weekly_data lag. epa_z is a same-season z-score among qualified (>=
    MIN_DROPBACKS) QBs that year. League-wide passing efficiency has drifted
    a lot across eras (mean EPA/play went from -0.002 in 2006 to +0.078 in
    2016 by our own numbers), so raw epa_per_dropback isn't comparable
    across seasons that far apart; epa_z is."""
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
        return pd.DataFrame(columns=["season", "passer_id", "epa_per_dropback", "n_dropbacks", "epa_z"])
    values = pd.concat(frames, ignore_index=True)

    qualified_mask = values["n_dropbacks"] >= MIN_DROPBACKS
    season_stats = values[qualified_mask].groupby("season")["epa_per_dropback"].agg(["mean", "std"])
    values = values.merge(season_stats, on="season", how="left")
    values["epa_z"] = (values["epa_per_dropback"] - values["mean"]) / values["std"].replace(0, np.nan)
    return values.drop(columns=["mean", "std"])


def _replacement_level_by_season(qb_values: pd.DataFrame) -> dict[int, float]:
    """25th percentile of era-normalized EPA/dropback (epa_z) among
    qualified (>= MIN_DROPBACKS) QBs that season. Stands in for "an unproven
    or backup-level starter"."""
    qualified = qb_values[qb_values["n_dropbacks"] >= MIN_DROPBACKS]
    levels = qualified.groupby("season")["epa_z"].quantile(0.25).to_dict()
    fallback = qualified["epa_z"].quantile(0.25) if not qualified.empty else 0.0
    return levels, fallback


def build_presumptive_starters(rosters: pd.DataFrame, qb_values: pd.DataFrame) -> pd.DataFrame:
    """season, team, qb_id. For team-seasons with no real Week-1 starter
    data yet (a season that hasn't been played), picks the roster's QB with
    the most PRIOR-season dropback volume as a presumptive starter (0 if
    none on file, e.g. a true rookie). This correctly picks up offseason
    trades and signings, since rosters.parquet already reflects those --
    spot-checked against a real 2026 trade and it had Myles Garrett listed
    on LA the same day."""
    qb_rosters = rosters[rosters["position"] == "QB"][["season", "team", "player_id", "week"]]
    qb_rosters = qb_rosters.sort_values("week", na_position="first").drop_duplicates(
        subset=["season", "player_id"], keep="last"
    )[["season", "team", "player_id"]]
    if qb_rosters.empty:
        return pd.DataFrame(columns=["season", "team", "qb_id"])

    candidates = qb_rosters.copy()
    candidates["prior_season"] = candidates["season"] - 1
    candidates = candidates.merge(
        qb_values.rename(columns={"season": "prior_season", "passer_id": "player_id"}),
        on=["prior_season", "player_id"],
        how="left",
    )
    candidates["n_dropbacks"] = candidates["n_dropbacks"].fillna(0)

    idx = candidates.groupby(["season", "team"])["n_dropbacks"].idxmax()
    return candidates.loc[idx, ["season", "team", "player_id"]].rename(columns={"player_id": "qb_id"}).reset_index(
        drop=True
    )


def build_qb_value_deltas(schedules: pd.DataFrame, rosters: pd.DataFrame | None = None) -> pd.DataFrame:
    """season, team, qb_value_delta: how much better or worse this season's
    Week-1 starter looks (by their OWN prior-season era-normalized EPA/
    dropback, or replacement level if unproven) than last season's actual
    starter's actual prior-season production. Zero if the starter didn't
    change. Comparable across eras since both sides are z-scores within
    their own season, not raw EPA.

    rosters: if given, team-seasons with no real Week-1 starter yet (a
    season that hasn't been played) fall back to a presumptive starter (see
    build_presumptive_starters) instead of being dropped entirely. Real
    Week-1 data always wins where it exists. Omit for historical-only use --
    every already-played season comes out the same either way."""
    starters = build_week1_starters(schedules)
    seasons = sorted(schedules["season"].unique())
    qb_values = compute_qb_value_by_season(seasons)
    replacement_by_season, fallback_level = _replacement_level_by_season(qb_values)

    if rosters is not None:
        presumptive = build_presumptive_starters(rosters, qb_values)
        known = starters[["season", "team"]].drop_duplicates()
        presumptive = presumptive.merge(known, on=["season", "team"], how="left", indicator=True)
        presumptive = presumptive[presumptive["_merge"] == "left_only"].drop(columns=["_merge"])
        starters = pd.concat([starters, presumptive], ignore_index=True)

    qualified = qb_values[qb_values["n_dropbacks"] >= MIN_DROPBACKS].set_index(["passer_id", "season"])["epa_z"]

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
    """Fits expected w_av ~ a + b*log(pick) on mature draft classes (season
    <= as_of_season - DRAFT_CURVE_MATURITY_YEARS), so it can be applied to a
    current draft class whose players haven't accrued career value yet.
    Returns a function pick_number -> expected value (clamped at >= 0)."""
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
    draft_picks = canonicalize_teams(draft_picks, ["team"])
    rows = []
    for season in sorted(draft_picks["season"].unique()):
        value_curve = fit_draft_value_curve(draft_picks, as_of_season=season)
        season_picks = draft_picks[draft_picks["season"] == season].dropna(subset=["pick", "team"])
        added = season_picks.assign(pick_value=season_picks["pick"].apply(value_curve)).groupby("team")["pick_value"].sum()
        rows.extend({"season": season, "team": team, "draft_capital_added": value} for team, value in added.items())
    return pd.DataFrame(rows)


def compute_skill_value_by_season(seasons: list[int]) -> pd.DataFrame:
    """season, player_id, skill_value: a SUM (not a rate) of EPA on plays
    where the player was the rusher, plus EPA on plays where they were the
    targeted receiver (completions and incompletions both, the standard
    targeted-EPA convention). Summing rather than averaging lets a player's
    value scale with usage/opportunity, not just efficiency, so a barely-
    used camp body ends up near zero without needing to be filtered out
    separately."""
    frames = []
    for season in seasons:
        path = PBP_DIR / f"{season}.parquet"
        if not path.exists():
            continue
        pbp = pd.read_parquet(path, columns=["rusher_id", "receiver_id", "epa", "rush", "pass"])
        rush_value = (
            pbp[(pbp["rush"] == 1) & pbp["rusher_id"].notna() & pbp["epa"].notna()]
            .groupby("rusher_id")["epa"]
            .sum()
            .rename_axis("player_id")
        )
        rec_value = (
            pbp[(pbp["pass"] == 1) & pbp["receiver_id"].notna() & pbp["epa"].notna()]
            .groupby("receiver_id")["epa"]
            .sum()
            .rename_axis("player_id")
        )
        combined = rush_value.add(rec_value, fill_value=0.0).rename("skill_value").reset_index()
        combined["season"] = season
        frames.append(combined)
    if not frames:
        return pd.DataFrame(columns=["season", "player_id", "skill_value"])
    return pd.concat(frames, ignore_index=True)


def build_skill_value_deltas(rosters: pd.DataFrame) -> pd.DataFrame:
    """season, team, skill_value_delta: the RB/WR/TE room's incoming value
    (each player on this season's roster, valued by their own prior-season
    production wherever they played, which is what picks up a trade) minus
    the team's own actual prior-season production. Generalizes
    qb_value_delta from one starter to a whole position group.

    Values are era-normalized: each player's raw skill_value gets converted
    to a same-season z-score against that season's full population before
    anything is summed, so a delta means "how many standard deviations,"
    not a raw EPA total that isn't comparable across eras.
    """
    # rosters.parquet has one row per (season, player_id) except for a
    # player traded mid-season (775 such duplicate pairs by our count,
    # mostly 2007-2015). Keep their most recent team that season (highest
    # `week`) so every downstream (season, player_id) lookup is unambiguous.
    skill_rosters = rosters[rosters["position"].isin(SKILL_POSITIONS)][["season", "team", "player_id", "week"]]
    skill_rosters = skill_rosters.sort_values("week", na_position="first").drop_duplicates(
        subset=["season", "player_id"], keep="last"
    )[["season", "team", "player_id"]]
    seasons = sorted(rosters["season"].unique())
    player_values = compute_skill_value_by_season(seasons)

    # Every rostered RB/WR/TE that season, zero-filled where they had no
    # qualifying plays, then z-scored against that season's full roster-wide
    # population.
    roster_values = skill_rosters.merge(player_values, on=["season", "player_id"], how="left")
    roster_values["skill_value"] = roster_values["skill_value"].fillna(0.0)
    season_stats = roster_values.groupby("season")["skill_value"].agg(["mean", "std"])
    roster_values = roster_values.merge(season_stats, on="season", how="left")
    roster_values["skill_value_z"] = (
        (roster_values["skill_value"] - roster_values["mean"]) / roster_values["std"].replace(0, np.nan)
    ).fillna(0.0)

    # 25th percentile of that season's z-scored values, as a replacement-
    # level stand-in for a rookie/newcomer with no NFL roster history --
    # same idea as the QB replacement level, just built from the full
    # roster population instead of a qualified-QB list.
    replacement_by_season = roster_values.groupby("season")["skill_value_z"].quantile(0.25).to_dict()
    fallback_level = roster_values["skill_value_z"].quantile(0.25) if not roster_values.empty else 0.0
    value_lookup = roster_values.set_index(["season", "player_id"])["skill_value_z"].sort_index()

    def value_of(player_id: str, as_of_season: int) -> float:
        if (as_of_season, player_id) in value_lookup.index:
            return value_lookup.loc[(as_of_season, player_id)]
        return replacement_by_season.get(as_of_season, fallback_level)

    # What each team actually got from its own RB/WR/TE room, per season --
    # this is the baseline the following season's transition compares against.
    team_season_value = roster_values.groupby(["season", "team"])["skill_value_z"].sum().reset_index()
    prior_team_value = team_season_value.copy()
    prior_team_value["season"] = prior_team_value["season"] + 1
    prior_team_value = prior_team_value.rename(columns={"skill_value_z": "prior_team_skill_value"})

    # This season's roster, valued by each player's prior-season production.
    incoming = skill_rosters.copy()
    incoming["value_season"] = incoming["season"] - 1
    incoming["skill_value_z"] = incoming.apply(lambda r: value_of(r["player_id"], r["value_season"]), axis=1)
    incoming_value = incoming.groupby(["season", "team"])["skill_value_z"].sum().reset_index()
    incoming_value = incoming_value.rename(columns={"skill_value_z": "incoming_skill_value"})

    merged = incoming_value.merge(prior_team_value, on=["season", "team"], how="left")
    merged["prior_team_skill_value"] = merged["prior_team_skill_value"].fillna(0.0)
    merged["skill_value_delta"] = merged["incoming_skill_value"] - merged["prior_team_skill_value"]
    return merged[["season", "team", "skill_value_delta"]]


def compute_special_teams_value_by_season(seasons: list[int]) -> pd.DataFrame:
    """season, player_id, special_teams_value: a SUM of EPA on plays where
    the player was the kicker (field goal or extra point attempt), plus EPA
    on plays where they were the punter. A player is essentially always one
    or the other, so this just picks up whichever applies, same shape as
    compute_skill_value_by_season combining rushing and receiving EPA."""
    frames = []
    for season in seasons:
        path = PBP_DIR / f"{season}.parquet"
        if not path.exists():
            continue
        pbp = pd.read_parquet(
            path,
            columns=[
                "kicker_player_id",
                "punter_player_id",
                "epa",
                "field_goal_attempt",
                "extra_point_attempt",
                "punt_attempt",
            ],
        )
        kicking_mask = (pbp["field_goal_attempt"] == 1) | (pbp["extra_point_attempt"] == 1)
        kicking_value = (
            pbp[kicking_mask & pbp["kicker_player_id"].notna() & pbp["epa"].notna()]
            .groupby("kicker_player_id")["epa"]
            .sum()
            .rename_axis("player_id")
        )
        punting_value = (
            pbp[(pbp["punt_attempt"] == 1) & pbp["punter_player_id"].notna() & pbp["epa"].notna()]
            .groupby("punter_player_id")["epa"]
            .sum()
            .rename_axis("player_id")
        )
        combined = kicking_value.add(punting_value, fill_value=0.0).rename("special_teams_value").reset_index()
        combined["season"] = season
        frames.append(combined)
    if not frames:
        return pd.DataFrame(columns=["season", "player_id", "special_teams_value"])
    return pd.concat(frames, ignore_index=True)


def build_special_teams_value_deltas(rosters: pd.DataFrame) -> pd.DataFrame:
    """season, team, special_teams_value_delta: same incoming-vs-prior-team-
    value structure as build_skill_value_deltas, for the K/P positions. Uses
    rosters' own player_id directly, since kicker_player_id/punter_player_id
    in pbp already use the same gsis format, so no crosswalk is needed
    here, unlike the defensive value delta's pfr_id situation."""
    st_rosters = rosters[rosters["position"].isin(SPECIAL_TEAMS_POSITIONS)][["season", "team", "player_id", "week"]]
    st_rosters = st_rosters.sort_values("week", na_position="first").drop_duplicates(
        subset=["season", "player_id"], keep="last"
    )[["season", "team", "player_id"]]
    seasons = sorted(rosters["season"].unique())
    player_values = compute_special_teams_value_by_season(seasons)

    roster_values = st_rosters.merge(player_values, on=["season", "player_id"], how="left")
    roster_values["special_teams_value"] = roster_values["special_teams_value"].fillna(0.0)
    season_stats = roster_values.groupby("season")["special_teams_value"].agg(["mean", "std"])
    roster_values = roster_values.merge(season_stats, on="season", how="left")
    roster_values["special_teams_value_z"] = (
        (roster_values["special_teams_value"] - roster_values["mean"]) / roster_values["std"].replace(0, np.nan)
    ).fillna(0.0)

    replacement_by_season = roster_values.groupby("season")["special_teams_value_z"].quantile(0.25).to_dict()
    fallback_level = roster_values["special_teams_value_z"].quantile(0.25) if not roster_values.empty else 0.0
    value_lookup = roster_values.set_index(["season", "player_id"])["special_teams_value_z"].sort_index()

    def value_of(player_id: str, as_of_season: int) -> float:
        if (as_of_season, player_id) in value_lookup.index:
            return value_lookup.loc[(as_of_season, player_id)]
        return replacement_by_season.get(as_of_season, fallback_level)

    team_season_value = roster_values.groupby(["season", "team"])["special_teams_value_z"].sum().reset_index()
    prior_team_value = team_season_value.copy()
    prior_team_value["season"] = prior_team_value["season"] + 1
    prior_team_value = prior_team_value.rename(columns={"special_teams_value_z": "prior_team_st_value"})

    incoming = st_rosters.copy()
    incoming["value_season"] = incoming["season"] - 1
    incoming["special_teams_value_z"] = incoming.apply(lambda r: value_of(r["player_id"], r["value_season"]), axis=1)
    incoming_value = incoming.groupby(["season", "team"])["special_teams_value_z"].sum().reset_index()
    incoming_value = incoming_value.rename(columns={"special_teams_value_z": "incoming_st_value"})

    merged = incoming_value.merge(prior_team_value, on=["season", "team"], how="left")
    merged["prior_team_st_value"] = merged["prior_team_st_value"].fillna(0.0)
    merged["special_teams_value_delta"] = merged["incoming_st_value"] - merged["prior_team_st_value"]
    return merged[["season", "team", "special_teams_value_delta"]]


def _secondary_pfr_id_crosswalk() -> pd.Series:
    """gsis-format player_id -> pfr_id, from data/raw/player_ids.parquet (a
    community-maintained cross-site ID table, independent of rosters.
    parquet's own pfr_id column). Used as a fallback only, in
    build_defense_value_deltas: recovers a pfr_id for about 61% of the
    defensive players rosters.parquet has none for, roughly halving the
    crosswalk gap. Real rosters.parquet pfr_id always wins; see
    _with_secondary_pfr_ids."""
    path = DATA_DIR / "player_ids.parquet"
    if not path.exists():
        return pd.Series(dtype=object)
    ids = pd.read_parquet(path, columns=["gsis_id", "pfr_id"])
    ids = ids.dropna(subset=["gsis_id", "pfr_id"]).drop_duplicates("gsis_id")
    return ids.set_index("gsis_id")["pfr_id"]


def _with_secondary_pfr_ids(rosters: pd.DataFrame) -> pd.DataFrame:
    """rosters with pfr_id backfilled from the secondary crosswalk wherever
    rosters.parquet's own pfr_id is missing (a no-op if player_ids.parquet
    hasn't been fetched)."""
    secondary = _secondary_pfr_id_crosswalk()
    if secondary.empty:
        return rosters
    rosters = rosters.copy()
    rosters["pfr_id"] = rosters["pfr_id"].fillna(rosters["player_id"].map(secondary))
    return rosters


def compute_defensive_value_by_season(seasons: list[int]) -> pd.DataFrame:
    """season, pfr_id, defensive_value: a composite of era-normalized
    (same-season z-scored) pass-rush production (prss, "pressures") and
    turnovers (interceptions), plus coverage quality (rat, opponent passer
    rating allowed when targeted, negated since lower is better and only
    counted for defenders meeting MIN_COVERAGE_TARGETS, so a DT/DE targeted
    once all season doesn't contribute pure noise). Z-scored against the
    population of defenders who recorded any stats that season, not the
    full bench-inclusive roster.

    A player-season with fewer than MIN_DEFENSIVE_SNAPS total defensive
    snaps gets forced to exactly 0 regardless of what the raw stats say,
    same reasoning as the MIN_COVERAGE_TARGETS gate on `rat`: a thin sample
    shouldn't be trusted either way. This also keeps build_defense_value_
    deltas' team-level sums symmetric on both sides, instead of the prior
    side (a full season's accumulated roster) outweighing the incoming
    side (a single preseason snapshot) just from raw player count."""
    path = DATA_DIR / "pfr_def_stats.parquet"
    if not path.exists():
        return pd.DataFrame(columns=["season", "pfr_id", "defensive_value"])
    df = pd.read_parquet(path, columns=["season", "pfr_id", "int", "prss", "rat", "tgt", "comb"])
    df = df[df["season"].isin(seasons)].rename(columns={"int": "interceptions"}).copy()
    # PFR includes an extra row for players who changed teams mid-season
    # (a "2TM" row alongside a named-team row for the same season/pfr_id).
    # Its convention isn't a clean aggregate: the "2TM" row sometimes has
    # less recorded activity than the named-team row, not more. Rather than
    # guess at PFR's semantics, just keep whichever duplicate has more
    # recorded tackle activity (comb) as the more complete stat line.
    df = df.sort_values("comb").drop_duplicates(subset=["season", "pfr_id"], keep="last")

    def zscore(values: pd.Series, qualify: pd.Series | None = None) -> pd.Series:
        qualify = pd.Series(True, index=values.index) if qualify is None else qualify
        grouped = values.where(qualify).groupby(df["season"])
        z = (values - grouped.transform("mean")) / grouped.transform("std").replace(0, np.nan)
        return z.where(qualify, 0.0).fillna(0.0)

    qualified_coverage = df["tgt"] >= MIN_COVERAGE_TARGETS
    df["defensive_value"] = (
        zscore(df["prss"]) + zscore(df["interceptions"]) - zscore(df["rat"], qualify=qualified_coverage)
    )

    snap_path = DATA_DIR / "snap_counts.parquet"
    if snap_path.exists():
        snaps = pd.read_parquet(snap_path, columns=["season", "pfr_player_id", "defense_snaps"])
        snaps = snaps[snaps["season"].isin(seasons)]
        season_snaps = (
            snaps.groupby(["season", "pfr_player_id"])["defense_snaps"]
            .sum()
            .rename("total_defense_snaps")
            .reset_index()
        )
        df = df.merge(
            season_snaps, left_on=["season", "pfr_id"], right_on=["season", "pfr_player_id"], how="left"
        )
        df["total_defense_snaps"] = df["total_defense_snaps"].fillna(0.0)
        df.loc[df["total_defense_snaps"] < MIN_DEFENSIVE_SNAPS, "defensive_value"] = 0.0

    return df[["season", "pfr_id", "defensive_value"]]


def build_defense_value_deltas(rosters: pd.DataFrame) -> pd.DataFrame:
    """season, team, defense_value_delta: same incoming-vs-prior-team-value
    structure as build_skill_value_deltas, generalized to the defensive
    front and coverage positions. Joined through rosters' pfr_id crosswalk
    (PFR's def-stats table doesn't use the gsis player_id format everything
    else does) in two passes, rosters.parquet's own pfr_id column first,
    then player_ids.parquet as a fallback (see _with_secondary_pfr_ids).
    Still incomplete: it skews toward practice-squad and inactive players
    neither source has a PFR id for. No PFR defensive data before 2018;
    build_offseason_features zero-fills those rows the same way
    draft_capital_added already is."""
    rosters = _with_secondary_pfr_ids(rosters)
    defense_rosters = rosters[rosters["position"].isin(DEFENSE_POSITIONS) & rosters["pfr_id"].notna()][
        ["season", "team", "pfr_id", "week"]
    ]
    # Same in-season-trade dedup as build_skill_value_deltas: keep the most recent team.
    defense_rosters = defense_rosters.sort_values("week", na_position="first").drop_duplicates(
        subset=["season", "pfr_id"], keep="last"
    )[["season", "team", "pfr_id"]]

    seasons = sorted(rosters["season"].unique())
    player_values = compute_defensive_value_by_season(seasons)
    if player_values.empty:
        return pd.DataFrame(columns=["season", "team", "defense_value_delta"])

    roster_values = defense_rosters.merge(player_values, on=["season", "pfr_id"], how="left")
    roster_values["defensive_value"] = roster_values["defensive_value"].fillna(0.0)

    replacement_by_season = roster_values.groupby("season")["defensive_value"].quantile(0.25).to_dict()
    fallback_level = roster_values["defensive_value"].quantile(0.25) if not roster_values.empty else 0.0
    value_lookup = roster_values.set_index(["season", "pfr_id"])["defensive_value"].sort_index()

    def value_of(pfr_id: str, as_of_season: int) -> float:
        if (as_of_season, pfr_id) in value_lookup.index:
            return value_lookup.loc[(as_of_season, pfr_id)]
        return replacement_by_season.get(as_of_season, fallback_level)

    team_season_value = roster_values.groupby(["season", "team"])["defensive_value"].sum().reset_index()
    prior_team_value = team_season_value.copy()
    prior_team_value["season"] = prior_team_value["season"] + 1
    prior_team_value = prior_team_value.rename(columns={"defensive_value": "prior_team_defensive_value"})

    incoming = defense_rosters.copy()
    incoming["value_season"] = incoming["season"] - 1
    incoming["defensive_value"] = incoming.apply(lambda r: value_of(r["pfr_id"], r["value_season"]), axis=1)
    incoming_value = incoming.groupby(["season", "team"])["defensive_value"].sum().reset_index()
    incoming_value = incoming_value.rename(columns={"defensive_value": "incoming_defensive_value"})

    merged = incoming_value.merge(prior_team_value, on=["season", "team"], how="left")
    merged["prior_team_defensive_value"] = merged["prior_team_defensive_value"].fillna(0.0)
    merged["defense_value_delta"] = merged["incoming_defensive_value"] - merged["prior_team_defensive_value"]
    return merged[["season", "team", "defense_value_delta"]]


def load_preseason_win_totals() -> pd.DataFrame:
    """season, team, preseason_win_total, from the manually-curated CSV
    (data/manual/preseason_win_totals.csv). 2021 came from a sportsbook's
    closing-line archive, 2026 from a FOX Sports preview article, and
    2007-2020 plus 2022-2025 were backfilled from covers.com's
    sportsoddshistory archive (public, no login required, allowed by
    robots.txt), giving full 32-team coverage for every season 2007-2026.
    Used as the 7th regression feature in adjustment.py's FEATURE_COLS."""
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
    [start_season, end_season]. Each row needs a prior season to compare
    against, so the earliest usable season is start_season + 1."""
    schedules = pd.read_parquet(DATA_DIR / "schedules.parquet")
    schedules = schedules[(schedules["season"] >= start_season) & (schedules["season"] <= end_season)]
    schedules = canonicalize_teams(schedules, ["home_team", "away_team"])
    draft_picks = pd.read_parquet(DATA_DIR / "draft_picks.parquet")
    rosters = pd.read_parquet(DATA_DIR / "rosters.parquet")
    rosters = rosters[(rosters["season"] >= start_season) & (rosters["season"] <= end_season)]
    rosters = canonicalize_teams(rosters, ["team"])

    features = build_coaching_changes(schedules)
    features = features.merge(build_qb_value_deltas(schedules, rosters), on=["season", "team"], how="left")
    features = features.merge(build_draft_capital_added(draft_picks), on=["season", "team"], how="left")
    features["draft_capital_added"] = features["draft_capital_added"].fillna(0.0)  # no picks that year = 0 added
    features = features.merge(build_skill_value_deltas(rosters), on=["season", "team"], how="left")
    features = features.merge(build_special_teams_value_deltas(rosters), on=["season", "team"], how="left")
    features = features.merge(build_defense_value_deltas(rosters), on=["season", "team"], how="left")
    features["defense_value_delta"] = features["defense_value_delta"].fillna(0.0)  # no PFR def data before 2018
    features = features.merge(load_preseason_win_totals(), on=["season", "team"], how="left")
    return features[features["season"] > start_season].reset_index(drop=True)
