"""Per-game feature table for the Stage 2 game outcome model: rolling team
EPA form (from play-by-play), in-season QB continuity, schedule/weather
context, and the Stage 1b Elo rating diff.
"""

from __future__ import annotations

import pandas as pd

from nfl_predictor.config import DATA_DIR, PBP_DIR, PROCESSED_DIR

ROLLING_WINDOW = 5

FEATURE_COLS = [
    "elo_diff",
    "home_off_epa_roll",
    "home_def_epa_roll",
    "away_off_epa_roll",
    "away_def_epa_roll",
    "home_qb_changed",
    "away_qb_changed",
    "home_rest",
    "away_rest",
    "rest_diff",
    "div_game",
    "is_dome",
    "temp",
    "wind",
]


def _team_game_epa(seasons: list[int]) -> pd.DataFrame:
    """game_id, season, week, team, off_epa, def_epa -- one row per team per
    game (so each game_id appears twice, once per side)."""
    frames = []
    for season in seasons:
        path = PBP_DIR / f"{season}.parquet"
        if not path.exists():
            continue
        cols = ["game_id", "season", "week", "posteam", "defteam", "epa", "pass", "rush"]
        pbp = pd.read_parquet(path, columns=cols)
        pbp = pbp[((pbp["pass"] == 1) | (pbp["rush"] == 1)) & pbp["epa"].notna()]
        off = pbp.groupby(["game_id", "season", "week", "posteam"])["epa"].mean().rename("off_epa").reset_index()
        off = off.rename(columns={"posteam": "team"})
        deff = pbp.groupby(["game_id", "season", "week", "defteam"])["epa"].mean().rename("def_epa").reset_index()
        deff = deff.rename(columns={"defteam": "team"})
        frames.append(off.merge(deff, on=["game_id", "season", "week", "team"], how="outer"))
    if not frames:
        return pd.DataFrame(columns=["game_id", "season", "week", "team", "off_epa", "def_epa"])
    return pd.concat(frames, ignore_index=True)


def build_rolling_epa(seasons: list[int], window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """game_id, team, off_epa_roll, def_epa_roll -- each team's trailing
    mean EPA over their last `window` games, SHIFTED so a game's own plays
    (and any later game's) never leak into its own feature. Chronological
    order runs across season boundaries (no reset at Week 1) -- "recent
    form" carrying a few games into a new season is intentional, not a bug.
    """
    team_games = _team_game_epa(seasons).sort_values(["team", "season", "week"])
    grouped = team_games.groupby("team")[["off_epa", "def_epa"]]
    rolled = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    team_games = team_games.assign(off_epa_roll=rolled["off_epa"], def_epa_roll=rolled["def_epa"])
    return team_games[["game_id", "team", "off_epa_roll", "def_epa_roll"]]


def _week1_ordered_team_games(schedules: pd.DataFrame) -> pd.DataFrame:
    """game_id, season, week, team, qb_id -- long format, chronologically
    orderable, one row per team per game."""
    home = schedules[["game_id", "season", "week", "home_team", "home_qb_id"]].rename(
        columns={"home_team": "team", "home_qb_id": "qb_id"}
    )
    away = schedules[["game_id", "season", "week", "away_team", "away_qb_id"]].rename(
        columns={"away_team": "team", "away_qb_id": "qb_id"}
    )
    return pd.concat([home, away], ignore_index=True)


def build_qb_continuity_flags(schedules: pd.DataFrame) -> pd.DataFrame:
    """game_id, team, qb_changed (1.0 if this game's starter differs from
    that team's immediately preceding game's starter, 0.0 if same, NaN for
    a team's first game in the dataset). In-season signal -- distinct from
    Stage 1b's offseason QB value delta."""
    long = _week1_ordered_team_games(schedules).sort_values(["team", "season", "week"])
    prev_qb = long.groupby("team")["qb_id"].shift(1)
    long = long.assign(qb_changed=((long["qb_id"] != prev_qb) & prev_qb.notna()).astype(float))
    long.loc[prev_qb.isna(), "qb_changed"] = float("nan")
    return long[["game_id", "team", "qb_changed"]]


def build_context_features(schedules: pd.DataFrame) -> pd.DataFrame:
    """game_id, home_rest, away_rest, rest_diff, div_game, is_dome, temp, wind."""
    df = schedules[["game_id", "home_rest", "away_rest", "div_game", "roof", "temp", "wind"]].copy()
    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(float)
    df["rest_diff"] = df["home_rest"] - df["away_rest"]

    outdoor_temp_mean = df.loc[~df["is_dome"].astype(bool), "temp"].mean()
    outdoor_wind_mean = df.loc[~df["is_dome"].astype(bool), "wind"].mean()
    df.loc[df["is_dome"].astype(bool), "temp"] = 70.0
    df.loc[df["is_dome"].astype(bool), "wind"] = 0.0
    df["temp"] = df["temp"].fillna(outdoor_temp_mean)
    df["wind"] = df["wind"].fillna(outdoor_wind_mean)

    return df.drop(columns=["roof"])


def build_game_features(start_season: int, end_season: int) -> pd.DataFrame:
    """One row per played game with FEATURE_COLS plus actual_home_win,
    home_margin, market_home_win_prob, and spread_line for backtesting."""
    schedules = pd.read_parquet(DATA_DIR / "schedules.parquet")
    schedules = schedules[
        schedules["home_score"].notna()
        & (schedules["season"] >= start_season)
        & (schedules["season"] <= end_season)
    ]

    elo_log = pd.read_parquet(PROCESSED_DIR / "elo_adjusted_game_log.parquet")
    elo_log = elo_log.assign(elo_diff=elo_log["pregame_elo_home"] - elo_log["pregame_elo_away"])

    seasons = sorted(schedules["season"].unique())
    rolling_epa = build_rolling_epa(seasons)
    qb_flags = build_qb_continuity_flags(schedules)
    context = build_context_features(schedules)

    games = elo_log[
        [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "actual_home_win",
            "market_home_win_prob",
            "elo_diff",
            "pred_home_win_prob",
        ]
    ].rename(columns={"pred_home_win_prob": "elo_pred_home_win_prob"})
    games["home_margin"] = elo_log["home_score"] - elo_log["away_score"]
    games = games.merge(schedules[["game_id", "spread_line"]], on="game_id", how="left")

    games = games.merge(
        rolling_epa.rename(columns={"team": "home_team", "off_epa_roll": "home_off_epa_roll", "def_epa_roll": "home_def_epa_roll"}),
        on=["game_id", "home_team"],
        how="left",
    )
    games = games.merge(
        rolling_epa.rename(columns={"team": "away_team", "off_epa_roll": "away_off_epa_roll", "def_epa_roll": "away_def_epa_roll"}),
        on=["game_id", "away_team"],
        how="left",
    )
    games = games.merge(
        qb_flags.rename(columns={"team": "home_team", "qb_changed": "home_qb_changed"}),
        on=["game_id", "home_team"],
        how="left",
    )
    games = games.merge(
        qb_flags.rename(columns={"team": "away_team", "qb_changed": "away_qb_changed"}),
        on=["game_id", "away_team"],
        how="left",
    )
    games = games.merge(context, on="game_id", how="left")

    return games.reset_index(drop=True)
