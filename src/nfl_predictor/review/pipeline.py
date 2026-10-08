"""Builds and appends to data/processed/season_review.csv: one row per
completed game, grading the frozen pregame prediction against what
actually happened. Idempotent -- a (season, game_id) already in the file
is never re-added, so this can be re-run every week (or every day) without
duplicating rows.

Regular-season games are graded against game_win_probabilities.csv, the
Monte Carlo prediction run_season_simulation.py wrote once before the
season started. That file is deliberately never regenerated mid-season
(see scripts/run_weekly_review.py's module docstring), so a regular-season
grade always reflects the actual pregame prediction, not a lookback with
hindsight baked in.

Playoff games don't have a preseason equivalent -- which two teams meet in,
say, the AFC Wild Card round isn't knowable before the season starts, so
there's no frozen per-matchup prediction to grade against. Once a playoff
matchup is real, this grades it against the same Elo-plus-adjustment model
(no refitting, no new features) simply carried forward through the real
games actually played so far -- the closest thing to "what the frozen
model would have said" that can exist for a matchup that didn't exist yet
when the model was fit.
"""

from __future__ import annotations

import logging

import pandas as pd

from nfl_predictor.config import DATA_DIR, DEFAULT_REGRESSION_START_SEASON, DEFAULT_START_SEASON, PBP_DIR, PROCESSED_DIR
from nfl_predictor.gamemodel.features import build_qb_continuity_flags, build_rolling_epa, team_game_epa
from nfl_predictor.review.explain import UpsetContext, describe_correct, explain_upset
from nfl_predictor.team_codes import canonicalize_teams

log = logging.getLogger(__name__)

REVIEW_PATH = PROCESSED_DIR / "season_review.csv"
GAME_PROBABILITIES_PATH = PROCESSED_DIR / "game_win_probabilities.csv"

# Positions whose "Out"/"Doubtful" report_status counts toward the injury
# factor -- the ones a real absence is likely to change a game's outcome.
KEY_INJURY_POSITIONS = {"QB", "RB", "WR", "TE", "T", "G", "C", "OL", "LT", "RT", "LG", "RG"}
INJURY_STATUSES_COUNTED = {"Out", "Doubtful"}

REVIEW_COLUMNS = [
    "season",
    "week",
    "game_type",
    "game_id",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "actual_winner",
    "predicted_winner",
    "predicted_win_prob",
    "prediction_source",
    "correct",
    "review_note",
]


def load_frozen_game_probabilities() -> pd.DataFrame:
    """The regular-season predictions written once, before the season
    started, by run_season_simulation.py. Just reads whatever's on disk --
    this file is never touched by anything in this module."""
    if not GAME_PROBABILITIES_PATH.exists():
        return pd.DataFrame(columns=["week", "game_id", "home_team", "away_team", "home_win_prob", "predicted_winner"])
    return pd.read_csv(GAME_PROBABILITIES_PATH)


def load_completed_season_review(season: int) -> pd.DataFrame:
    """season_review.csv rows for one fully-completed season (regular
    season plus playoffs).

    Not called anywhere in this pipeline yet. It exists so a future
    offseason-adjustment feature (e.g. "how often was this team involved
    in an explainable upset last season") has a ready-made loader to build
    on, without needing to re-derive the CSV-reading logic. Deliberately
    NOT wired into build_offseason_features today: there's no real season
    of review data yet to check whether such a feature would actually help
    the fit, the same way every other feature in this pipeline got checked
    against real outcomes before being trusted (see offseason_features.py's
    module docstring). Call this only for a season that has already
    finished -- never the season currently in progress, which is exactly
    the "don't feed live predictions back into themselves" rule
    run_weekly_review.py's docstring describes."""
    if not REVIEW_PATH.exists():
        return pd.DataFrame(columns=REVIEW_COLUMNS)
    df = pd.read_csv(REVIEW_PATH)
    return df[df["season"] == season].reset_index(drop=True)


def _completed_games(season: int) -> pd.DataFrame:
    schedules = pd.read_parquet(DATA_DIR / "schedules.parquet")
    schedules = schedules[schedules["season"] == season]
    schedules = canonicalize_teams(schedules, ["home_team", "away_team"])
    return schedules[schedules["home_score"].notna() & schedules["away_score"].notna()].copy()


def _already_reviewed_game_ids(season: int) -> set[str]:
    if not REVIEW_PATH.exists():
        return set()
    existing = pd.read_csv(REVIEW_PATH)
    return set(existing.loc[existing["season"] == season, "game_id"])


def _actual_winner(row: pd.Series) -> str:
    if row["home_score"] > row["away_score"]:
        return row["home_team"]
    if row["away_score"] > row["home_score"]:
        return row["away_team"]
    return "TIE"


def _favored_side(home_team: str, away_team: str, home_win_prob: float) -> tuple[str, str, float]:
    if home_win_prob >= 0.5:
        return home_team, away_team, home_win_prob
    return away_team, home_team, 1.0 - home_win_prob


def _turnover_margins(season: int, game_ids: set[str]) -> dict[tuple[str, str], int]:
    """(game_id, team) -> that team's takeaways minus giveaways in that
    game (interceptions thrown + fumbles lost, both ways). Empty if pbp for
    `season` hasn't been fetched yet."""
    path = PBP_DIR / f"{season}.parquet"
    if not path.exists():
        return {}
    pbp = pd.read_parquet(path, columns=["game_id", "posteam", "defteam", "interception", "fumble_lost"])
    pbp = pbp[pbp["game_id"].isin(game_ids)]
    turnovers = pbp["interception"].fillna(0) + pbp["fumble_lost"].fillna(0)
    giveaways = turnovers.groupby([pbp["game_id"], pbp["posteam"]]).sum()
    takeaways = turnovers.groupby([pbp["game_id"], pbp["defteam"]]).sum()
    margin = takeaways.subtract(giveaways, fill_value=0.0)
    return {(gid, team): int(v) for (gid, team), v in margin.items()}


def _key_injury_counts(season: int, week: int, game_type: str) -> dict[str, int]:
    """team -> count of KEY_INJURY_POSITIONS starters listed Out/Doubtful
    for that (season, week, game_type). Empty if injuries.parquet hasn't
    been fetched yet."""
    path = DATA_DIR / "injuries.parquet"
    if not path.exists():
        return {}
    injuries = pd.read_parquet(path, columns=["season", "week", "game_type", "team", "position", "report_status"])
    week_injuries = injuries[
        (injuries["season"] == season)
        & (injuries["week"] == week)
        & (injuries["game_type"] == game_type)
        & injuries["position"].isin(KEY_INJURY_POSITIONS)
        & injuries["report_status"].isin(INJURY_STATUSES_COUNTED)
    ]
    return week_injuries.groupby("team").size().to_dict()


def _grade_game(
    row: pd.Series,
    favored: str,
    underdog: str,
    favored_prob: float,
    prediction_source: str,
    turnover_margins: dict[tuple[str, str], int],
    epa_actual: dict[tuple[str, str], float],
    epa_rolling: dict[tuple[str, str], float],
    qb_changed: dict[tuple[str, str], float],
    key_injuries: dict[str, int],
) -> dict:
    actual_winner = _actual_winner(row)
    correct = actual_winner == favored
    home_won_margin = int(row["home_score"] - row["away_score"])
    favored_margin = home_won_margin if favored == row["home_team"] else -home_won_margin

    if correct:
        note = describe_correct(favored, underdog, favored_prob)
    else:
        ctx = UpsetContext(
            favored_team=favored,
            underdog_team=underdog,
            favored_win_prob=favored_prob,
            margin=favored_margin,
            favored_turnover_margin=turnover_margins.get((row["game_id"], favored)),
            favored_off_epa_actual=epa_actual.get((row["game_id"], favored)),
            favored_off_epa_rolling=epa_rolling.get((row["game_id"], favored)),
            # qb_changed values are 1.0/0.0/NaN (NaN = no prior game on file
            # to compare to, e.g. a team's first tracked game); bool(NaN) is
            # True in Python, so treat missing/NaN as "no change" explicitly
            # rather than silently misreading it as one.
            favored_qb_changed=qb_changed.get((row["game_id"], favored)) == 1.0,
            favored_key_injuries_out=key_injuries.get(favored),
        )
        note = explain_upset(ctx)

    return {
        "season": int(row["season"]),
        "week": int(row["week"]),
        "game_type": row["game_type"],
        "game_id": row["game_id"],
        "home_team": row["home_team"],
        "away_team": row["away_team"],
        "home_score": row["home_score"],
        "away_score": row["away_score"],
        "actual_winner": actual_winner,
        "predicted_winner": favored,
        "predicted_win_prob": round(favored_prob, 4),
        "prediction_source": prediction_source,
        "correct": correct,
        "review_note": note,
    }


def _review_regular_season_games(season: int, games: pd.DataFrame) -> list[dict]:
    predictions = load_frozen_game_probabilities()
    games = games.merge(predictions[["game_id", "home_win_prob"]], on="game_id", how="left")

    missing = games["home_win_prob"].isna()
    if missing.any():
        log.warning(
            "%d regular-season game(s) have no frozen prediction in %s, skipping: %s",
            missing.sum(),
            GAME_PROBABILITIES_PATH,
            games.loc[missing, "game_id"].tolist(),
        )
        games = games[~missing]
    if games.empty:
        return []

    game_ids = set(games["game_id"])
    turnover_margins = _turnover_margins(season, game_ids)

    season_schedules = pd.read_parquet(DATA_DIR / "schedules.parquet")
    season_schedules = canonicalize_teams(season_schedules[season_schedules["season"] == season], ["home_team", "away_team"])
    qb_flags = build_qb_continuity_flags(season_schedules)
    qb_changed = {(r.game_id, r.team): r.qb_changed for r in qb_flags.itertuples(index=False)}

    epa_rolling_df = build_rolling_epa([season])
    epa_rolling = {(r.game_id, r.team): r.off_epa_roll for r in epa_rolling_df.itertuples(index=False)}
    epa_actual_df = team_game_epa([season])
    epa_actual = {(r.game_id, r.team): r.off_epa for r in epa_actual_df.itertuples(index=False)}

    rows = []
    for _, row in games.iterrows():
        favored, underdog, favored_prob = _favored_side(row["home_team"], row["away_team"], row["home_win_prob"])
        key_injuries = _key_injury_counts(season, int(row["week"]), "REG")
        rows.append(
            _grade_game(
                row,
                favored,
                underdog,
                favored_prob,
                "frozen_preseason_simulation",
                turnover_margins,
                epa_actual,
                epa_rolling,
                qb_changed,
                key_injuries,
            )
        )
    return rows


def _review_playoff_games(
    season: int, games: pd.DataFrame, start_season: int, regression_start_season: int
) -> list[dict]:
    from nfl_predictor.ratings.adjustment import fit_adjusted_elo_pipeline

    pipeline = fit_adjusted_elo_pipeline(start_season, regression_start_season=regression_start_season)
    game_log = pipeline.adjusted_log.set_index("game_id")

    game_ids = set(games["game_id"])
    turnover_margins = _turnover_margins(season, game_ids)

    rows = []
    for _, row in games.iterrows():
        if row["game_id"] not in game_log.index:
            log.warning("Playoff game %s missing from the Elo replay, skipping", row["game_id"])
            continue
        home_win_prob = float(game_log.loc[row["game_id"], "pred_home_win_prob"])
        favored, underdog, favored_prob = _favored_side(row["home_team"], row["away_team"], home_win_prob)
        key_injuries = _key_injury_counts(season, int(row["week"]), row["game_type"])
        # Playoff games are single-elimination and don't repeat weekly like
        # the regular season, so there's no meaningful "rolling EPA" or
        # "changed since last game" signal to compute here -- both factors
        # just come back empty, and explain_upset degrades gracefully.
        rows.append(
            _grade_game(
                row,
                favored,
                underdog,
                favored_prob,
                "live_elo_at_kickoff",
                turnover_margins,
                {},
                {},
                {},
                key_injuries,
            )
        )
    return rows


def build_weekly_review(
    season: int,
    weeks: list[int] | None = None,
    start_season: int = DEFAULT_START_SEASON,
    regression_start_season: int = DEFAULT_REGRESSION_START_SEASON,
) -> pd.DataFrame:
    """New season_review.csv rows for every completed game in `season` not
    already in the file, restricted to `weeks` if given. See this module's
    docstring for how regular-season and playoff games each get graded."""
    completed = _completed_games(season)
    if weeks is not None:
        completed = completed[completed["week"].isin(weeks)]
    completed = completed[~completed["game_id"].isin(_already_reviewed_game_ids(season))]
    if completed.empty:
        return pd.DataFrame(columns=REVIEW_COLUMNS)

    reg_games = completed[completed["game_type"] == "REG"]
    post_games = completed[completed["game_type"] != "REG"]

    rows: list[dict] = []
    if not reg_games.empty:
        rows.extend(_review_regular_season_games(season, reg_games))
    if not post_games.empty:
        rows.extend(_review_playoff_games(season, post_games, start_season, regression_start_season))

    return pd.DataFrame(rows, columns=REVIEW_COLUMNS).sort_values(["week", "game_id"]).reset_index(drop=True)


def append_review(new_rows: pd.DataFrame) -> pd.DataFrame:
    """Appends new_rows to REVIEW_PATH (creating it if needed) and returns
    the full, combined table. A no-op (just returns the existing file) if
    new_rows is empty."""
    if REVIEW_PATH.exists():
        existing = pd.read_csv(REVIEW_PATH)
        combined = pd.concat([existing, new_rows], ignore_index=True) if not new_rows.empty else existing
    else:
        combined = new_rows

    combined = combined.drop_duplicates(subset=["season", "game_id"], keep="last")
    combined = combined.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(REVIEW_PATH, index=False)
    return combined
