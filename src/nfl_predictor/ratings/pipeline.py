"""Orchestrates the Elo rating engine: loads schedules, replays every game
in chronological order updating team ratings, applies season-to-season mean
reversion, and backtests the resulting win-probability predictions against
both a naive baseline and the market (Vegas moneylines).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_predictor.config import DATA_DIR, PROCESSED_DIR
from nfl_predictor.metrics import brier_score, log_loss
from nfl_predictor.team_codes import canonicalize_teams
from nfl_predictor.ratings.elo import (
    DEFAULT_K,
    DEFAULT_MEAN,
    expected_home_win_prob,
    hfa_from_home_win_rate,
    regress_to_mean,
    update_ratings,
)

GAME_LOG_PATH = PROCESSED_DIR / "elo_game_log.parquet"
CURRENT_RATINGS_PATH = PROCESSED_DIR / "elo_current_ratings.parquet"


@dataclass
class BacktestSummary:
    n_games: int
    hfa_used: float
    elo_brier: float
    elo_log_loss: float
    baseline_brier: float
    baseline_log_loss: float
    market_brier: float | None
    market_log_loss: float | None
    market_coverage: int


def _load_played_games(start_season: int, end_season: int | None) -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "schedules.parquet")
    df = df[df["home_score"].notna() & df["away_score"].notna()].copy()
    df = df[df["season"] >= start_season]
    if end_season is not None:
        df = df[df["season"] <= end_season]
    # No-op for 2021+ (already standard codes); gives relocated franchises
    # (Rams/Chargers/Raiders) continuous identity across their old/new codes
    # when this is run further back in history.
    df = canonicalize_teams(df, ["home_team", "away_team"])
    return df.sort_values(["season", "week", "gameday"]).reset_index(drop=True)


def _moneyline_to_implied_prob(moneyline: float) -> float:
    if moneyline < 0:
        return -moneyline / (-moneyline + 100)
    return 100 / (moneyline + 100)


def _market_home_win_prob(row: dict) -> float | None:
    home_ml, away_ml = row.get("home_moneyline"), row.get("away_moneyline")
    if pd.isna(home_ml) or pd.isna(away_ml):
        return None
    p_home_raw = _moneyline_to_implied_prob(home_ml)
    p_away_raw = _moneyline_to_implied_prob(away_ml)
    # Remove the sportsbook's vig by normalizing the pair to sum to 1.
    return p_home_raw / (p_home_raw + p_away_raw)


def run(
    start_season: int,
    end_season: int | None = None,
    k: float = DEFAULT_K,
    hfa: float | None = None,
    regress_frac: float = 0.4,
    season_adjustments: dict[tuple[int, str], float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, BacktestSummary]:
    """Replay every played game from start_season..end_season in order,
    tracking Elo ratings. Returns (game_log, current_ratings, backtest).

    hfa=None estimates home-field advantage empirically from this same
    dataset's home win rate (see elo.hfa_from_home_win_rate) rather than
    hardcoding a constant -- note this means the estimate has "seen" the
    full sample rather than being purely out-of-sample, a standard practical
    simplification for this stage.

    season_adjustments: optional {(season, team): elo_points} applied on top
    of the regular regress_to_mean shift at each season boundary -- see
    ratings/adjustment.py, which fits these from offseason signals
    (coaching change, QB continuity, draft capital).
    """
    games = _load_played_games(start_season, end_season)
    if games.empty:
        raise ValueError(f"No played games found for seasons {start_season}-{end_season}")

    if hfa is None:
        home_win_rate = (games["home_score"] > games["away_score"]).mean()
        hfa = hfa_from_home_win_rate(home_win_rate)

    ratings: dict[str, float] = {}
    current_season: int | None = None
    rows = []

    for game in games.itertuples(index=False):
        if current_season is not None and game.season != current_season:
            new_season = game.season
            ratings = {
                team: regress_to_mean(r, DEFAULT_MEAN, regress_frac)
                + (season_adjustments or {}).get((new_season, team), 0.0)
                for team, r in ratings.items()
            }
        current_season = game.season

        elo_home = ratings.setdefault(game.home_team, DEFAULT_MEAN)
        elo_away = ratings.setdefault(game.away_team, DEFAULT_MEAN)

        pred_home_win_prob = expected_home_win_prob(elo_home, elo_away, hfa)
        new_elo_home, new_elo_away = update_ratings(
            elo_home, elo_away, game.home_score, game.away_score, hfa, k
        )
        ratings[game.home_team] = new_elo_home
        ratings[game.away_team] = new_elo_away

        actual = (
            1.0
            if game.home_score > game.away_score
            else 0.0
            if game.home_score < game.away_score
            else 0.5
        )
        rows.append(
            {
                "game_id": game.game_id,
                "season": game.season,
                "week": game.week,
                "game_type": game.game_type,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "home_score": game.home_score,
                "away_score": game.away_score,
                "pregame_elo_home": elo_home,
                "pregame_elo_away": elo_away,
                "pred_home_win_prob": pred_home_win_prob,
                "actual_home_win": actual,
                "postgame_elo_home": new_elo_home,
                "postgame_elo_away": new_elo_away,
                "market_home_win_prob": _market_home_win_prob(game._asdict()),
            }
        )

    game_log = pd.DataFrame(rows)

    current_ratings = (
        pd.DataFrame({"team": list(ratings.keys()), "elo_rating": list(ratings.values())})
        .sort_values("elo_rating", ascending=False)
        .reset_index(drop=True)
    )

    preds = game_log["pred_home_win_prob"].to_numpy()
    actuals = game_log["actual_home_win"].to_numpy()
    baseline = np.full_like(preds, 0.5)

    market_mask = game_log["market_home_win_prob"].notna()
    market_preds = game_log.loc[market_mask, "market_home_win_prob"].to_numpy()
    market_actuals = game_log.loc[market_mask, "actual_home_win"].to_numpy()

    summary = BacktestSummary(
        n_games=len(game_log),
        hfa_used=hfa,
        elo_brier=brier_score(preds, actuals),
        elo_log_loss=log_loss(preds, actuals),
        baseline_brier=brier_score(baseline, actuals),
        baseline_log_loss=log_loss(baseline, actuals),
        market_brier=brier_score(market_preds, market_actuals) if market_mask.any() else None,
        market_log_loss=log_loss(market_preds, market_actuals) if market_mask.any() else None,
        market_coverage=int(market_mask.sum()),
    )

    return game_log, current_ratings, summary


def save(game_log: pd.DataFrame, current_ratings: pd.DataFrame) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    game_log.to_parquet(GAME_LOG_PATH, index=False)
    current_ratings.to_parquet(CURRENT_RATINGS_PATH, index=False)
