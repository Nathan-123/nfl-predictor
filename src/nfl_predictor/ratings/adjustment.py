"""Fits how many Elo points each offseason signal (coaching change, QB value
delta, draft capital added, etc.) is actually worth, using real history
instead of hand-picking the magnitudes.

Target: a team's actual end-of-season Elo rating minus its season-start
(post mean-reversion) rating, i.e. how much it out- or under-performed the
naive carryover. Fit with plain OLS, validated with leave-one-season-out
cross-validation since the sample is small (a handful of season transitions
times 32 teams).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "coaching_change",
    "qb_value_delta",
    "draft_capital_added",
    "skill_value_delta",
    "special_teams_value_delta",
    "defense_value_delta",
    "preseason_win_total",
]

# When fitting on a widened regression window, the first few seasons of that
# window get dropped: ratings need a few seasons to move off the synthetic
# 1500 starting point before "rating_change" means anything beyond warm-up
# noise.
ELO_WARMUP_SEASONS = 4

# project_upcoming_season's fallback for a feature that's missing on the
# upcoming season's row (e.g. preseason_win_total hasn't been sourced yet).
# 0.0 is the right neutral value for every other feature here, since
# they're all deltas centered near zero, but 0.0 for preseason_win_total
# would read as "expected to go 0-17," a strongly negative signal. 8.5,
# half of a 17-game season, is the actual neutral point for a raw win total.
FEATURE_FALLBACK = {col: 0.0 for col in FEATURE_COLS}
FEATURE_FALLBACK["preseason_win_total"] = 8.5


@dataclass
class FittedModel:
    coefficients: dict[str, float]  # keys: "intercept" + FEATURE_COLS
    n_rows: int


def _team_long_format(game_log: pd.DataFrame) -> pd.DataFrame:
    home = game_log[["season", "week", "home_team", "pregame_elo_home", "postgame_elo_home"]].rename(
        columns={"home_team": "team", "pregame_elo_home": "pregame_elo", "postgame_elo_home": "postgame_elo"}
    )
    away = game_log[["season", "week", "away_team", "pregame_elo_away", "postgame_elo_away"]].rename(
        columns={"away_team": "team", "pregame_elo_away": "pregame_elo", "postgame_elo_away": "postgame_elo"}
    )
    return pd.concat([home, away], ignore_index=True)


def compute_season_rating_changes(game_log: pd.DataFrame) -> pd.DataFrame:
    """season, team, start_rating, end_rating, rating_change (end - start)."""
    long = _team_long_format(game_log).sort_values(["team", "season", "week"])
    grouped = long.groupby(["season", "team"])
    result = pd.DataFrame(
        {"start_rating": grouped["pregame_elo"].first(), "end_rating": grouped["postgame_elo"].last()}
    ).reset_index()
    result["rating_change"] = result["end_rating"] - result["start_rating"]
    return result


def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df))] + [df[c].to_numpy(dtype=float) for c in FEATURE_COLS])


def fit_ols(df: pd.DataFrame) -> FittedModel:
    X = _design_matrix(df)
    y = df["rating_change"].to_numpy(dtype=float)
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    return FittedModel(coefficients=dict(zip(["intercept"] + FEATURE_COLS, coefs)), n_rows=len(df))


def predict(model: FittedModel, df: pd.DataFrame) -> np.ndarray:
    beta = np.array([model.coefficients["intercept"]] + [model.coefficients[c] for c in FEATURE_COLS])
    return _design_matrix(df) @ beta


def fit_with_loso_cv(features: pd.DataFrame, game_log: pd.DataFrame) -> tuple[FittedModel, pd.DataFrame]:
    """Merges offseason features with the realized rating-change target, fits
    the full model, and produces leave-one-season-out cross-validated
    predictions for every row. With this few season transitions, evaluating
    a model on data it was trained on would be too easy to fool ourselves
    with. LOSO gives an honest read on how well it generalizes.

    Returns (full_model, df), where df has one row per (season, team) with
    both complete features and a known outcome, plus a
    `loso_predicted_adjustment` column.
    """
    targets = compute_season_rating_changes(game_log)
    merged = features.merge(targets, on=["season", "team"], how="inner")
    df = merged.dropna(subset=FEATURE_COLS + ["rating_change"]).reset_index(drop=True)

    full_model = fit_ols(df)

    loso_predicted = pd.Series(np.nan, index=df.index)
    for season in sorted(df["season"].unique()):
        train = df[df["season"] != season]
        test = df[df["season"] == season]
        if train.empty or test.empty:
            continue
        model = fit_ols(train)
        loso_predicted.loc[test.index] = predict(model, test)
    df = df.assign(loso_predicted_adjustment=loso_predicted)

    return full_model, df


def season_adjustments_from_loso(df: pd.DataFrame) -> dict[tuple[int, str], float]:
    """(season, team) -> LOSO-predicted adjustment, ready to feed into
    ratings.pipeline.run(season_adjustments=...)."""
    return {(row.season, row.team): row.loso_predicted_adjustment for row in df.itertuples(index=False)}


def project_upcoming_season(
    features: pd.DataFrame, full_model: FittedModel, upcoming_season: int
) -> dict[str, float]:
    """team -> projected adjustment for a season with no known outcome yet
    (next season, say), using the model fit on all available history --
    there's nothing to hold out for a season that hasn't happened. Missing
    features (QB continuity before Week 1 starters are set, for instance)
    fall back to each feature's own neutral point; see FEATURE_FALLBACK."""
    upcoming = features[features["season"] == upcoming_season].copy()
    upcoming[FEATURE_COLS] = upcoming[FEATURE_COLS].fillna(FEATURE_FALLBACK)
    upcoming["projected_adjustment"] = predict(full_model, upcoming)
    return dict(zip(upcoming["team"], upcoming["projected_adjustment"]))


@dataclass
class AdjustedEloPipeline:
    baseline_log: pd.DataFrame
    baseline_summary: object  # ratings.pipeline.BacktestSummary, pre-adjustment
    adjusted_log: pd.DataFrame
    adjusted_ratings: pd.DataFrame
    adjusted_summary: object  # ratings.pipeline.BacktestSummary
    features: pd.DataFrame
    full_model: FittedModel
    loso_df: pd.DataFrame
    max_season: int


def fit_adjusted_elo_pipeline(start_season: int, regression_start_season: int | None = None) -> AdjustedEloPipeline:
    """Runs the full Stage 1 -> Stage 1b pipeline in one call: baseline Elo,
    offseason features, the LOSO-fit adjustment model, then an Elo re-run
    with those adjustments applied. Every script that needs "the current
    adjusted ratings" (run_offseason_adjustment.py, the season simulator)
    goes through this, so they can't quietly drift out of sync with each
    other.

    start_season: the production Elo engine's start. Current ratings,
    backtest numbers, and everything Stage 3 consumes are anchored here and
    don't move with regression_start_season.

    regression_start_season: if set earlier than start_season, the
    offseason-adjustment model gets fit on a wider window of season
    transitions (more independent trials to learn from) via a second,
    separate Elo run used only to generate (season, team, rating_change)
    targets (see ratings.pipeline.run). The first ELO_WARMUP_SEASONS of that
    wider run get dropped before fitting, since ratings need time to move
    off the synthetic 1500 start. The resulting season_adjustments dict only
    ever gets looked up for (season, team) keys the production run's season
    boundaries actually hit, so the extra pre-start_season entries in it are
    harmless.
    """
    from nfl_predictor.ratings.offseason_features import build_offseason_features
    from nfl_predictor.ratings.pipeline import run as run_elo

    baseline_log, _, baseline_summary = run_elo(start_season=start_season)
    max_season = int(baseline_log["season"].max())

    regression_start = regression_start_season or start_season
    widened = regression_start_season is not None and regression_start != start_season
    regression_log = run_elo(start_season=regression_start)[0] if widened else baseline_log

    features = build_offseason_features(regression_start, max_season + 1)
    if widened:
        # Only the widened path needs the warm-up seasons dropped --
        # build_offseason_features already excludes the very first row (no
        # prior season to compare against), which is all the non-widened
        # path ever relied on.
        trusted_from = regression_start + ELO_WARMUP_SEASONS
        features = features[features["season"] >= trusted_from].reset_index(drop=True)

    full_model, loso_df = fit_with_loso_cv(features, regression_log)
    season_adjustments = season_adjustments_from_loso(loso_df)

    adjusted_log, adjusted_ratings, adjusted_summary = run_elo(
        start_season=start_season, season_adjustments=season_adjustments
    )

    return AdjustedEloPipeline(
        baseline_log=baseline_log,
        baseline_summary=baseline_summary,
        adjusted_log=adjusted_log,
        adjusted_ratings=adjusted_ratings,
        adjusted_summary=adjusted_summary,
        features=features,
        full_model=full_model,
        loso_df=loso_df,
        max_season=max_season,
    )
