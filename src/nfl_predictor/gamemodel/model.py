"""XGBoost game outcome model: a classifier for home win probability and a
regressor for point margin, both trained on the Stage 2 feature table and
backtested with an expanding walk-forward-by-season split (it never trains
on a season it's then evaluated on), same spirit as Stage 1b's LOSO CV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from nfl_predictor.gamemodel.features import FEATURE_COLS
from nfl_predictor.metrics import brier_score, log_loss, mae

# Deliberately shallow and regularized: with roughly 600-950 training rows
# per walk-forward fold and 14 features, XGBoost's defaults overfit badly
# (tried it, Brier came out worse than plain Elo). Picked by comparing
# average walk-forward Brier across a few regularization levels, not
# exhaustively tuned, just enough to stop the model fitting noise.
CLASSIFIER_PARAMS = dict(
    n_estimators=60, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
    eval_metric="logloss",
)
REGRESSOR_PARAMS = dict(
    n_estimators=60, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, reg_lambda=5,
)


@dataclass
class BacktestReport:
    n_games: int
    test_seasons: list[int]
    gbm_brier: float
    gbm_log_loss: float
    elo_brier: float
    elo_log_loss: float
    market_brier: float
    market_log_loss: float
    baseline_brier: float
    baseline_log_loss: float
    gbm_margin_mae: float
    elo_margin_mae: float
    market_margin_mae: float


def _fit_elo_margin_baseline(train: pd.DataFrame) -> tuple[float, float]:
    """OLS home_margin ~ elo_diff on the training fold. A simple, refit-
    every-fold baseline for "what would plain Elo predict the margin to be",
    to compare the GBM's margin predictions against."""
    x = train["elo_diff"].to_numpy(dtype=float)
    y = train["home_margin"].to_numpy(dtype=float)
    A = np.vstack([np.ones_like(x), x]).T
    intercept, slope = np.linalg.lstsq(A, y, rcond=None)[0]
    return intercept, slope


def walk_forward_backtest(features: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window backtest: for each season (from the 3rd available
    onward), train on all strictly-earlier seasons and predict that season.
    Returns one row per test-set game with gbm_win_prob and gbm_margin
    alongside the existing elo/market/actual columns."""
    df = features.dropna(subset=FEATURE_COLS + ["actual_home_win", "home_margin"]).reset_index(drop=True)
    seasons = sorted(df["season"].unique())
    test_seasons = seasons[2:]  # first two seasons are train-only warmup

    predictions = []
    for test_season in test_seasons:
        train = df[df["season"] < test_season]
        test = df[df["season"] == test_season]
        if train.empty or test.empty:
            continue

        # XGBClassifier needs discrete class labels, so the handful of
        # historical ties (actual_home_win == 0.5) get excluded from
        # training only. They're still predicted on and still scored
        # (brier/log_loss handle a 0.5 target fine), so they aren't
        # silently dropped from the backtest.
        train_clf = train[train["actual_home_win"] != 0.5]
        clf = XGBClassifier(**CLASSIFIER_PARAMS)
        clf.fit(train_clf[FEATURE_COLS], train_clf["actual_home_win"])
        win_prob = clf.predict_proba(test[FEATURE_COLS])[:, 1]

        reg = XGBRegressor(**REGRESSOR_PARAMS)
        reg.fit(train[FEATURE_COLS], train["home_margin"])
        margin = reg.predict(test[FEATURE_COLS])

        intercept, slope = _fit_elo_margin_baseline(train)
        elo_margin = intercept + slope * test["elo_diff"].to_numpy(dtype=float)

        predictions.append(
            test.assign(gbm_win_prob=win_prob, gbm_margin=margin, elo_margin_baseline=elo_margin)
        )

    return pd.concat(predictions, ignore_index=True) if predictions else df.iloc[0:0]


def summarize_backtest(predictions: pd.DataFrame) -> BacktestReport:
    actual_win = predictions["actual_home_win"].to_numpy()
    baseline = np.full_like(actual_win, 0.5, dtype=float)
    actual_margin = predictions["home_margin"].to_numpy(dtype=float)

    return BacktestReport(
        n_games=len(predictions),
        test_seasons=sorted(predictions["season"].unique().tolist()),
        gbm_brier=brier_score(predictions["gbm_win_prob"].to_numpy(), actual_win),
        gbm_log_loss=log_loss(predictions["gbm_win_prob"].to_numpy(), actual_win),
        elo_brier=brier_score(predictions["elo_pred_home_win_prob"].to_numpy(), actual_win),
        elo_log_loss=log_loss(predictions["elo_pred_home_win_prob"].to_numpy(), actual_win),
        market_brier=brier_score(predictions["market_home_win_prob"].to_numpy(), actual_win),
        market_log_loss=log_loss(predictions["market_home_win_prob"].to_numpy(), actual_win),
        baseline_brier=brier_score(baseline, actual_win),
        baseline_log_loss=log_loss(baseline, actual_win),
        gbm_margin_mae=mae(predictions["gbm_margin"].to_numpy(), actual_margin),
        elo_margin_mae=mae(predictions["elo_margin_baseline"].to_numpy(), actual_margin),
        # nflverse's spread_line convention has positive meaning the home
        # team is favored, so it's already the market's implied home margin.
        market_margin_mae=mae(predictions["spread_line"].to_numpy(dtype=float), actual_margin),
    )


def fit_feature_importance(features: pd.DataFrame) -> pd.Series:
    """Fit a single classifier on all available data (not part of the
    backtest) purely to report which features it leans on."""
    df = features.dropna(subset=FEATURE_COLS + ["actual_home_win"])
    df = df[df["actual_home_win"] != 0.5]
    clf = XGBClassifier(**CLASSIFIER_PARAMS)
    clf.fit(df[FEATURE_COLS], df["actual_home_win"])
    return pd.Series(clf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
