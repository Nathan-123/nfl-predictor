"""Shared scoring functions for backtesting predictions against outcomes."""

from __future__ import annotations

import numpy as np


def brier_score(preds: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean((preds - actuals) ** 2))


def log_loss(preds: np.ndarray, actuals: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(preds, eps, 1 - eps)
    return float(-np.mean(actuals * np.log(p) + (1 - actuals) * np.log(1 - p)))


def mae(preds: np.ndarray, actuals: np.ndarray) -> float:
    return float(np.mean(np.abs(preds - actuals)))
