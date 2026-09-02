"""Pure Elo rating math, no I/O or pandas involved. Follows the standard
FiveThirtyEight-style NFL Elo approach: a logistic win probability model,
updated after each game with a K-factor scaled by a margin-of-victory
multiplier so blowouts don't move ratings linearly with the score.
"""

from __future__ import annotations

import math

DEFAULT_MEAN = 1500.0
DEFAULT_K = 20.0


def expected_home_win_prob(elo_home: float, elo_away: float, hfa: float) -> float:
    """Win probability for the home team under the logistic Elo model, given
    a home-field-advantage bonus (in Elo points)."""
    elo_diff = (elo_home + hfa) - elo_away
    return 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))


def mov_multiplier(margin: float, elo_diff: float) -> float:
    """Margin-of-victory dampener. Log-scales the point margin, then discounts
    it based on how big a favorite the winner already was pre-game (elo_diff,
    signed from the winner's perspective), so a blowout by a heavy favorite
    doesn't move ratings as much as the same margin from an underdog."""
    return math.log(abs(margin) + 1) * (2.2 / (0.001 * abs(elo_diff) + 2.2))


def update_ratings(
    elo_home: float,
    elo_away: float,
    home_score: float,
    away_score: float,
    hfa: float,
    k: float = DEFAULT_K,
) -> tuple[float, float]:
    """One game's rating update. Returns (new_elo_home, new_elo_away)."""
    expected = expected_home_win_prob(elo_home, elo_away, hfa)
    actual = 1.0 if home_score > away_score else 0.0 if home_score < away_score else 0.5

    margin = home_score - away_score
    elo_diff_signed = (elo_home + hfa) - elo_away
    # Flip to the winner's perspective for the MOV formula.
    winner_diff = elo_diff_signed if margin >= 0 else -elo_diff_signed
    shift = k * mov_multiplier(margin, winner_diff) * (actual - expected)

    return elo_home + shift, elo_away - shift


def regress_to_mean(rating: float, mean: float = DEFAULT_MEAN, regress_frac: float = 0.4) -> float:
    """Pulls `rating` a fraction of the way back toward the league mean
    between seasons. Default is 0.4 rather than the textbook 1/3; a sweep
    against the 2021+ backtest showed 0.4-0.5 edging out 1/3 (0.2247/0.2245
    vs 0.2249 Brier). Went with 0.4 rather than the best single point (0.5)
    since that gap is small enough to be one sweep's noise, not a real
    optimum."""
    return rating - regress_frac * (rating - mean)


def hfa_from_home_win_rate(home_win_rate: float) -> float:
    """Back out the Elo home-field-advantage constant that reproduces a
    given home win rate between two otherwise equal teams."""
    home_win_rate = min(max(home_win_rate, 1e-6), 1 - 1e-6)  # keep the logit finite
    return 400 * math.log10(home_win_rate / (1 - home_win_rate))
