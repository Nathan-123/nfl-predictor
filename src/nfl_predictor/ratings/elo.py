"""Pure Elo rating math -- no I/O, no pandas. One game at a time.

Follows the well-established FiveThirtyEight NFL Elo methodology: a logistic
win-probability model, updated after each game by a K-factor scaled by a
margin-of-victory multiplier that dampens blowouts (an autocorrelated 50-0
win shouldn't move ratings 5x more than a 10-0 win).
"""

from __future__ import annotations

import math

DEFAULT_MEAN = 1500.0
DEFAULT_K = 20.0


def expected_home_win_prob(elo_home: float, elo_away: float, hfa: float) -> float:
    """P(home team wins) under the logistic Elo model, given a home-field-
    advantage bonus (in Elo points) added to the home team's rating."""
    elo_diff = (elo_home + hfa) - elo_away
    return 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))


def mov_multiplier(margin: float, elo_diff: float) -> float:
    """Margin-of-victory dampener: log-scales the point margin, then divides
    out most of the boost a big margin gives to an already-big favorite.
    `elo_diff` is the *pre-game* home-minus-away difference (including HFA)
    signed from the winner's perspective; `margin` is the winner's margin
    (always >= 0)."""
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
    # Winner-perspective diff for the MOV formula: how big a favorite was the
    # actual winner going in (a big win as the underdog swings more than a
    # big win as the already-heavy favorite).
    winner_diff = elo_diff_signed if margin >= 0 else -elo_diff_signed
    shift = k * mov_multiplier(margin, winner_diff) * (actual - expected)

    return elo_home + shift, elo_away - shift


def regress_to_mean(rating: float, mean: float = DEFAULT_MEAN, regress_frac: float = 1 / 3) -> float:
    """Season-carryover mean reversion: pull `rating` a fraction of the way
    back toward the league mean. regress_frac=1/3 means "lose a third of
    your distance from average" -- the standard Elo off-season treatment."""
    return rating - regress_frac * (rating - mean)


def hfa_from_home_win_rate(home_win_rate: float) -> float:
    """Back out an Elo home-field-advantage constant (in points) that would
    reproduce the observed home win rate for two otherwise-equal teams."""
    home_win_rate = min(max(home_win_rate, 1e-6), 1 - 1e-6)  # keep logit finite
    return 400 * math.log10(home_win_rate / (1 - home_win_rate))
