"""Rule-based "why was this wrong" explanations for a missed game
prediction. No modeling or fitting here, no external API calls: just a
handful of factors computed from data the pipeline already fetches (actual
turnovers, actual vs. rolling-average EPA, in-season QB continuity, injury
reports), ranked by how unusual each one was, with the top factor(s)
written up as a short sentence citing the real numbers.

The thresholds below (TURNOVER_MARGIN_THRESHOLD, EPA_COLLAPSE_THRESHOLD,
etc.) are reasonable round numbers, not fit from real outcomes -- there's
no real 2026 upset data yet to fit them against. Worth revisiting once a
season's worth of season_review.csv rows exist to check them against, the
same way every other threshold in this codebase (MIN_DROPBACKS,
MIN_COVERAGE_TARGETS, ...) already got checked against real data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TURNOVER_MARGIN_THRESHOLD = -1  # favored team giving the ball away net >=1 more time than they took it away
EPA_COLLAPSE_THRESHOLD = -0.15  # actual off_epa/play this far below the pregame rolling average is a real off day
KEY_INJURY_COUNT_THRESHOLD = 2  # favored team starters (QB/RB/WR/TE/OL) listed Out or Doubtful that week
CLOSE_GAME_MARGIN = 8  # a one-score game
CLOSE_GAME_PROB_CEILING = 0.65  # favorite wasn't a heavy favorite to begin with

MAX_FACTORS_CITED = 2  # how many ranked factors make it into the explanation sentence


@dataclass
class UpsetContext:
    """Everything explain_upset needs about one incorrectly-predicted game,
    from the losing favorite's point of view. Any field can be None when
    that signal isn't available (e.g. no PFR defensive data, or the pbp
    file for a season hasn't been fetched yet) -- explain_upset treats a
    None field as "that factor doesn't apply" rather than raising."""

    favored_team: str
    underdog_team: str
    favored_win_prob: float  # the frozen pregame prediction's win probability for favored_team
    margin: int  # favored_team's score minus underdog_team's score (negative, since favored_team lost)
    favored_turnover_margin: int | None = None  # favored_team's takeaways minus giveaways in this game
    favored_off_epa_actual: float | None = None  # favored_team's actual offensive EPA/play in this game
    favored_off_epa_rolling: float | None = None  # favored_team's pregame rolling offensive EPA/play
    favored_qb_changed: bool | None = None  # True if favored_team started a different QB than their prior game
    favored_key_injuries_out: int | None = None  # count of favored_team starters listed Out/Doubtful that week


def _missing(value: float | None) -> bool:
    """True for None or NaN. A value pulled from a pandas lookup for a game
    with no data to compute it (e.g. a team's rolling EPA before it's
    played a single game this season) comes back as NaN, not None --
    `x is None` alone lets a NaN quietly through, and NaN compares False to
    everything, so a plain `> threshold` check doesn't catch it either."""
    return value is None or (isinstance(value, float) and math.isnan(value))


def _qb_change_factor(ctx: UpsetContext) -> tuple[float, str] | None:
    if not ctx.favored_qb_changed:
        return None
    return 3.0, (
        f"{ctx.favored_team} started a different quarterback than in their previous game, "
        "which the Elo-based prediction has no way to account for"
    )


def _turnover_factor(ctx: UpsetContext) -> tuple[float, str] | None:
    if ctx.favored_turnover_margin is None or ctx.favored_turnover_margin > TURNOVER_MARGIN_THRESHOLD:
        return None
    severity = float(-ctx.favored_turnover_margin)
    return severity, f"{ctx.favored_team} lost the turnover battle (turnover margin: {ctx.favored_turnover_margin:+d})"


def _epa_collapse_factor(ctx: UpsetContext) -> tuple[float, str] | None:
    if _missing(ctx.favored_off_epa_actual) or _missing(ctx.favored_off_epa_rolling):
        return None
    deviation = ctx.favored_off_epa_actual - ctx.favored_off_epa_rolling
    if deviation > EPA_COLLAPSE_THRESHOLD:
        return None
    severity = -deviation / abs(EPA_COLLAPSE_THRESHOLD)
    return severity, (
        f"{ctx.favored_team}'s offense managed just {ctx.favored_off_epa_actual:.2f} EPA/play, "
        f"well below the {ctx.favored_off_epa_rolling:.2f} they'd been averaging coming in"
    )


def _injury_factor(ctx: UpsetContext) -> tuple[float, str] | None:
    if ctx.favored_key_injuries_out is None or ctx.favored_key_injuries_out < KEY_INJURY_COUNT_THRESHOLD:
        return None
    severity = ctx.favored_key_injuries_out * 0.5
    return severity, (
        f"{ctx.favored_team} had {ctx.favored_key_injuries_out} key starters listed Out or Doubtful that week"
    )


_FACTOR_FUNCS = [_qb_change_factor, _turnover_factor, _epa_collapse_factor, _injury_factor]


def explain_upset(ctx: UpsetContext) -> str:
    """The top MAX_FACTORS_CITED applicable factors, ranked by severity and
    joined into one sentence with the real numbers cited. Falls back to a
    "this was close to a coin flip" framing for a game that was never a
    confident prediction to begin with, and to an honest "nothing stands
    out" note when neither applies -- better than forcing a confident-
    sounding cause onto a game that just went the other way."""
    factors = [f for f in (fn(ctx) for fn in _FACTOR_FUNCS) if f is not None]
    factors.sort(key=lambda f: f[0], reverse=True)

    if factors:
        # Every factor sentence already starts with a (capitalized) team
        # code, so no further capitalizing is needed -- and str.capitalize()
        # would actively hurt here, lowercasing every team code and acronym
        # after the first character of the joined string.
        sentences = [text for _, text in factors[:MAX_FACTORS_CITED]]
        return "; also, ".join(sentences) + "."

    is_close_game = ctx.favored_win_prob < CLOSE_GAME_PROB_CEILING and abs(ctx.margin) <= CLOSE_GAME_MARGIN
    if is_close_game:
        return (
            f"This was close to a toss-up to begin with (favored at only {ctx.favored_win_prob:.0%}), "
            f"and the final margin was just {abs(ctx.margin)} points. Normal game-to-game variance, "
            "not a clear modeling miss."
        )

    turnover_note = (
        f"turnover margin {ctx.favored_turnover_margin:+d}" if ctx.favored_turnover_margin is not None else "turnovers unknown"
    )
    return (
        f"No single factor stands out ({turnover_note}, no QB change, "
        "offensive production in line with recent form): likely just normal variance the model can't fully capture."
    )


def describe_correct(favored_team: str, underdog_team: str, favored_win_prob: float) -> str:
    """The review_note for a correctly-predicted game."""
    return f"Correctly predicted {favored_team} over {underdog_team} ({favored_win_prob:.0%} pregame)."
