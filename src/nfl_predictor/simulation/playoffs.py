"""Single-elimination playoff bracket simulation, layered on top of Stage 3's
regular-season machinery: same margin-sampling model, same elo.update_ratings,
same RNG -- run inside the same per-simulation loop so each simulated
postseason starts from that exact simulation's own final ratings and seeding.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nfl_predictor.ratings.elo import update_ratings
from nfl_predictor.simulation.season import margin_to_scores


def _play_single_elim_game(
    home: str,
    away: str,
    ratings: dict[str, float],
    hfa: float,
    k: float,
    margin_intercept: float,
    margin_slope: float,
    margin_std: float,
    rng: np.random.Generator,
    neutral: bool = False,
) -> str:
    """One elimination game -- no ties (NFL playoffs go to sudden-death
    overtime until decided), so a margin that rounds to exactly 0 is
    resampled. Updates `ratings` in place via the same Elo update used
    everywhere else, and returns the winning team code.

    `neutral` (Super Bowl only): both the fitted margin model's intercept
    and Elo's hfa were fit on real games that virtually all had an actual
    home team, so both encode a real home-field scoring edge -- for a
    neutral-site game, drop both rather than incorrectly favoring whichever
    team is passed as `home`.
    """
    effective_hfa = 0.0 if neutral else hfa
    effective_intercept = 0.0 if neutral else margin_intercept

    margin = 0
    while margin == 0:
        predicted_margin = effective_intercept + margin_slope * (ratings[home] - ratings[away])
        margin = int(round(rng.normal(predicted_margin, margin_std)))

    home_score, away_score = margin_to_scores(margin)
    ratings[home], ratings[away] = update_ratings(
        ratings[home], ratings[away], home_score, away_score, effective_hfa, k
    )
    return home if home_score > away_score else away


@dataclass
class PlayoffResult:
    divisional_teams: set[str]  # reached the divisional round (bye team + 3 wild-card winners, per conference)
    conference_championship_teams: set[str]  # reached the conference championship game
    super_bowl_teams: set[str]  # reached the Super Bowl (the 2 conference champions)
    champion: str


def _simulate_conference(
    seeds: list[str],
    ratings: dict[str, float],
    hfa: float,
    k: float,
    margin_intercept: float,
    margin_slope: float,
    margin_std: float,
    rng: np.random.Generator,
) -> tuple[str, set[str], set[str]]:
    """seeds: 7 team codes, best (1) to worst (7). Returns (conference
    champion, divisional-round teams, conference-championship-game teams)."""
    seed_rank = {team: i + 1 for i, team in enumerate(seeds)}
    args = (ratings, hfa, k, margin_intercept, margin_slope, margin_std, rng)

    # Wild Card: 2v7, 3v6, 4v5 (better seed hosts); #1 seed has a bye.
    wc_pairs = [(seeds[1], seeds[6]), (seeds[2], seeds[5]), (seeds[3], seeds[4])]
    wc_winners = [_play_single_elim_game(hi, lo, *args) for hi, lo in wc_pairs]

    divisional_teams = set(wc_winners) | {seeds[0]}

    # Divisional re-seeding (real NFL rule, not a fixed bracket): the #1 seed
    # plays whichever survivor has the worst remaining seed; the other two
    # survivors play each other, better remaining seed hosting.
    surviving = sorted(wc_winners, key=lambda t: seed_rank[t])
    lowest_survivor = surviving[-1]
    other_two = surviving[:-1]

    div_winner_1 = _play_single_elim_game(seeds[0], lowest_survivor, *args)
    div_winner_2 = _play_single_elim_game(other_two[0], other_two[1], *args)

    conf_championship_teams = {div_winner_1, div_winner_2}
    finalists = sorted(conf_championship_teams, key=lambda t: seed_rank[t])
    conference_champion = _play_single_elim_game(finalists[0], finalists[1], *args)

    return conference_champion, divisional_teams, conf_championship_teams


def simulate_playoffs(
    seeds_by_conference: dict[str, list[str]],
    ratings: dict[str, float],
    hfa: float,
    k: float,
    margin_intercept: float,
    margin_slope: float,
    margin_std: float,
    rng: np.random.Generator,
) -> PlayoffResult:
    """seeds_by_conference: conference name -> 7 seeds (best to worst), e.g.
    from standings.seed_conference. `ratings` should be that same
    simulation's final regular-season ratings, not the season-starting ones."""
    divisional_teams: set[str] = set()
    conference_championship_teams: set[str] = set()
    conference_champions = []

    for seeds in seeds_by_conference.values():
        champion, div_teams, champ_game_teams = _simulate_conference(
            seeds, ratings, hfa, k, margin_intercept, margin_slope, margin_std, rng
        )
        divisional_teams |= div_teams
        conference_championship_teams |= champ_game_teams
        conference_champions.append(champion)

    super_bowl_teams = set(conference_champions)
    champion = _play_single_elim_game(
        conference_champions[0],
        conference_champions[1],
        ratings,
        hfa,
        k,
        margin_intercept,
        margin_slope,
        margin_std,
        rng,
        neutral=True,
    )

    return PlayoffResult(
        divisional_teams=divisional_teams,
        conference_championship_teams=conference_championship_teams,
        super_bowl_teams=super_bowl_teams,
        champion=champion,
    )
