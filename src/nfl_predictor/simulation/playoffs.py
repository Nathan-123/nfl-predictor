"""Single-elimination playoff bracket simulation, layered on top of Stage 3's
regular-season machinery: same margin-sampling model, same
elo.update_ratings, same RNG. Runs inside the same per-simulation loop so
each simulated postseason starts from that exact simulation's own final
ratings and seeding.
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
    """One elimination game. No ties allowed (NFL playoffs go to sudden-
    death overtime until decided), so a margin that rounds to exactly 0
    gets resampled. Updates `ratings` in place via the same Elo update used
    everywhere else, and returns the winning team code.

    `neutral` (Super Bowl only): both the fitted margin model's intercept
    and Elo's hfa were fit on real games that almost all had an actual home
    team, so both encode a real home-field scoring edge. For a neutral-site
    game, drop both rather than incorrectly favoring whichever team happens
    to be passed as `home`.
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
    """seeds_by_conference: conference name -> 7 seeds, best to worst (e.g.
    from standings.seed_conference). `ratings` should be that same
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


@dataclass
class BracketGame:
    round: str  # "Wild Card" | "Divisional" | "Conference Championship" | "Super Bowl"
    conference: str | None  # None for the Super Bowl
    home_team: str
    away_team: str
    home_seed: int | None
    away_seed: int | None
    winner: str


# ---- detailed replay: real random draws, but with the game-by-game bracket
# recorded. simulate_playoffs' hot-path version above only returns team sets
# reaching each round, which is enough for the aggregate Monte Carlo counts
# but not enough to print an actual bracket for one chosen sim.


def _simulate_conference_detailed(
    conference: str,
    seeds: list[str],
    ratings: dict[str, float],
    hfa: float,
    k: float,
    margin_intercept: float,
    margin_slope: float,
    margin_std: float,
    rng: np.random.Generator,
    games: list[BracketGame],
) -> str:
    """Same bracket structure and real random margin draws as
    _simulate_conference, but appends a BracketGame record for every game
    played too. See simulate_playoffs_detailed."""
    seed_rank = {team: i + 1 for i, team in enumerate(seeds)}
    args = (ratings, hfa, k, margin_intercept, margin_slope, margin_std, rng)

    def play(round_name: str, home: str, away: str) -> str:
        winner = _play_single_elim_game(home, away, *args)
        games.append(
            BracketGame(
                round=round_name,
                conference=conference,
                home_team=home,
                away_team=away,
                home_seed=seed_rank[home],
                away_seed=seed_rank[away],
                winner=winner,
            )
        )
        return winner

    wc_pairs = [(seeds[1], seeds[6]), (seeds[2], seeds[5]), (seeds[3], seeds[4])]
    wc_winners = [play("Wild Card", hi, lo) for hi, lo in wc_pairs]

    surviving = sorted(wc_winners, key=lambda t: seed_rank[t])
    lowest_survivor = surviving[-1]
    other_two = surviving[:-1]

    div_winner_1 = play("Divisional", seeds[0], lowest_survivor)
    div_winner_2 = play("Divisional", other_two[0], other_two[1])

    finalists = sorted([div_winner_1, div_winner_2], key=lambda t: seed_rank[t])
    return play("Conference Championship", finalists[0], finalists[1])


def simulate_playoffs_detailed(
    seeds_by_conference: dict[str, list[str]],
    ratings: dict[str, float],
    hfa: float,
    k: float,
    margin_intercept: float,
    margin_slope: float,
    margin_std: float,
    rng: np.random.Generator,
) -> tuple[list[BracketGame], str]:
    """The real-randomness counterpart to simulate_playoffs_deterministic.
    Same stochastic margin draws as simulate_playoffs (reuses the same
    tested _play_single_elim_game), but returns the actual game-by-game
    bracket instead of just team sets, for building a human-readable table
    out of one chosen, already-realistic simulation."""
    games: list[BracketGame] = []
    conference_champions = []
    for conference, seeds in seeds_by_conference.items():
        conference_champions.append(
            _simulate_conference_detailed(
                conference, seeds, ratings, hfa, k, margin_intercept, margin_slope, margin_std, rng, games
            )
        )

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
    games.append(
        BracketGame(
            round="Super Bowl",
            conference=None,
            home_team=conference_champions[0],
            away_team=conference_champions[1],
            home_seed=None,
            away_seed=None,
            winner=champion,
        )
    )
    return games, champion


# ---- deterministic ("model's best single guess") bracket, no RNG ---------


def _play_single_elim_game_deterministic(
    home: str,
    away: str,
    ratings: dict[str, float],
    hfa: float,
    k: float,
    margin_intercept: float,
    margin_slope: float,
    neutral: bool = False,
) -> tuple[str, int]:
    """Same no-RNG "model's point estimate wins" rule as
    season.simulate_one_season_deterministic (see that docstring for the
    margin==0 tie-break). Returns (winner, home_score - away_score) for
    display."""
    effective_hfa = 0.0 if neutral else hfa
    effective_intercept = 0.0 if neutral else margin_intercept

    predicted_margin = effective_intercept + margin_slope * (ratings[home] - ratings[away])
    margin = int(round(predicted_margin)) or 1

    home_score, away_score = margin_to_scores(margin)
    ratings[home], ratings[away] = update_ratings(ratings[home], ratings[away], home_score, away_score, effective_hfa, k)
    winner = home if home_score > away_score else away
    return winner, margin


def _simulate_conference_deterministic(
    conference: str,
    seeds: list[str],
    ratings: dict[str, float],
    hfa: float,
    k: float,
    margin_intercept: float,
    margin_slope: float,
    games: list[BracketGame],
) -> str:
    """seeds: 7 team codes, best (1) to worst (7). Appends every game played
    to `games` (in place) and returns the conference champion."""
    seed_rank = {team: i + 1 for i, team in enumerate(seeds)}

    def play(round_name: str, home: str, away: str) -> str:
        winner, _ = _play_single_elim_game_deterministic(home, away, ratings, hfa, k, margin_intercept, margin_slope)
        games.append(
            BracketGame(
                round=round_name,
                conference=conference,
                home_team=home,
                away_team=away,
                home_seed=seed_rank[home],
                away_seed=seed_rank[away],
                winner=winner,
            )
        )
        return winner

    # Wild Card: 2v7, 3v6, 4v5 (better seed hosts); #1 seed has a bye.
    wc_pairs = [(seeds[1], seeds[6]), (seeds[2], seeds[5]), (seeds[3], seeds[4])]
    wc_winners = [play("Wild Card", hi, lo) for hi, lo in wc_pairs]

    # Divisional re-seeding (real NFL rule): #1 seed plays the worst-seeded
    # survivor; the other two survivors play each other, better seed hosting.
    surviving = sorted(wc_winners, key=lambda t: seed_rank[t])
    lowest_survivor = surviving[-1]
    other_two = surviving[:-1]

    div_winner_1 = play("Divisional", seeds[0], lowest_survivor)
    div_winner_2 = play("Divisional", other_two[0], other_two[1])

    finalists = sorted([div_winner_1, div_winner_2], key=lambda t: seed_rank[t])
    return play("Conference Championship", finalists[0], finalists[1])


def simulate_playoffs_deterministic(
    seeds_by_conference: dict[str, list[str]],
    ratings: dict[str, float],
    hfa: float,
    k: float,
    margin_intercept: float,
    margin_slope: float,
) -> tuple[list[BracketGame], str]:
    """The no-RNG counterpart to simulate_playoffs. seeds_by_conference and
    `ratings` should come from season.simulate_one_season_deterministic's
    own output, not a Monte Carlo sim's. Returns every game played
    (including the Super Bowl) and the champion."""
    games: list[BracketGame] = []
    conference_champions = []
    for conference, seeds in seeds_by_conference.items():
        conference_champions.append(
            _simulate_conference_deterministic(conference, seeds, ratings, hfa, k, margin_intercept, margin_slope, games)
        )

    champion, _ = _play_single_elim_game_deterministic(
        conference_champions[0], conference_champions[1], ratings, hfa, k, margin_intercept, margin_slope, neutral=True
    )
    games.append(
        BracketGame(
            round="Super Bowl",
            conference=None,
            home_team=conference_champions[0],
            away_team=conference_champions[1],
            home_seed=None,
            away_seed=None,
            winner=champion,
        )
    )
    return games, champion
