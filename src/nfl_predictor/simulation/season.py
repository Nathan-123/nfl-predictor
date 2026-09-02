"""Monte Carlo season simulation, driven by the Elo+adjustment engine rather
than the Stage 2 GBM: a season sim is recursive (each simulated outcome
feeds the next game's pregame ratings), which is exactly what Elo is built
for, while the GBM's best features (rolling EPA, in-season QB continuity)
only exist for real, already-played games.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nfl_predictor.config import DATA_DIR
from nfl_predictor.ratings.elo import regress_to_mean, update_ratings
from nfl_predictor.simulation.standings import new_standings, record_game, seed_conference
from nfl_predictor.team_codes import canonicalize_teams

# playoffs.py is imported lazily inside run_simulations to avoid a circular
# import (it imports margin_to_scores from this module).


def fit_margin_model(elo_game_log: pd.DataFrame) -> tuple[float, float, float]:
    """OLS home_margin ~ intercept + slope * elo_diff on real history, plus
    the residual std. Same approach as gamemodel.model's Elo-margin
    baseline, refit here on the full dataset rather than per walk-forward
    fold, since this drives random scoreline sampling rather than a
    backtest."""
    elo_diff = (elo_game_log["pregame_elo_home"] - elo_game_log["pregame_elo_away"]).to_numpy(dtype=float)
    margin = (elo_game_log["home_score"] - elo_game_log["away_score"]).to_numpy(dtype=float)
    A = np.vstack([np.ones_like(elo_diff), elo_diff]).T
    intercept, slope = np.linalg.lstsq(A, margin, rcond=None)[0]
    residual_std = float((margin - A @ [intercept, slope]).std())
    return float(intercept), float(slope), residual_std


def project_starting_ratings(
    current_ratings: pd.DataFrame, adjustments: dict[str, float], regress_frac: float = 1 / 3
) -> dict[str, float]:
    """Applies the same season-boundary treatment ratings.pipeline.run()
    uses internally (mean reversion plus the fitted offseason adjustment) to
    get each team's rating at the start of the upcoming season."""
    return {
        row.team: regress_to_mean(row.elo_rating) + adjustments.get(row.team, 0.0)
        for row in current_ratings.itertuples(index=False)
    }


def load_team_division_conference() -> tuple[dict[str, str], dict[str, str]]:
    """team -> division, team -> conference, restricted to the 32 current
    team codes. schedules.parquet spans back to 2007 and, before
    canonicalizing, carries old codes for relocated franchises (OAK/SD/STL);
    team_desc.parquet carries those same old codes too."""
    schedules = pd.read_parquet(DATA_DIR / "schedules.parquet")
    schedules = canonicalize_teams(schedules, ["home_team", "away_team"])
    current_teams = set(schedules["home_team"].unique()) | set(schedules["away_team"].unique())
    team_desc = pd.read_parquet(DATA_DIR / "team_desc.parquet")
    team_desc = team_desc[team_desc["team_abbr"].isin(current_teams)].drop_duplicates("team_abbr")
    divisions = dict(zip(team_desc["team_abbr"], team_desc["team_division"]))
    conferences = dict(zip(team_desc["team_abbr"], team_desc["team_conf"]))
    return divisions, conferences


def load_season_schedule(season: int) -> pd.DataFrame:
    schedules = pd.read_parquet(DATA_DIR / "schedules.parquet")
    season_games = schedules[(schedules["season"] == season) & (schedules["game_type"] == "REG")]
    return season_games.sort_values(["week", "game_id"]).reset_index(drop=True)


def margin_to_scores(margin: int) -> tuple[int, int]:
    """Converts a signed point margin into a synthetic (home_score,
    away_score) pair. elo.update_ratings only cares about the sign (who won)
    and magnitude (for the MOV multiplier), not the actual score level, so
    this is just the simplest pair with the right difference."""
    return max(margin, 0), max(-margin, 0)


def simulate_one_season(
    schedule: pd.DataFrame,
    starting_ratings: dict[str, float],
    hfa: float,
    k: float,
    margin_intercept: float,
    margin_slope: float,
    margin_std: float,
    divisions: dict[str, str],
    conferences: dict[str, str],
    rng: np.random.Generator,
    game_tally: dict[str, list[float]] | None = None,
    game_log: list[tuple] | None = None,
):
    """game_tally: optional accumulator (mutated in place, not reset here).
    Pass a dict shared across many calls to build up per-game win/margin
    statistics without retaining every individual game result. Keyed by
    game_id -> [home_win_credit_sum, margin_sum, n_sims] (a tie contributes
    0.5 credit, matching how standings.py scores a tie).

    game_log: optional list (mutated in place) that instead records this
    call's own individual game results, one (week, game_id, home_team,
    away_team, winner, margin) tuple per game (winner is "TIE" for a tie,
    never None). Not meant to be shared across calls like game_tally is;
    pass a fresh list each time to recover one specific simulated season's
    results afterward (see run_simulations' regular_season_details)."""
    ratings = dict(starting_ratings)
    standings = new_standings(list(ratings.keys()))

    for game in schedule.itertuples(index=False):
        home, away = game.home_team, game.away_team
        predicted_margin = margin_intercept + margin_slope * (ratings[home] - ratings[away])
        margin = int(round(rng.normal(predicted_margin, margin_std)))
        if margin == 0:
            # Real NFL games almost never end in a genuine tie (about 0.28%
            # of 2021+ games by our own count), but naively rounding a
            # continuous normal draw puts about 3% of its mass at exactly 0
            # for an even matchup, roughly 10x too often, since the model
            # has no separate notion of "went to overtime." Cheap fix:
            # redraw once, as if that were the overtime period, and only
            # call it a tie if that also lands on 0.
            margin = int(round(rng.normal(predicted_margin, margin_std)))
        home_score, away_score = margin_to_scores(margin)

        if game_tally is not None:
            entry = game_tally.setdefault(game.game_id, [0.0, 0.0, 0])
            win_credit = 1.0 if home_score > away_score else 0.5 if home_score == away_score else 0.0
            entry[0] += win_credit
            entry[1] += margin
            entry[2] += 1

        if game_log is not None:
            winner = "TIE" if home_score == away_score else home if home_score > away_score else away
            game_log.append((game.week, game.game_id, home, away, winner, margin))

        record_game(standings, home, away, home_score, away_score, divisions, conferences)
        ratings[home], ratings[away] = update_ratings(ratings[home], ratings[away], home_score, away_score, hfa, k)

    return standings, ratings


def simulate_one_season_deterministic(
    schedule: pd.DataFrame,
    starting_ratings: dict[str, float],
    hfa: float,
    k: float,
    margin_intercept: float,
    margin_slope: float,
    divisions: dict[str, str],
    conferences: dict[str, str],
):
    """One single, non-random pass through the schedule: each game's winner
    is whichever team the fitted margin model favors, with no draw from the
    residual distribution. Answers "what does the model expect to happen"
    rather than run_simulations' "how likely is each outcome," giving one
    coherent win-loss record and (fed into simulate_playoffs_deterministic)
    one coherent bracket, instead of an aggregate over many random draws.

    A predicted_margin that rounds to exactly 0 (only possible from a near-
    exact rating tie) is broken toward the home team, consistent with
    margin_intercept already encoding a real average home-field scoring
    edge from the same historical fit."""
    ratings = dict(starting_ratings)
    standings = new_standings(list(ratings.keys()))

    for game in schedule.itertuples(index=False):
        home, away = game.home_team, game.away_team
        predicted_margin = margin_intercept + margin_slope * (ratings[home] - ratings[away])
        margin = int(round(predicted_margin)) or 1
        home_score, away_score = margin_to_scores(margin)

        record_game(standings, home, away, home_score, away_score, divisions, conferences)
        ratings[home], ratings[away] = update_ratings(ratings[home], ratings[away], home_score, away_score, hfa, k)

    return standings, ratings


@dataclass
class SimulationResults:
    win_totals: pd.DataFrame  # one row per (sim, team): wins (ties count as 0.5)
    summary: pd.DataFrame  # one row per team: aggregated probabilities
    # Present only when run_simulations(keep_regular_season_details=True):
    # one (standings, seeds_by_conference, final_ratings, game_log) tuple
    # per sim, in sim order, so a caller can pick one specific, already-
    # simulated season afterward (see run_season_simulation.py's
    # "representative simulation") without re-running anything. game_log is
    # that sim's own real, upset-inclusive results, unlike
    # game_probabilities' aggregate "who's favored" view below.
    regular_season_details: list[tuple] | None = None
    # One row per scheduled game (not per sim): home_win_prob/avg_margin
    # aggregated across all n_sims, i.e. what fraction of realistic seasons
    # had the home team winning that specific matchup.
    game_probabilities: pd.DataFrame | None = None
    # One row per playoff bracket slot (e.g. "AFC Wild Card, 2-seed vs
    # 7-seed") aggregated across all n_sims; see run_simulations' docstring
    # for why slots, not team names, are the stable unit here.
    playoff_slot_probabilities: pd.DataFrame | None = None


def run_simulations(
    n_sims: int,
    schedule: pd.DataFrame,
    starting_ratings: dict[str, float],
    hfa: float,
    k: float,
    margin_intercept: float,
    margin_slope: float,
    margin_std: float,
    divisions: dict[str, str],
    conferences: dict[str, str],
    seed: int | None = None,
    keep_regular_season_details: bool = False,
) -> SimulationResults:
    """Runs n_sims full seasons plus playoffs and returns the aggregate
    results (see SimulationResults). Also builds two per-game/per-slot
    prediction tables from the same n_sims runs: regular-season games have a
    fixed, known matchup every sim (Week 5 KC @ DEN is always Week 5 KC @
    DEN), so they're aggregated by game_id directly. Playoff games don't,
    since which two teams meet in, say, the AFC Wild Card round depends on
    how the regular season went in that specific sim. Those get aggregated
    by structural bracket slot instead (conference + round + position, e.g.
    "the #2 seed's Wild Card game"), which every sim fills exactly once
    regardless of who's in it."""
    from nfl_predictor.simulation.playoffs import simulate_playoffs_detailed

    rng = np.random.default_rng(seed)
    teams = list(starting_ratings.keys())
    conference_names = sorted(set(conferences.values()))

    win_rows = []
    regular_season_details = [] if keep_regular_season_details else None
    game_tally: dict[str, list[float]] = {}
    playoff_slot_tally: dict[tuple, dict] = {}
    playoff_counts = {t: 0 for t in teams}
    division_counts = {t: 0 for t in teams}
    one_seed_counts = {t: 0 for t in teams}
    won_wildcard_counts = {t: 0 for t in teams}
    conf_championship_counts = {t: 0 for t in teams}
    super_bowl_counts = {t: 0 for t in teams}
    champion_counts = {t: 0 for t in teams}

    for sim in range(n_sims):
        # Only the one sim eventually picked as "representative" (see
        # scripts/run_season_simulation.py) actually needs its game log kept,
        # but which one that'll be isn't known until every sim has run. So a
        # fresh per-sim list is captured here whenever regular_season_details
        # is being kept at all, and the script just discards the ones it
        # doesn't end up using.
        sim_game_log = [] if regular_season_details is not None else None
        standings, final_ratings = simulate_one_season(
            schedule,
            starting_ratings,
            hfa,
            k,
            margin_intercept,
            margin_slope,
            margin_std,
            divisions,
            conferences,
            rng,
            game_tally=game_tally,
            game_log=sim_game_log,
        )
        for t in teams:
            rec = standings[t]
            win_rows.append({"sim": sim, "team": t, "wins": rec.wins + 0.5 * rec.ties})

        seeds_by_conference = {}
        for conf in conference_names:
            conf_teams = [t for t in teams if conferences[t] == conf]
            seeds = seed_conference(conf_teams, divisions, standings)
            seeds_by_conference[conf] = seeds
            for t in seeds:
                playoff_counts[t] += 1
            for t in seeds[:4]:
                division_counts[t] += 1
            one_seed_counts[seeds[0]] += 1

        if regular_season_details is not None:
            regular_season_details.append((standings, seeds_by_conference, dict(final_ratings), sim_game_log))

        games, champion = simulate_playoffs_detailed(
            seeds_by_conference, final_ratings, hfa, k, margin_intercept, margin_slope, margin_std, rng
        )
        _tally_playoff_slots(games, playoff_slot_tally)

        divisional_teams = {g.home_team for g in games if g.round == "Divisional"} | {
            g.away_team for g in games if g.round == "Divisional"
        }
        conference_championship_teams = {g.home_team for g in games if g.round == "Conference Championship"} | {
            g.away_team for g in games if g.round == "Conference Championship"
        }
        super_bowl_teams = {games[-1].home_team, games[-1].away_team}  # simulate_playoffs_detailed appends SB last
        for t in divisional_teams:
            won_wildcard_counts[t] += 1
        for t in conference_championship_teams:
            conf_championship_counts[t] += 1
        for t in super_bowl_teams:
            super_bowl_counts[t] += 1
        champion_counts[champion] += 1

    win_totals = pd.DataFrame(win_rows)
    grouped = win_totals.groupby("team")["wins"]
    summary = pd.DataFrame(
        {
            "team": grouped.mean().index,
            "mean_wins": grouped.mean().to_numpy(),
            "median_wins": grouped.median().to_numpy(),
            "wins_p10": grouped.quantile(0.10).to_numpy(),
            "wins_p90": grouped.quantile(0.90).to_numpy(),
            "playoff_prob": [playoff_counts[t] / n_sims for t in grouped.mean().index],
            "division_prob": [division_counts[t] / n_sims for t in grouped.mean().index],
            "one_seed_prob": [one_seed_counts[t] / n_sims for t in grouped.mean().index],
            "won_wildcard_prob": [won_wildcard_counts[t] / n_sims for t in grouped.mean().index],
            "conf_championship_prob": [conf_championship_counts[t] / n_sims for t in grouped.mean().index],
            "super_bowl_prob": [super_bowl_counts[t] / n_sims for t in grouped.mean().index],
            "champion_prob": [champion_counts[t] / n_sims for t in grouped.mean().index],
        }
    ).sort_values("mean_wins", ascending=False).reset_index(drop=True)

    schedule_lookup = schedule.set_index("game_id")[["week", "home_team", "away_team"]]
    game_prob_rows = []
    for game_id, (win_credit, margin_sum, n) in game_tally.items():
        row = schedule_lookup.loc[game_id]
        home_win_prob = win_credit / n
        game_prob_rows.append(
            {
                "week": row["week"],
                "game_id": game_id,
                "home_team": row["home_team"],
                "away_team": row["away_team"],
                "home_win_prob": home_win_prob,
                "predicted_winner": row["home_team"] if home_win_prob >= 0.5 else row["away_team"],
                "avg_margin": margin_sum / n,
            }
        )
    game_probabilities = (
        pd.DataFrame(game_prob_rows).sort_values(["week", "game_id"]).reset_index(drop=True)
    )

    playoff_slot_rows = []
    for (conf, round_name, slot_index), entry in playoff_slot_tally.items():
        n = entry["n"]
        (home_team, away_team), occurrence_count = entry["team_pairs"].most_common(1)[0]
        (home_seed, away_seed), _ = entry["seed_pairs"].most_common(1)[0]
        matchup = (
            f"{home_team} ({int(home_seed)}) vs {away_team} ({int(away_seed)})"
            if home_seed is not None
            else f"{home_team} vs {away_team}"
        )
        playoff_slot_rows.append(
            {
                "conference": conf,
                "round": round_name,
                "slot": slot_index,
                "typical_matchup": matchup,
                "matchup_occurrence_pct": occurrence_count / n,
                "home_side_win_prob": entry["home_win"] / n,
                "n_sims": n,
            }
        )
    playoff_slot_probabilities = (
        pd.DataFrame(playoff_slot_rows).sort_values(["conference", "round", "slot"]).reset_index(drop=True)
        if playoff_slot_rows
        else pd.DataFrame(
            columns=["conference", "round", "slot", "typical_matchup", "matchup_occurrence_pct", "home_side_win_prob", "n_sims"]
        )
    )

    return SimulationResults(
        win_totals=win_totals,
        summary=summary,
        regular_season_details=regular_season_details,
        game_probabilities=game_probabilities,
        playoff_slot_probabilities=playoff_slot_probabilities,
    )


def _tally_playoff_slots(games: list, tally: dict[tuple, dict]) -> None:
    """Groups one bracket's `games` (from simulate_playoffs_detailed) by
    structural slot (conference, round, and position within that round's
    fixed play order) and accumulates win/team-identity stats into `tally`
    (mutated in place, shared and called once per sim). See run_simulations'
    docstring for why slots, not team names, are the stable cross-sim unit
    here."""
    by_conference: dict[str, list] = {}
    for g in games:
        if g.round != "Super Bowl":
            by_conference.setdefault(g.conference, []).append(g)

    for conf, conf_games in by_conference.items():
        for slot_index, g in enumerate(conf_games):
            _tally_one_playoff_game((conf, g.round, slot_index), g, tally)

    sb_game = games[-1]  # simulate_playoffs_detailed always appends the Super Bowl last
    _tally_one_playoff_game(("Super Bowl", "Super Bowl", 0), sb_game, tally)


def _tally_one_playoff_game(key: tuple, g, tally: dict[tuple, dict]) -> None:
    entry = tally.setdefault(key, {"home_win": 0.0, "n": 0, "team_pairs": Counter(), "seed_pairs": Counter()})
    entry["home_win"] += 1.0 if g.winner == g.home_team else 0.0
    entry["n"] += 1
    entry["team_pairs"][(g.home_team, g.away_team)] += 1
    entry["seed_pairs"][(g.home_seed, g.away_seed)] += 1
