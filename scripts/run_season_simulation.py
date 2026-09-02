#!/usr/bin/env python
"""CLI entrypoint for Stage 3 (regular season) + Stage 4 (playoffs): Monte
Carlo simulation of the upcoming NFL season through to a Super Bowl champion.

Example:
    python scripts/run_season_simulation.py --n-sims 10000
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nfl_predictor.config import DEFAULT_REGRESSION_START_SEASON, DEFAULT_START_SEASON, PROCESSED_DIR
from nfl_predictor.ratings.adjustment import fit_adjusted_elo_pipeline, project_upcoming_season
from nfl_predictor.ratings.elo import DEFAULT_K
from nfl_predictor.simulation.playoffs import simulate_playoffs_detailed
from nfl_predictor.simulation.season import (
    fit_margin_model,
    load_season_schedule,
    load_team_division_conference,
    project_starting_ratings,
    run_simulations,
)

WIN_TOTALS_PATH = PROCESSED_DIR / "season_simulation_win_totals.parquet"
SUMMARY_PATH = PROCESSED_DIR / "season_simulation_summary.parquet"
PROJECTED_RECORD_PATH = PROCESSED_DIR / "projected_final_record.csv"
PROJECTED_BRACKET_PATH = PROCESSED_DIR / "projected_playoff_bracket.csv"
REPRESENTATIVE_GAMES_PATH = PROCESSED_DIR / "representative_season_games.csv"
GAME_PROBABILITIES_PATH = PROCESSED_DIR / "game_win_probabilities.csv"
PLAYOFF_SLOT_PROBABILITIES_PATH = PROCESSED_DIR / "playoff_slot_probabilities.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo simulate the upcoming NFL regular season.")
    parser.add_argument("--start-season", type=int, default=DEFAULT_START_SEASON)
    parser.add_argument("--regression-start-season", type=int, default=DEFAULT_REGRESSION_START_SEASON)
    parser.add_argument("--n-sims", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Fitting Elo (Stage 1) + offseason adjustment (Stage 1b)...")
    pipeline = fit_adjusted_elo_pipeline(args.start_season, regression_start_season=args.regression_start_season)
    upcoming_season = pipeline.max_season + 1

    projected_adjustments = project_upcoming_season(pipeline.features, pipeline.full_model, upcoming_season)
    starting_ratings = project_starting_ratings(pipeline.adjusted_ratings, projected_adjustments)

    margin_intercept, margin_slope, margin_std = fit_margin_model(pipeline.adjusted_log)
    print(
        f"Margin model: intercept={margin_intercept:.2f}, slope={margin_slope:.4f}, "
        f"residual_std={margin_std:.2f}, hfa={pipeline.adjusted_summary.hfa_used:.1f}"
    )

    schedule = load_season_schedule(upcoming_season)
    if schedule.empty:
        raise SystemExit(f"No {upcoming_season} regular-season schedule found in the cache.")
    divisions, conferences = load_team_division_conference()

    print(f"\nRunning {args.n_sims} simulations of the {upcoming_season} season ({len(schedule)} games each)...")
    results = run_simulations(
        n_sims=args.n_sims,
        schedule=schedule,
        starting_ratings=starting_ratings,
        hfa=pipeline.adjusted_summary.hfa_used,
        k=DEFAULT_K,
        margin_intercept=margin_intercept,
        margin_slope=margin_slope,
        margin_std=margin_std,
        divisions=divisions,
        conferences=conferences,
        seed=args.seed,
        keep_regular_season_details=True,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    results.win_totals.to_parquet(WIN_TOTALS_PATH, index=False)
    results.summary.to_parquet(SUMMARY_PATH, index=False)

    print(f"\n{upcoming_season} season simulation summary (sorted by mean projected wins):\n")
    display = results.summary.copy()
    for col in ["mean_wins", "median_wins", "wins_p10", "wins_p90"]:
        display[col] = display[col].round(1)
    pct_cols = [
        "playoff_prob",
        "division_prob",
        "one_seed_prob",
        "won_wildcard_prob",
        "conf_championship_prob",
        "super_bowl_prob",
        "champion_prob",
    ]
    for col in pct_cols:
        display[col] = (display[col] * 100).round(1)
    print(display.to_string(index=False))

    # ---- per-game / per-slot win probabilities, aggregated across all sims
    # The "best possible prediction for every game" output: for each
    # scheduled regular-season game, what fraction of the n_sims runs above
    # had the home team winning that specific matchup, properly accounting
    # for the fact that by, say, Week 10, each sim's teams carry whatever
    # rating drift that sim's own earlier random results produced. Playoff
    # games get aggregated by structural bracket slot instead of team name,
    # since who's even in a given playoff game depends on the regular
    # season; see run_simulations' docstring.
    game_probs_display = results.game_probabilities.copy()
    game_probs_display["home_win_prob"] = (game_probs_display["home_win_prob"] * 100).round(1)
    game_probs_display["avg_margin"] = game_probs_display["avg_margin"].round(1)
    results.game_probabilities.to_csv(GAME_PROBABILITIES_PATH, index=False)

    print(f"\n{upcoming_season} game-by-game win probabilities (every scheduled game, aggregated across all {args.n_sims} sims):\n")
    print(game_probs_display.to_string(index=False))

    slot_probs_display = results.playoff_slot_probabilities.copy()
    slot_probs_display["matchup_occurrence_pct"] = (slot_probs_display["matchup_occurrence_pct"] * 100).round(1)
    slot_probs_display["home_side_win_prob"] = (slot_probs_display["home_side_win_prob"] * 100).round(1)
    results.playoff_slot_probabilities.to_csv(PLAYOFF_SLOT_PROBABILITIES_PATH, index=False)

    print(f"\n{upcoming_season} playoff bracket-slot win probabilities (aggregated across all {args.n_sims} sims):\n")
    print(slot_probs_display.to_string(index=False))

    print(f"\nSaved: {GAME_PROBABILITIES_PATH}")
    print(f"Saved: {PLAYOFF_SLOT_PROBABILITIES_PATH}")

    # ---- one realistic representative simulation: record + bracket --------
    # A complement to the Monte Carlo summary above, but built from a real
    # simulated season (real random variance, so favorites do sometimes
    # lose) rather than a no-randomness "favorite always wins" run, which
    # produces unrealistic blowout records (16-1, 1-16) that don't match the
    # summary's own mean_wins. Picked from the n_sims already run above: the
    # single simulation whose per-team win total is closest (least squared
    # error) to the aggregate summary's median_wins, i.e. the most "typical"
    # of the realistic seasons already drawn, not a fabricated extreme.
    target_wins = results.summary.set_index("team")["median_wins"]
    win_pivot = results.win_totals.pivot(index="sim", columns="team", values="wins")
    squared_error = ((win_pivot - target_wins) ** 2).sum(axis=1)
    representative_sim = int(squared_error.idxmin())
    rep_standings, rep_seeds_by_conference, rep_ratings, rep_game_log = results.regular_season_details[representative_sim]

    # This sim's own real game-by-game results, including real upsets,
    # unlike game_win_probabilities.csv's aggregate "who's favored" view.
    # Joined against that same aggregate table so an "upset" (the model's
    # favorite actually lost, in this specific realistic draw) gets flagged
    # explicitly rather than left for the reader to spot by eye.
    rep_games_df = pd.DataFrame(rep_game_log, columns=["week", "game_id", "home_team", "away_team", "winner", "margin"])
    rep_games_df = rep_games_df.merge(
        results.game_probabilities[["game_id", "home_win_prob", "predicted_winner"]], on="game_id", how="left"
    )
    rep_games_df["upset"] = (rep_games_df["winner"] != "TIE") & (rep_games_df["winner"] != rep_games_df["predicted_winner"])
    rep_games_df = rep_games_df.sort_values(["week", "game_id"]).reset_index(drop=True)
    rep_games_df.to_csv(REPRESENTATIVE_GAMES_PATH, index=False)

    n_upsets = int(rep_games_df["upset"].sum())
    print(
        f"\n{upcoming_season} representative-simulation game log "
        f"({n_upsets} of {len(rep_games_df)} games were upsets vs. the aggregate favorite):\n"
    )
    rep_games_display = rep_games_df.copy()
    rep_games_display["home_win_prob"] = (rep_games_display["home_win_prob"] * 100).round(1)
    rep_games_display["margin"] = rep_games_display["margin"].round(1)
    print(rep_games_display.to_string(index=False))
    print(f"\nSaved: {REPRESENTATIVE_GAMES_PATH}")

    seed_lookup = {t: i + 1 for seeds in rep_seeds_by_conference.values() for i, t in enumerate(seeds)}
    record_rows = [
        {
            "team": team,
            "wins": rec.wins,
            "losses": rec.losses,
            "ties": rec.ties,
            "division": divisions[team],
            "conference": conferences[team],
            "made_playoffs": team in seed_lookup,
            "seed": seed_lookup.get(team),
        }
        for team, rec in rep_standings.items()
    ]
    record_df = (
        pd.DataFrame(record_rows)
        .sort_values(["made_playoffs", "wins"], ascending=[False, False])
        .reset_index(drop=True)
    )
    record_df.to_csv(PROJECTED_RECORD_PATH, index=False)

    print(
        f"\n{upcoming_season} projected final record "
        f"(one representative simulation, #{representative_sim} of {args.n_sims}, "
        "closest to the summary's median win totals):\n"
    )
    print(record_df.to_string(index=False))

    # A fresh random draw for the playoffs, seeded from that same
    # simulation's real final standings/ratings. Not a literal replay of
    # sim #N's own postseason (which wasn't retained, to save memory), but
    # the same stochastic model applied to the same realistic regular season.
    rng_playoff = np.random.default_rng(None if args.seed is None else args.seed + 1)
    bracket_games, champion = simulate_playoffs_detailed(
        seeds_by_conference=rep_seeds_by_conference,
        ratings=dict(rep_ratings),
        hfa=pipeline.adjusted_summary.hfa_used,
        k=DEFAULT_K,
        margin_intercept=margin_intercept,
        margin_slope=margin_slope,
        margin_std=margin_std,
        rng=rng_playoff,
    )
    bracket_df = pd.DataFrame([asdict(g) for g in bracket_games])
    bracket_df.to_csv(PROJECTED_BRACKET_PATH, index=False)

    print(f"\n{upcoming_season} projected playoff bracket (from that same representative simulation):\n")
    print(bracket_df.to_string(index=False))
    print(f"\nProjected Super Bowl champion: {champion}")

    print(f"\nSaved: {PROJECTED_RECORD_PATH}")
    print(f"Saved: {PROJECTED_BRACKET_PATH}")


if __name__ == "__main__":
    main()
