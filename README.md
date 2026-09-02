# nfl-predictor

Predicts NFL games, full seasons, and the playoffs using public data. Team
ratings, an offseason layer that turns roster/coaching changes into rating
shifts, a game outcome model, and a Monte Carlo simulator that plays out an
entire season (and the postseason) thousands of times.

Every weight in this pipeline is fit from real historical outcomes, not
picked by hand, and every stage gets backtested against a coin-flip
baseline and the betting market before I trust it.

## Contents

- [How it's built](#how-its-built)
- [Setup](#setup)
- [Running the pipeline end to end](#running-the-pipeline-end-to-end)
- [Stage by stage](#stage-by-stage)
- [Output files](#output-files)
- [Data sources](#data-sources)
- [Tests](#tests)
- [Project layout](#project-layout)
- [Known limitations](#known-limitations)

## How it's built

Four stages, each building on the last, plus a reporting layer on top.

**1. Elo ratings** (`ratings/elo.py`, `ratings/pipeline.py`). A standard
FiveThirtyEight-style Elo engine. Every played game updates both teams'
ratings by a margin-of-victory-scaled amount, with mean reversion applied at
each season boundary.

**2. Offseason adjustment** (`ratings/offseason_features.py`,
`ratings/adjustment.py`). Mean reversion alone treats every team the same
going into a new season, which throws away information we actually have in
the offseason: did the coach change, is the new QB better or worse than the
old one, how much draft capital came in, how did the skill-position/
special-teams/defensive rooms turn over, what does the market's own
preseason win total say. This stage fits how many Elo points each of those
signals is worth, using plain OLS on real season-to-season history,
validated with leave-one-season-out cross-validation, rather than guessing
at the numbers.

**3. Game outcome model** (`gamemodel/features.py`, `gamemodel/model.py`).
An XGBoost classifier and regressor trained on rolling team EPA, in-season
QB continuity, rest/weather context, and the Elo rating gap. Backtested
with an expanding walk-forward split so it never trains on a season it's
then evaluated on. This one's a standalone comparison point; it needs
already-played games' rolling stats, so it doesn't feed into the season
simulator.

**4. Season and playoff simulation** (`simulation/season.py`,
`simulation/playoffs.py`, `simulation/standings.py`). Elo, not the GBM,
drives the season sim, because a season is recursive: each simulated result
changes both teams' ratings before the next game gets played, which is
exactly what Elo is built for. The GBM's best features only exist for real,
already-played games. This stage runs the full schedule thousands of times
with real NFL tiebreak rules and division/wild-card seeding, then simulates
the resulting playoff bracket (with real re-seeding, not a fixed bracket)
for every one of those seasons.

**5. Reporting** (`simulation/reporting.py`,
`scripts/build_prediction_report.py`). Turns the raw simulation output into
something readable: a win probability for every scheduled game, a win
probability for every playoff bracket slot, and one full realistic season
(upsets included) pulled out of the simulation pool as a concrete story.

## Setup

Requires Python 3.11+.

```bash
git clone https://github.com/Nathan-123/nfl-predictor.git
cd nfl-predictor
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

No package install step needed. Every script under `scripts/` adds `src/`
to `sys.path` itself, so `python scripts/<name>.py` works as long as the
path to the script is right.

`data/raw/` (fetched source data) and `data/processed/` (model output) are
both gitignored and empty on a fresh clone. The first pipeline run below
populates `data/raw/`. The one exception is
`data/manual/preseason_win_totals.csv`, which is small, hand-curated, and
committed to the repo since it can't be regenerated automatically.

## Running the pipeline end to end

Run these in order from the repo root. Each stage reads the previous
stage's cached output, so skipping a step (or running an early one with a
narrower season range than a later one needs) shows up as a missing-file or
empty-data error.

```bash
# 1. Pull and cache the raw datasets (schedules, rosters, play-by-play, etc.)
#    Fetches every dataset back to 2007, including full play-by-play, so the
#    first run can take a while depending on connection speed. Cached to
#    data/raw/ afterward; re-running only refetches what you ask it to.
python scripts/run_pipeline.py --start-season 2007

# 2. Fit and backtest the Elo engine
python scripts/run_elo.py --start-season 2021

# 3. Fit the offseason adjustment layer on top of Elo, using a wider
#    historical window (more independent season-transitions to fit on)
python scripts/run_offseason_adjustment.py --start-season 2021 --regression-start-season 2007

# 4. (Optional, standalone) fit and backtest the XGBoost game model
python scripts/run_game_model.py --start-season 2021

# 5. Simulate the upcoming season 10,000 times, including the playoffs
python scripts/run_season_simulation.py --n-sims 10000

# 6. Build the combined, readable prediction report
python scripts/build_prediction_report.py
```

Step 1 fetches back to 2007 for most datasets (a few, like `snap_counts`,
only go back as far as their upstream source allows; see
`config.MIN_SEASON`). That wider window is what lets step 3's
`--regression-start-season 2007` fit the offseason-adjustment model on
around 15 independent season-transitions instead of the ~4 a 2021-only
window would give it.

Step 5 is the one you'll re-run most, since roster moves, injuries, and the
market's own win totals shift throughout the offseason. It re-derives
everything from steps 2 and 3 internally, so there's no need to re-run
those first unless the underlying cached data changed.

## Stage by stage

### 1. Elo ratings

```bash
python scripts/run_elo.py --start-season 2021 [--k 20] [--hfa <points>] [--regress 0.4]
```

Replays every played game from `--start-season` onward in order, updating
both teams' ratings after each one. `--hfa` defaults to an empirical
estimate from the same data's home win rate instead of a fixed constant.
`--regress` controls how much of each team's rating gets pulled back toward
the 1500 league mean at each season boundary; the default (0.4) came from
sweeping this fraction against the real backtest rather than using the
textbook 1/3.

Prints a backtest comparison (Brier score and log loss) against a naive
50/50 baseline and, where moneyline odds exist, the betting market.

### 2. Offseason adjustment

```bash
python scripts/run_offseason_adjustment.py --start-season 2021 --regression-start-season 2007
```

Builds seven preseason features per team-season (coaching change, QB value
delta, draft capital added, skill-position value delta, special-teams value
delta, defense value delta, and the market's preseason win total) and fits
how many Elo points each is worth against real end-of-season outcomes.
Prints the fitted coefficients, a backtest comparison against plain Elo and
the market, and a projected adjustment for every team heading into the next
season.

The QB, skill-position, and defensive value features are all built the same
way: each rostered player gets valued by their own prior-season production,
era-normalized with a same-season z-score since league-wide efficiency has
drifted a lot across eras. That way a team picks up a departing player's
value moving to their new team and a newly-rostered player's value moving
in. A player with no track record falls back to a replacement-level value
(the 25th percentile of that season's population) instead of zero, and
draft picks get valued through a pick-value curve fit on mature draft
classes.

### 3. Game outcome model

```bash
python scripts/run_game_model.py --start-season 2021
```

A standalone comparison model: an XGBoost classifier (win probability) and
regressor (point margin) trained on rolling team EPA, in-season QB
continuity, rest days, and weather, plus the Elo rating gap. Backtested
with an expanding walk-forward split (train on every strictly earlier
season, predict the next one) against Elo, the market, and the 50/50
baseline. This doesn't feed into the season simulator, since its best
features only exist for games that have already been played, which a
season simulation can't provide about its own hypothetical future games.

### 4. Season and playoff simulation

```bash
python scripts/run_season_simulation.py --n-sims 10000 [--seed <int>]
```

Refits Elo plus the offseason adjustment internally, projects every team's
starting rating for the upcoming season, then runs `--n-sims` full 17-game
seasons (real NFL tiebreak logic, real division/wild-card seeding) and a
playoff bracket (with real divisional-round re-seeding, not a fixed
bracket) for each one. `--seed` makes a run fully reproducible. Left
unset, the aggregate probabilities barely move between runs since 10,000
samples is plenty for the law of large numbers to kick in, but the specific
"representative season" story described below will be a different random
draw each time.

It produces four things. A per-team summary: mean/median wins, playoff
odds, division odds, and odds at each playoff round through Super Bowl
champion. Game-by-game win probabilities for every scheduled regular-season
game, aggregated across every simulation, which accounts for a team's
rating by, say, Week 10 depending on how that specific simulation's earlier
weeks went. Playoff bracket-slot win probabilities: since which two teams
meet in a given playoff round depends on how the regular season went, these
get aggregated by structural bracket position (the AFC's #2 seed's Wild
Card game, say) instead of by team name, since every simulation fills that
slot exactly once regardless of who ends up in it. And one representative
simulation, picked from the pool as the single simulated season whose win
totals land closest to the aggregate summary's median, i.e. the most
statistically typical real, upset-inclusive season already drawn. An
earlier version of this instead ran one artificial best-guess season where
the favorite always won every game; that produced unrealistic blowout
records, since removing all game-to-game variance does that.

### 5. Prediction report

```bash
python scripts/build_prediction_report.py
```

Reads the representative simulation's own outputs
(`representative_season_games.csv`, `projected_playoff_bracket.csv`,
`projected_final_record.csv`) and combines them into one CSV with three
sections: every regular-season game's real result (with the model's
pregame confidence and an explicit upset flag), the real playoff bracket,
and the final record. All three sections come from the exact same simulated
season, so they can't disagree with each other the way independently
picking each game's favorite would.

## Output files

Everything lands in `data/processed/`.

| File | Produced by | What it is |
|---|---|---|
| `elo_current_ratings.parquet`, `elo_game_log.parquet` | `run_elo.py` | Plain Elo ratings and full game-by-game log |
| `elo_adjusted_current_ratings.parquet`, `elo_adjusted_game_log.parquet` | `run_offseason_adjustment.py` | Same, with the offseason adjustment applied |
| `game_model_predictions.parquet` | `run_game_model.py` | Walk-forward backtest predictions from the GBM |
| `season_simulation_summary.csv/.parquet` | `run_season_simulation.py` | Per-team win totals and playoff odds across all simulations |
| `season_simulation_win_totals.csv/.parquet` | `run_season_simulation.py` | Raw per-(simulation, team) win totals behind the summary |
| `game_win_probabilities.csv` | `run_season_simulation.py` | Aggregate win probability for every scheduled game |
| `playoff_slot_probabilities.csv` | `run_season_simulation.py` | Aggregate win probability for every playoff bracket slot |
| `representative_season_games.csv` | `run_season_simulation.py` | The representative simulation's real week-by-week results |
| `projected_final_record.csv` | `run_season_simulation.py` | The representative simulation's real final standings |
| `projected_playoff_bracket.csv` | `run_season_simulation.py` | The representative simulation's real playoff bracket |
| `prediction_report.csv` | `build_prediction_report.py` | All three representative-simulation outputs combined into one file |

The `.csv` copies exist alongside `.parquet` so they can be opened directly
in a text editor or spreadsheet app. A `.parquet` file isn't readable
without a script.

## Data sources

[nfl_data_py](https://github.com/nflverse/nfl_data_py) is the main one:
schedules, play-by-play, rosters, injuries, depth charts, draft picks,
combine results, snap counts, and a cross-site player ID table. All free,
no API key.

Pro Football Reference, pulled through `nfl_data_py.import_seasonal_pfr`,
supplies advanced defensive stats (pressures, coverage, opponent passer
rating allowed) that nflverse doesn't have. It gets joined against rosters
through a `pfr_id` crosswalk built in two passes: the roster data's own
`pfr_id` column first, then the secondary cross-site ID table as a fallback
for whatever's missing.

Historical NFL preseason win totals came from covers.com's
sportsoddshistory archive for 2007-2020 and 2022-2025 (2021 and 2026 came
from other public sources), backfilled by hand into
`data/manual/preseason_win_totals.csv` since no free, complete,
machine-readable source of this exists. Public pages, no login required,
and covers.com's robots.txt allows it.

`config.py` centralizes every dataset's earliest available season and any
season-range quirks. `team_codes.py` reconciles the inconsistent team
abbreviations different sources use for relocated franchises (Rams,
Chargers, Raiders) and historical short-code variants.

## Tests

```bash
pytest tests/ -q
```

Most of these are pure unit tests on synthetic data (mock rosters,
play-by-play, schedules) and need no network access. A handful are
integration tests that read the real cached `data/raw/` files, so they need
step 1 above to have been run at least once. `test_pipeline.py` is the one
real network test in the suite: a smoke test that hits nfl_data_py's hosted
files directly, redirected to a throwaway temp directory for the duration
of the module so it never touches the real cache. It's included by default
in the command above, so leave it out explicitly if you're offline:

```bash
pytest tests/ -q --ignore=tests/test_pipeline.py
```

## Project layout

```
src/nfl_predictor/
  config.py      - paths, dataset season ranges, shared constants
  team_codes.py  - cross-source team abbreviation reconciliation
  metrics.py     - brier_score, log_loss, mae

  data/
    fetch.py     - one thin wrapper per nfl_data_py dataset
    pipeline.py  - fetch + cache orchestration, manifest tracking

  ratings/
    elo.py                  - pure Elo math, no I/O
    pipeline.py             - replays real history, backtests Elo
    offseason_features.py   - the 7 preseason signal features
    adjustment.py           - fits/applies the offseason adjustment model

  gamemodel/
    features.py  - Stage 2 feature table (rolling EPA, etc.)
    model.py     - XGBoost fit + walk-forward backtest

  simulation/
    standings.py  - tiebreak logic, playoff seeding
    season.py     - Monte Carlo regular-season simulation
    playoffs.py   - single-elimination bracket simulation
    reporting.py  - combines representative-sim outputs into a report

scripts/  - one CLI entrypoint per stage (see "Stage by stage" above)
tests/    - pytest suite, one file per module above

data/
  manual/     - small hand-curated files (tracked in git)
  raw/        - fetched/cached datasets (gitignored)
  processed/  - model output (gitignored)
```

## Known limitations

A few things worth knowing before trusting any single number out of this
pipeline too literally.

The offseason-adjustment fit is working with a small sample. Even with the
widened 2007+ window there are only about 15 independent
season-transitions to fit 7 features on, so the direction of a backtest
improvement is informative but the exact magnitude isn't something to
lean on hard.

The playoff crosswalk and defensive stats have real gaps. Not every
rostered defender has a resolvable PFR ID, and PFR's own defensive tables
only go back to 2018.

Tie modeling is an approximation, not a real overtime simulation. A drawn
game gets one extra "overtime" redraw rather than a dedicated overtime
model, tuned to land close to the real historical tie rate rather than
modeling the actual OT rules.

The GBM and the season simulator aren't the same model. The GBM is a
useful standalone accuracy comparison, but it isn't what actually drives
the season simulation's game-by-game outcomes. Elo is, for the recursive-
rating reason described above.
