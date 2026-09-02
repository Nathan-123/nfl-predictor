"""Standings bookkeeping and playoff seeding for one simulated season.

A note on the tiebreak logic: the real NFL tiebreaker procedure is a long
sequential list (head-to-head, division record, conference record, common
games, strength of victory, strength of schedule, net points/touchdowns in
various scopes, then a coin toss), applied iteratively to whatever subset of
teams remains tied at each step. This implementation simplifies that into
one composite sort key per team (win_pct, head-to-head-among-the-tied-group
pct, division pct, conference pct, point differential, team code) and sorts
by it directly. It's a single-pass approximation of the real sequential
process: it matches the official procedure in the large majority of cases
but can diverge in edge cases. Good enough to seed a Monte Carlo simulation,
not a substitute for the actual rulebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TeamRecord:
    team: str
    wins: int = 0
    losses: int = 0
    ties: int = 0
    div_wins: int = 0
    div_losses: int = 0
    div_ties: int = 0
    conf_wins: int = 0
    conf_losses: int = 0
    conf_ties: int = 0
    points_for: int = 0
    points_against: int = 0
    h2h: dict[str, list[int]] = field(default_factory=dict)  # opponent -> [wins, losses, ties]


def new_standings(teams: list[str]) -> dict[str, TeamRecord]:
    return {t: TeamRecord(team=t) for t in teams}


def _pct(wins: int, losses: int, ties: int) -> float:
    games = wins + losses + ties
    return (wins + 0.5 * ties) / games if games else 0.0


def record_game(
    standings: dict[str, TeamRecord],
    home: str,
    away: str,
    home_score: int,
    away_score: int,
    divisions: dict[str, str],
    conferences: dict[str, str],
) -> None:
    home_rec, away_rec = standings[home], standings[away]
    home_rec.points_for += home_score
    home_rec.points_against += away_score
    away_rec.points_for += away_score
    away_rec.points_against += home_score

    if home_score > away_score:
        home_result, away_result = "W", "L"
    elif home_score < away_score:
        home_result, away_result = "L", "W"
    else:
        home_result = away_result = "T"

    same_div = divisions[home] == divisions[away]
    same_conf = conferences[home] == conferences[away]

    for rec, result, opponent in [(home_rec, home_result, away), (away_rec, away_result, home)]:
        if result == "W":
            rec.wins += 1
        elif result == "L":
            rec.losses += 1
        else:
            rec.ties += 1
        if same_div:
            if result == "W":
                rec.div_wins += 1
            elif result == "L":
                rec.div_losses += 1
            else:
                rec.div_ties += 1
        if same_conf:
            if result == "W":
                rec.conf_wins += 1
            elif result == "L":
                rec.conf_losses += 1
            else:
                rec.conf_ties += 1
        h2h = rec.h2h.setdefault(opponent, [0, 0, 0])
        if result == "W":
            h2h[0] += 1
        elif result == "L":
            h2h[1] += 1
        else:
            h2h[2] += 1


def _sort_keys(teams: list[str], standings: dict[str, TeamRecord]) -> dict[str, tuple]:
    win_pct = {t: _pct(standings[t].wins, standings[t].losses, standings[t].ties) for t in teams}
    groups: dict[float, list[str]] = {}
    for t in teams:
        groups.setdefault(win_pct[t], []).append(t)

    keys = {}
    for group in groups.values():
        for t in group:
            rec = standings[t]
            if len(group) > 1:
                w = l = ti = 0
                for opp in group:
                    if opp == t:
                        continue
                    ow, ol, ot = rec.h2h.get(opp, [0, 0, 0])
                    w, l, ti = w + ow, l + ol, ti + ot
                h2h_pct = _pct(w, l, ti)
            else:
                h2h_pct = 0.0
            div_pct = _pct(rec.div_wins, rec.div_losses, rec.div_ties)
            conf_pct = _pct(rec.conf_wins, rec.conf_losses, rec.conf_ties)
            point_diff = rec.points_for - rec.points_against
            keys[t] = (win_pct[t], h2h_pct, div_pct, conf_pct, point_diff, t)
    return keys


def rank_teams(teams: list[str], standings: dict[str, TeamRecord]) -> list[str]:
    """Best team first, per the composite tiebreak key described above."""
    keys = _sort_keys(teams, standings)
    return sorted(teams, key=lambda t: keys[t], reverse=True)


def seed_conference(
    conference_teams: list[str], divisions: dict[str, str], standings: dict[str, TeamRecord]
) -> list[str]:
    """7 seeds for one conference, best first: the 4 division winners
    (ranked among themselves), then the best 3 non-division-winners."""
    by_division: dict[str, list[str]] = {}
    for t in conference_teams:
        by_division.setdefault(divisions[t], []).append(t)

    division_winners, wildcard_pool = [], []
    for div_teams in by_division.values():
        ranked = rank_teams(div_teams, standings)
        division_winners.append(ranked[0])
        wildcard_pool.extend(ranked[1:])

    seeded_winners = rank_teams(division_winners, standings)
    wildcards = rank_teams(wildcard_pool, standings)[:3]
    return seeded_winners + wildcards
