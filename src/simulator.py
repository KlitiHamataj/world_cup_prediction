"""Simulate the full World Cup 2026 from the trained model.

- Group stage: rank teams by expected points (xPts = 3*P(win) + 1*P(draw)).
- Knockout: propagate winners through the official bracket template.
  A knockout tie can't end in a draw, so when DRAW is the most likely outcome
  we resolve it with a coin flip that is *slightly* biased toward the team with
  the higher win probability (controlled by COIN_BIAS).

Matches are played at a neutral venue, EXCEPT when a host nation
(USA, Canada, Mexico) plays in its own country: that team gets home advantage.
"""
from __future__ import annotations
import random
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEAMS_CSV = ROOT / "data" / "raw" / "wc_2026_teams.csv"
FIXTURES_CSV = ROOT / "data" / "raw" / "wc_2026_fixtures.csv"

# 0.0 = pure 50/50 coin flip ; 1.0 = decide entirely by win probability.
# 0.5 keeps it close to a coin flip with a slight edge to the favourite.
COIN_BIAS = 0.5

KO_ORDER = ["Round of 32", "Round of 16", "Quarter-final",
            "Semi-final", "3rd Place Match", "Final"]

# Co-host nations of World Cup 2026. They get home advantage only when the
# match is played in their own country (matched against the fixture's `country`).
HOSTS = {"USA", "Canada", "Mexico"}


def _match_probs(predictor, t1, t2, country=None):
    """Probabilities as {home, draw, away} *relative to (t1, t2)*.

    If one of the teams is a host nation playing in its own country, that team
    is treated as the home side; otherwise the match is neutral.
    """
    if country in HOSTS:
        if country == t1:
            return predictor.predict(t1, t2, neutral=False, is_world_cup=True)
        if country == t2:
            p = predictor.predict(t2, t1, neutral=False, is_world_cup=True)
            return {"home": p["away"], "draw": p["draw"], "away": p["home"]}
    return predictor.predict(t1, t2, neutral=True, is_world_cup=True)


def _outcome_label(p: dict) -> str:
    return {"home": "team1 win", "draw": "draw", "away": "team2 win"}[max(p, key=p.get)]


def simulate_group_stage(predictor, teams: pd.DataFrame, fixtures: pd.DataFrame):
    """Return (standings_by_group, group_matches, best_thirds)."""
    rank_lookup = dict(zip(teams["team"], teams["fifa_rank"]))
    group_of = dict(zip(teams["team"], teams["group"]))

    xpts = {t: 0.0 for t in teams["team"]}
    matches = []

    gs = fixtures[fixtures["stage"] == "Group Stage"]
    for _, m in gs.iterrows():
        t1, t2 = m["team1"], m["team2"]
        p = _match_probs(predictor, t1, t2, country=m["country"])
        xpts[t1] += 3 * p["home"] + 1 * p["draw"]
        xpts[t2] += 3 * p["away"] + 1 * p["draw"]
        matches.append({"group": m["group"], "team1": t1, "team2": t2,
                        "p_home": p["home"], "p_draw": p["draw"], "p_away": p["away"],
                        "outcome": _outcome_label(p)})

    standings = {}
    thirds = []
    for g in sorted(teams["group"].unique()):
        gteams = [t for t in teams["team"] if group_of[t] == g]
        rows = [{"team": t, "xpts": round(xpts[t], 2), "fifa_rank": rank_lookup[t]} for t in gteams]
        df = pd.DataFrame(rows).sort_values(["xpts", "fifa_rank"], ascending=[False, True]).reset_index(drop=True)
        df["position"] = df.index + 1
        df["qualified"] = df["position"] <= 2
        standings[g] = df
        third = df.iloc[2]
        thirds.append({"team": third["team"], "group": g, "xpts": third["xpts"], "fifa_rank": third["fifa_rank"]})

    best_thirds = pd.DataFrame(thirds).sort_values(["xpts", "fifa_rank"], ascending=[False, True]).reset_index(drop=True)
    best_thirds["rank"] = best_thirds.index + 1
    best_thirds["qualified"] = best_thirds["rank"] <= 8
    return standings, matches, best_thirds


def _decide(predictor, t1, t2, rng, country=None):
    """Return (winner, loser, probs, coin_flipped)."""
    p = _match_probs(predictor, t1, t2, country=country)
    top = max(p, key=p.get)
    if top == "draw":
        denom = p["home"] + p["away"]
        cond1 = p["home"] / denom if denom else 0.5          # team1 conditional win share
        p1 = 0.5 + COIN_BIAS * (cond1 - 0.5)                 # nudged toward favourite
        winner, loser = (t1, t2) if rng.random() < p1 else (t2, t1)
        return winner, loser, p, True
    winner, loser = (t1, t2) if p["home"] >= p["away"] else (t2, t1)
    return winner, loser, p, False


def simulate_knockout(predictor, standings, best_thirds, fixtures, seed: int = 42):
    """Return (ko_matches, champion, third_place)."""
    rng = random.Random(seed)

    # Resolve the initial Round-of-32 slots.
    resolve = {}
    for g, df in standings.items():
        resolve[f"1{g}"] = df.iloc[0]["team"]
        resolve[f"2{g}"] = df.iloc[1]["team"]
    qualified_thirds = best_thirds[best_thirds["qualified"]].reset_index(drop=True)
    for i, row in qualified_thirds.iterrows():
        resolve[f"Best 3rd #{i + 1}"] = row["team"]

    ko_matches = []
    champion = None
    third_place = None
    sf_losers = []

    # Output-label counters per stage (match the placeholders used downstream).
    out_prefix = {"Round of 32": "R32 W", "Round of 16": "QF",
                  "Quarter-final": "SF", "Semi-final": "Finalist "}

    for stage in KO_ORDER:
        rows = fixtures[fixtures["stage"] == stage].reset_index(drop=True)

        if stage == "3rd Place Match":
            t1, t2 = sf_losers[0], sf_losers[1]
            country = rows.iloc[0]["country"] if not rows.empty else None
            winner, loser, p, flip = _decide(predictor, t1, t2, rng, country=country)
            third_place = winner
            ko_matches.append({"stage": stage, "slot1": "SF1 loser", "slot2": "SF2 loser",
                               "team1": t1, "team2": t2, "p_home": p["home"], "p_draw": p["draw"],
                               "p_away": p["away"], "winner": winner, "coin_flip": flip})
            continue

        for i, m in rows.iterrows():
            s1, s2 = m["team1"], m["team2"]
            t1, t2 = resolve.get(s1, s1), resolve.get(s2, s2)
            winner, loser, p, flip = _decide(predictor, t1, t2, rng, country=m["country"])

            if stage == "Semi-final":
                sf_losers.append(loser)
            label = f"{out_prefix[stage]}{i + 1}" if stage in out_prefix else None
            if label:
                resolve[label] = winner
            if stage == "Final":
                champion = winner

            ko_matches.append({"stage": stage, "slot1": s1, "slot2": s2,
                               "team1": t1, "team2": t2, "p_home": p["home"], "p_draw": p["draw"],
                               "p_away": p["away"], "winner": winner, "coin_flip": flip})

    return ko_matches, champion, third_place


def load_inputs():
    teams = pd.read_csv(TEAMS_CSV)
    fixtures = pd.read_csv(FIXTURES_CSV)
    return teams, fixtures


def run_full_simulation(predictor, seed: int = 42):
    teams, fixtures = load_inputs()
    standings, group_matches, best_thirds = simulate_group_stage(predictor, teams, fixtures)
    ko_matches, champion, third_place = simulate_knockout(predictor, standings, best_thirds, fixtures, seed=seed)
    return {
        "standings": standings,
        "group_matches": group_matches,
        "best_thirds": best_thirds,
        "ko_matches": ko_matches,
        "champion": champion,
        "third_place": third_place,
    }
