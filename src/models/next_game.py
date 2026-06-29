"""Upcoming-match odds for the dashboard.

Pulls the latest bookmaker 1X2 odds from the database, averages them across all
books, removes the bookmaker margin (de-vig) to get market probabilities, and
pairs each match with the model's own prediction so the dashboard can show a
"market vs model" comparison for the next few games.
"""
from __future__ import annotations

from datetime import date

from src.data_pipeline.db import get_db, queries

# Odds are stored under normalised (training) names; the flag table and the rest
# of the UI use World-Cup display names. Only these two differ.
DISPLAY_NAMES = {"United States": "USA", "Turkey": "Türkiye"}


def _display(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def _devig(odds: dict) -> dict:
    """Decimal odds -> implied probabilities, normalised to sum to 1 (de-vigged)."""
    raw = {k: 1.0 / v for k, v in odds.items() if v}
    total = sum(raw.values())
    if not total:
        return {}
    return {k: v / total for k, v in raw.items()}


def _build_match(rows, predictor) -> dict:
    """Turn all bookmaker rows for one match into a market-vs-model summary."""
    home = rows["home_team"].iloc[0]
    away = rows["away_team"].iloc[0]
    avg = {
        "home": float(rows["home_win"].mean()),
        "draw": float(rows["draw"].mean()),
        "away": float(rows["away_win"].mean()),
    }
    market = _devig(avg)

    model = None
    if predictor is not None:
        try:
            p = predictor.predict(_display(home), _display(away),
                                  neutral=True, is_world_cup=True)
            model = {"home": p["home"], "draw": p["draw"], "away": p["away"]}
        except Exception:
            model = None

    def pct(d):
        return {k: round(v * 100, 1) for k, v in d.items()} if d else None

    edge = None
    if model and market:
        edge = {k: round((model[k] - market[k]) * 100, 1) for k in ("home", "draw", "away")}

    return {
        "date": rows["match_date"].iloc[0],
        "home": _display(home),
        "away": _display(away),
        "n_books": int(rows["source"].nunique()),
        "avg_odds": {k: round(v, 2) for k, v in avg.items()},
        "market": pct(market),
        "model": pct(model),
        "edge": edge,
    }


def get_next_games(predictor=None, limit: int = 6) -> list[dict]:
    """Market-vs-model odds for the next `limit` matches that have odds.

    Returns matches dated today-or-later first; if none are upcoming, falls back
    to the most recent matches we have odds for. Empty list if no odds at all.
    """
    try:
        with get_db() as conn:
            df = queries.latest_odds(conn)
    except Exception:
        return []
    if df is None or df.empty:
        return []

    df = df.dropna(subset=["match_date"])
    if df.empty:
        return []

    today = date.today().isoformat()
    upcoming = df[df["match_date"] >= today]
    have_upcoming = not upcoming.empty
    pool = upcoming if have_upcoming else df

    # One entry per match: soonest-first if upcoming, else most-recent-first.
    keys = (pool[["match_date", "home_team", "away_team"]]
            .drop_duplicates()
            .sort_values("match_date", ascending=have_upcoming))

    games = []
    for _, k in keys.head(limit).iterrows():
        rows = pool[(pool["match_date"] == k["match_date"]) &
                    (pool["home_team"] == k["home_team"]) &
                    (pool["away_team"] == k["away_team"])]
        if not rows.empty:
            games.append(_build_match(rows, predictor))
    return games


def get_next_game(predictor=None) -> dict | None:
    """Single next match (kept for convenience / backward compatibility)."""
    games = get_next_games(predictor, limit=1)
    return games[0] if games else None
