"""
Feature engineering

Computes ML ready features from raw match data. For Amelie and Mark to call
build_training_features() and get a clean DataFrame ready for model training.

FEATURES

Result column = "HOME" / "DRAW" / "AWAY".
All features are computed using only data BEFORE that match

Rankings & Strength:
  home_ranking          — FIFA ranking of home team (lower = better)
  away_ranking          — FIFA ranking of away team
  ranking_diff          — away rank minus home rank (positive = home team ranked higher)
  home_elo              — Elo rating of home team
  away_elo              — Elo rating of away team
  elo_diff              — home Elo minus away Elo (positive = home team stronger)

Squad Data:
  home_squad_age        — average age of home team squad
  away_squad_age        — average age of away team squad
  home_market_value     — total squad market value in EUR for home team
  away_market_value     — total squad market value in EUR for away team
  market_value_ratio    — home market value / away market value (>1 = home squad worth more, <1 = away squad worth more)

Rolling Form (last 5 matches):
  home_form_win_rate    — win rate (0.0 to 1.0)
  home_form_pts_rate    — points earned / max possible points (0.0 to 1.0, draws count)
  home_form_gf          — total goals scored in last 5 matches
  home_form_ga          — total goals conceded in last 5 matches
  home_form_gd          — goal difference (gf minus ga)
  away_form_win_rate    — same as above for away team
  away_form_pts_rate    — same as above for away team
  away_form_gf          — same as above for away team
  away_form_ga          — same as above for away team
  away_form_gd          — same as above for away team

Head-to-Head (last 10 meetings):
  h2h_matches           — number of previous meetings between the two teams
  h2h_home_wins         — how many times the home team won
  h2h_away_wins         — how many times the away team won
  h2h_draws             — how many times they drew

Context:
  home_days_rest        — days since home team's last match (max 180)
  away_days_rest        — days since away team's last match (max 180)
  neutral               — 1 if neutral venue, 0 if home advantage applies
  is_world_cup          — 1 if the match is a World Cup match, 0 otherwise
"""

import pandas as pd
import numpy as np
from collections import deque, defaultdict
from src.db import get_db


def _outcome(home_score, away_score) -> str:
    if home_score > away_score:
        return "HOME"
    elif home_score < away_score:
        return "AWAY"
    return "DRAW"


# ---------------------------------------------------------------------------
# Build the full features for training
# ---------------------------------------------------------------------------

def build_training_features(form_window: int = 5, h2h_window: int = 10) -> pd.DataFrame:
    """
    Build a feature DataFrame from the historical database.
    Each row = one match, with features computed from data BEFORE that match.

    Processes all matches chronologically so we never re-scan the full history.
    """
    with get_db() as conn:
        hist = pd.read_sql_query(
            "SELECT date, home_team, away_team, home_score, away_score, "
            "tournament, neutral FROM matches_historical "
            "WHERE home_score IS NOT NULL ORDER BY date",
            conn,
        )
        teams = pd.read_sql_query(
            "SELECT name, fifa_ranking, elo_rating, squad_avg_age, market_value FROM teams",
            conn,
        )

    if hist.empty:
        raise ValueError("No historical data in database. Run ingest_historical first.")

    # Static lookups
    rank_map = dict(zip(teams["name"], teams["fifa_ranking"]))
    elo_map = dict(zip(teams["name"], teams["elo_rating"]))
    age_map = dict(zip(teams["name"], teams["squad_avg_age"]))
    value_map = dict(zip(teams["name"], teams["market_value"]))

    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)

    # Each entry: (date, goals_for, goals_against)
    team_form = defaultdict(lambda: deque(maxlen=form_window))

    # Last match date per team (for rest days)
    team_last_date = {}

    # H2H state
    h2h_history = defaultdict(lambda: deque(maxlen=h2h_window))

    cutoff = pd.Timestamp("2010-01-01")    # Change this date for data previous to 2010
    rows = []

    for idx, match in hist.iterrows():
        home = match["home_team"]
        away = match["away_team"]
        hs = int(match["home_score"])
        as_ = int(match["away_score"])
        match_date = match["date"]

        # Only build features for training window, but update state for ALL matches
        if match_date >= cutoff:
            # --- Compute features from current state (BEFORE this match) ---

            home_recent = list(team_form[home])
            away_recent = list(team_form[away])

            row = {
                "date": match_date.strftime("%Y-%m-%d"),
                "home_team": home,
                "away_team": away,
                "tournament": match["tournament"],
                "neutral": int(match.get("neutral", 0)),
                "result": _outcome(hs, as_),

                # Rankings & Elo
                "home_ranking": rank_map.get(home),
                "away_ranking": rank_map.get(away),
                "ranking_diff": (rank_map.get(away, 100) - rank_map.get(home, 100)),
                "home_elo": elo_map.get(home),
                "away_elo": elo_map.get(away),
                "elo_diff": (elo_map.get(home, 1500) - elo_map.get(away, 1500)),

                # Squad data
                "home_squad_age": age_map.get(home),
                "away_squad_age": age_map.get(away),
                "home_market_value": value_map.get(home),
                "away_market_value": value_map.get(away),
                "market_value_ratio": (
                    (value_map.get(home, 0) / value_map.get(away, 1))
                    if value_map.get(away) else None
                ),

                # Form
                **_form_features(home_recent, "home"),
                **_form_features(away_recent, "away"),

                # H2H
                **_h2h_features(h2h_history[frozenset({home, away})], home, away),

                # Rest days
                "home_days_rest": min((match_date - team_last_date[home]).days, 180)
                    if home in team_last_date else 180,
                "away_days_rest": min((match_date - team_last_date[away]).days, 180)
                    if away in team_last_date else 180,
            }
            rows.append(row)

        #  Update state AFTER extracting features

        # Update form buffers
        team_form[home].append((match_date, hs, as_))   # (date, gf, ga)
        team_form[away].append((match_date, as_, hs))    # flipped for away

        # Update last match date
        team_last_date[home] = match_date
        team_last_date[away] = match_date

        # Update H2H
        h2h_history[frozenset({home, away})].append((home, hs, as_))

    features_df = pd.DataFrame(rows)

    # Tournament type encoding
    wc_keywords = ["world cup", "fifa"]
    features_df["is_world_cup"] = features_df["tournament"].fillna("").str.lower().apply(
        lambda t: int(any(kw in t for kw in wc_keywords))
    )

    print(f"Built {len(features_df):,} training samples with {features_df.shape[1]} features")
    return features_df


def _form_features(recent: list, prefix: str) -> dict:
    """Compute form features from a teams recent match buffer."""
    if not recent:
        return {
            f"{prefix}_form_win_rate": 0.0,
            f"{prefix}_form_pts_rate": 0.0,
            f"{prefix}_form_gf": 0,
            f"{prefix}_form_ga": 0,
            f"{prefix}_form_gd": 0,
        }

    wins, draws, gf, ga = 0, 0, 0, 0
    for _, goals_for, goals_against in recent:
        gf += goals_for
        ga += goals_against
        if goals_for > goals_against:
            wins += 1
        elif goals_for == goals_against:
            draws += 1

    n = len(recent)
    points = wins * 3 + draws
    return {
        f"{prefix}_form_win_rate": wins / n,
        f"{prefix}_form_pts_rate": points / (n * 3),
        f"{prefix}_form_gf": gf,
        f"{prefix}_form_ga": ga,
        f"{prefix}_form_gd": gf - ga,
    }


def _h2h_features(h2h_list: deque, home: str, away: str) -> dict:
    """Compute head-to-head features from the H2H buffer."""
    if not h2h_list:
        return {"h2h_matches": 0, "h2h_home_wins": 0, "h2h_away_wins": 0, "h2h_draws": 0}

    home_wins, away_wins, draws = 0, 0, 0
    for team_a, score_a, score_b in h2h_list:
        # team_a is always the home team of that particular match
        # we need to figure out who scored what relative to current home/away
        if team_a == home:
            hg, ag = score_a, score_b
        else:
            hg, ag = score_b, score_a

        if hg > ag:
            home_wins += 1
        elif ag > hg:
            away_wins += 1
        else:
            draws += 1

    return {
        "h2h_matches": len(h2h_list),
        "h2h_home_wins": home_wins,
        "h2h_away_wins": away_wins,
        "h2h_draws": draws,
    }


# ---------------------------------------------------------------------------
# Single match prediction features
# ---------------------------------------------------------------------------

def build_prediction_features(home_team: str, away_team: str,
                              match_date: str, neutral: int = 1,
                              form_window: int = 5) -> dict:
    """
    Build features for a single upcoming match.
    Same feature set as training, but computed from the latest available data.
    """
    with get_db() as conn:
        teams = pd.read_sql_query(
            "SELECT name, fifa_ranking, elo_rating, squad_avg_age, market_value FROM teams",
            conn,
        )

        # Get last N matches for each team (much faster than loading all history)
        home_recent = pd.read_sql_query(
            "SELECT date, home_team, away_team, home_score, away_score FROM ("
            "  SELECT date, home_team, away_team, home_score, away_score FROM matches_historical"
            "  WHERE (home_team = ? OR away_team = ?) AND home_score IS NOT NULL"
            "  UNION ALL"
            "  SELECT date, home_team, away_team, home_score, away_score FROM matches_wc2026"
            "  WHERE (home_team = ? OR away_team = ?) AND status = 'FINISHED'"
            ") ORDER BY date DESC LIMIT ?",
            conn, params=(home_team, home_team, home_team, home_team, form_window),
        )

        away_recent = pd.read_sql_query(
            "SELECT date, home_team, away_team, home_score, away_score FROM ("
            "  SELECT date, home_team, away_team, home_score, away_score FROM matches_historical"
            "  WHERE (home_team = ? OR away_team = ?) AND home_score IS NOT NULL"
            "  UNION ALL"
            "  SELECT date, home_team, away_team, home_score, away_score FROM matches_wc2026"
            "  WHERE (home_team = ? OR away_team = ?) AND status = 'FINISHED'"
            ") ORDER BY date DESC LIMIT ?",
            conn, params=(away_team, away_team, away_team, away_team, form_window),
        )

        h2h = pd.read_sql_query(
            "SELECT home_team, home_score, away_score FROM matches_historical "
            "WHERE (home_team = ? AND away_team = ?) OR (home_team = ? AND away_team = ?) "
            "ORDER BY date DESC LIMIT 10",
            conn, params=(home_team, away_team, away_team, home_team),
        )

    rank_map = dict(zip(teams["name"], teams["fifa_ranking"]))
    elo_map = dict(zip(teams["name"], teams["elo_rating"]))
    age_map = dict(zip(teams["name"], teams["squad_avg_age"]))
    value_map = dict(zip(teams["name"], teams["market_value"]))

    # Build form buffers from query results
    home_buf = []
    for _, r in home_recent.iterrows():
        if r["home_team"] == home_team:
            home_buf.append((r["date"], int(r["home_score"]), int(r["away_score"])))
        else:
            home_buf.append((r["date"], int(r["away_score"]), int(r["home_score"])))

    away_buf = []
    for _, r in away_recent.iterrows():
        if r["home_team"] == away_team:
            away_buf.append((r["date"], int(r["home_score"]), int(r["away_score"])))
        else:
            away_buf.append((r["date"], int(r["away_score"]), int(r["home_score"])))

    # H2H buffer
    h2h_buf = deque()
    for _, r in h2h.iterrows():
        h2h_buf.append((r["home_team"], int(r["home_score"]), int(r["away_score"])))

    # Rest days
    ref = pd.to_datetime(match_date)
    home_rest = 180
    if not home_recent.empty:
        last = pd.to_datetime(home_recent.iloc[0]["date"])
        home_rest = min((ref - last).days, 180)
    away_rest = 180
    if not away_recent.empty:
        last = pd.to_datetime(away_recent.iloc[0]["date"])
        away_rest = min((ref - last).days, 180)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "date": match_date,
        "neutral": neutral,

        "home_ranking": rank_map.get(home_team),
        "away_ranking": rank_map.get(away_team),
        "ranking_diff": (rank_map.get(away_team, 100) - rank_map.get(home_team, 100)),
        "home_elo": elo_map.get(home_team),
        "away_elo": elo_map.get(away_team),
        "elo_diff": (elo_map.get(home_team, 1500) - elo_map.get(away_team, 1500)),

        "home_squad_age": age_map.get(home_team),
        "away_squad_age": age_map.get(away_team),
        "home_market_value": value_map.get(home_team),
        "away_market_value": value_map.get(away_team),
        "market_value_ratio": (
            (value_map.get(home_team, 0) / value_map.get(away_team, 1))
            if value_map.get(away_team) else None
        ),

        **_form_features(home_buf, "home"),
        **_form_features(away_buf, "away"),
        **_h2h_features(h2h_buf, home_team, away_team),

        "home_days_rest": home_rest,
        "away_days_rest": away_rest,
        "is_world_cup": 1,
    }


def compute_odds_features(home_team: str, away_team: str, match_date: str) -> dict:
    """Convert betting odds to implied probabilities."""
    with get_db() as conn:
        odds = pd.read_sql_query(
            "SELECT home_win, draw, away_win FROM odds "
            "WHERE home_team = ? AND away_team = ? AND match_date = ?",
            conn, params=(home_team, away_team, match_date),
        )

    if odds.empty:
        return {
            "odds_home_prob": None,
            "odds_draw_prob": None,
            "odds_away_prob": None,
            "odds_spread": None,
        }

    # Convert decimal odds to implied probability
    odds["home_prob"] = 1 / odds["home_win"]
    odds["draw_prob"] = 1 / odds["draw"]
    odds["away_prob"] = 1 / odds["away_win"]

    return {
        "odds_home_prob": odds["home_prob"].mean(),
        "odds_draw_prob": odds["draw_prob"].mean(),
        "odds_away_prob": odds["away_prob"].mean(),
        "odds_spread": odds["home_prob"].max() - odds["home_prob"].min(),
    }