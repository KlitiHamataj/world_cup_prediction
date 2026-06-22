"""
Feature engineering

Computes ML-ready features from raw match data. For Amelie and Mark to call
build_training_features() and get a clean DataFrame ready for model training.

FEATURES

Target column = "HOME" / "DRAW" / "AWAY".
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

Rest & Context:
  home_days_rest        — days since home team's last match (max 180)
  away_days_rest        — days since away team's last match (max 180)
  neutral               — 1 if neutral venue, 0 if home advantage applies
  is_world_cup          — 1 if the match is a World Cup match, 0 otherwise
"""

import pandas as pd
import numpy as np
from src.db import get_db, queries


def _outcome(row: pd.Series) -> str:
    """Determine match outcome from the home team perspective."""
    if row["home_score"] > row["away_score"]:
        return "HOME"
    elif row["home_score"] < row["away_score"]:
        return "AWAY"
    return "DRAW"


def compute_rolling_form(df: pd.DataFrame, team: str, n: int = 5) -> dict:
    """
    Compute rolling form stats for a team over their last N matches.
    Works on any DataFrame with: date, home_team, away_team, home_score, away_score.
    """
    # Get matches involving this team, sorted by date descending
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    team_matches = df[mask].sort_values("date", ascending=False).head(n)

    if team_matches.empty:
        return {
            "form_matches": 0, "form_wins": 0, "form_draws": 0, "form_losses": 0,
            "form_goals_for": 0, "form_goals_against": 0,
            "form_goal_diff": 0, "form_win_rate": 0.0,
            "form_points_rate": 0.0,
        }

    wins, draws, losses = 0, 0, 0
    goals_for, goals_against = 0, 0

    for _, m in team_matches.iterrows():
        if m["home_team"] == team:
            gf, ga = m["home_score"], m["away_score"]
        else:
            gf, ga = m["away_score"], m["home_score"]

        goals_for += gf
        goals_against += ga

        if gf > ga:
            wins += 1
        elif gf < ga:
            losses += 1
        else:
            draws += 1

    total = len(team_matches)
    points = wins * 3 + draws
    max_points = total * 3

    return {
        "form_matches": total,
        "form_wins": wins,
        "form_draws": draws,
        "form_losses": losses,
        "form_goals_for": goals_for,
        "form_goals_against": goals_against,
        "form_goal_diff": goals_for - goals_against,
        "form_win_rate": wins / total if total else 0,
        "form_points_rate": points / max_points if max_points else 0,
    }


def compute_head_to_head(df: pd.DataFrame, team_a: str, team_b: str,
                         n: int = 10) -> dict:
    """H2H record between two teams over their last N encounters."""
    mask = (
        ((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
        ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    )
    h2h = df[mask].sort_values("date", ascending=False).head(n)

    if h2h.empty:
        return {"h2h_matches": 0, "h2h_a_wins": 0, "h2h_b_wins": 0, "h2h_draws": 0}

    a_wins, b_wins, draws = 0, 0, 0
    for _, m in h2h.iterrows():
        if m["home_team"] == team_a:
            gf_a, gf_b = m["home_score"], m["away_score"]
        else:
            gf_a, gf_b = m["away_score"], m["home_score"]

        if gf_a > gf_b:
            a_wins += 1
        elif gf_b > gf_a:
            b_wins += 1
        else:
            draws += 1

    return {
        "h2h_matches": len(h2h),
        "h2h_a_wins": a_wins,
        "h2h_b_wins": b_wins,
        "h2h_draws": draws,
    }


def compute_days_since_last(df: pd.DataFrame, team: str,
                            reference_date: str) -> int:
    """Days between the reference date and the team's most recent match."""
    mask = (df["home_team"] == team) | (df["away_team"] == team)
    team_df = df[mask].copy()
    team_df["date"] = pd.to_datetime(team_df["date"])
    ref = pd.to_datetime(reference_date)
    past = team_df[team_df["date"] < ref]
    if past.empty:
        return 999  # no prior data
    return (ref - past["date"].max()).days


# ---------------------------------------------------------------------------
# Build the full feature matrix for training
# ---------------------------------------------------------------------------

def build_training_features(form_window: int = 5, h2h_window: int = 10) -> pd.DataFrame:
    """
    Build a feature DataFrame from the historical database.
    Each row = one match, with features computed from data BEFORE that match.

    Returns DataFrame ready for sklearn:
        date, home_team, away_team, target (HOME/DRAW/AWAY), + feature columns.
    """
    with get_db() as conn:
        # All historical matches
        hist = pd.read_sql_query(
            "SELECT * FROM matches_historical WHERE home_score IS NOT NULL ORDER BY date",
            conn,
        )
        # Team metadata
        teams = pd.read_sql_query(
            "SELECT name, fifa_ranking, elo_rating, squad_avg_age, market_value FROM teams",
            conn,
        )

    if hist.empty:
        raise ValueError("No historical data in database. Run ingest_historical first.")

    # Build lookup for rankings/elo/squad
    rank_map = dict(zip(teams["name"], teams["fifa_ranking"]))
    elo_map = dict(zip(teams["name"], teams["elo_rating"]))
    age_map = dict(zip(teams["name"], teams["squad_avg_age"]))
    value_map = dict(zip(teams["name"], teams["market_value"]))

    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)

    # We compute features only for a useful training window
    # (matches before ~2010 have limited feature data, and the game was different)
    cutoff = pd.Timestamp("2010-01-01")
    train_mask = hist["date"] >= cutoff
    train_idx = hist[train_mask].index

    rows = []
    for idx in train_idx:
        match = hist.loc[idx]
        # Only use matches BEFORE this one for features
        prior = hist.loc[:idx - 1] if idx > 0 else hist.iloc[:0]

        home, away = match["home_team"], match["away_team"]
        match_date = match["date"].strftime("%Y-%m-%d")

        home_form = compute_rolling_form(prior, home, form_window)
        away_form = compute_rolling_form(prior, away, form_window)
        h2h = compute_head_to_head(prior, home, away, h2h_window)

        row = {
            "date": match_date,
            "home_team": home,
            "away_team": away,
            "tournament": match.get("tournament"),
            "neutral": int(match.get("neutral", 0)),
            "target": _outcome(match),

            # Rankings & Elo (static — latest available)
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

            # Home form
            "home_form_win_rate": home_form["form_win_rate"],
            "home_form_pts_rate": home_form["form_points_rate"],
            "home_form_gf": home_form["form_goals_for"],
            "home_form_ga": home_form["form_goals_against"],
            "home_form_gd": home_form["form_goal_diff"],

            # Away form
            "away_form_win_rate": away_form["form_win_rate"],
            "away_form_pts_rate": away_form["form_points_rate"],
            "away_form_gf": away_form["form_goals_for"],
            "away_form_ga": away_form["form_goals_against"],
            "away_form_gd": away_form["form_goal_diff"],

            # H2H
            "h2h_matches": h2h["h2h_matches"],
            "h2h_home_wins": h2h["h2h_a_wins"],
            "h2h_away_wins": h2h["h2h_b_wins"],
            "h2h_draws": h2h["h2h_draws"],

            # Rest
            "home_days_rest": compute_days_since_last(prior, home, match_date),
            "away_days_rest": compute_days_since_last(prior, away, match_date),
        }
        rows.append(row)

    features_df = pd.DataFrame(rows)

    # Cap extreme rest values
    for col in ["home_days_rest", "away_days_rest"]:
        features_df[col] = features_df[col].clip(upper=180)

    # Tournament type encoding
    wc_keywords = ["world cup", "fifa"]
    features_df["is_world_cup"] = features_df["tournament"].fillna("").str.lower().apply(
        lambda t: int(any(kw in t for kw in wc_keywords))
    )

    print(f"✓ Built {len(features_df):,} training samples with {features_df.shape[1]} features")
    return features_df


def build_prediction_features(home_team: str, away_team: str,
                              match_date: str, neutral: int = 1,
                              form_window: int = 5) -> dict:
    """
    Build features for a single upcoming match (for inference).
    Same feature set as training, but computed from the latest available data.
    """
    with get_db() as conn:
        hist = pd.read_sql_query(
            "SELECT * FROM matches_historical WHERE home_score IS NOT NULL ORDER BY date",
            conn,
        )
        wc = pd.read_sql_query(
            "SELECT date, home_team, away_team, home_score, away_score, stage AS tournament "
            "FROM matches_wc2026 WHERE status='FINISHED'",
            conn,
        )
        teams = pd.read_sql_query(
            "SELECT name, fifa_ranking, elo_rating, squad_avg_age, market_value FROM teams",
            conn,
        )

    # Combine historical + current WC results
    wc["neutral"] = 1
    all_matches = pd.concat([hist, wc], ignore_index=True)
    all_matches["date"] = pd.to_datetime(all_matches["date"])
    all_matches = all_matches.sort_values("date")

    rank_map = dict(zip(teams["name"], teams["fifa_ranking"]))
    elo_map = dict(zip(teams["name"], teams["elo_rating"]))
    age_map = dict(zip(teams["name"], teams["squad_avg_age"]))
    value_map = dict(zip(teams["name"], teams["market_value"]))

    home_form = compute_rolling_form(all_matches, home_team, form_window)
    away_form = compute_rolling_form(all_matches, away_team, form_window)
    h2h = compute_head_to_head(all_matches, home_team, away_team, 10)

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

        "home_form_win_rate": home_form["form_win_rate"],
        "home_form_pts_rate": home_form["form_points_rate"],
        "home_form_gf": home_form["form_goals_for"],
        "home_form_ga": home_form["form_goals_against"],
        "home_form_gd": home_form["form_goal_diff"],

        "away_form_win_rate": away_form["form_win_rate"],
        "away_form_pts_rate": away_form["form_points_rate"],
        "away_form_gf": away_form["form_goals_for"],
        "away_form_ga": away_form["form_goals_against"],
        "away_form_gd": away_form["form_goal_diff"],

        "h2h_matches": h2h["h2h_matches"],
        "h2h_home_wins": h2h["h2h_a_wins"],
        "h2h_away_wins": h2h["h2h_b_wins"],
        "h2h_draws": h2h["h2h_draws"],

        "home_days_rest": compute_days_since_last(all_matches, home_team, match_date),
        "away_days_rest": compute_days_since_last(all_matches, away_team, match_date),

        "is_world_cup": 1,
    }
