"""
World Cup 2026 — Winner Profiles Pipeline
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MIN_PROB = 0.03


# ── NAME NORMALISATION ────────────────────────────────────────────────────────
def build_name_map() -> dict:
    manual = {
        "USA":                 "United States",
        "Republic of Ireland": "Ireland",
        "West Germany":        "Germany",
        "Soviet Union":        "Russia",
        "FR Yugoslavia":       "Serbia",
        "Zaire":               "DR Congo",
        "Dutch East Indies":   "Indonesia",
    }
    try:
        df_former = pd.read_csv(DATA_DIR / "former_names.csv")
        auto = dict(zip(df_former["former"], df_former["current"]))
        auto.update(manual)
        return auto
    except FileNotFoundError:
        return manual


def normalize(name: str, name_map: dict) -> str:
    return name_map.get(name, name)


# ── LOAD & PREPROCESS ─────────────────────────────────────────────────────────
def load_data(name_map: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    editions = pd.read_csv(DATA_DIR / "wc_all_editions.csv")
    matches = pd.read_csv(DATA_DIR / "wc_all_matches.csv")
    elo = pd.read_csv(DATA_DIR / "elo_ratings_wc2026.csv", parse_dates=["snapshot_date"])
    
    editions["champion"] = editions["champion"].apply(lambda x: normalize(x, name_map) if pd.notna(x) else x)
    editions["host_list"] = editions["host"].apply(lambda h: [normalize(x.strip(), name_map) for x in str(h).split("/")])
    
    matches["team1"] = matches["team1"].apply(lambda x: normalize(x, name_map))
    matches["team2"] = matches["team2"].apply(lambda x: normalize(x, name_map))
    
    elo["country"] = elo["country"].apply(lambda x: normalize(x, name_map))
    elo = elo.sort_values("snapshot_date").drop_duplicates(["country", "year"], keep="last")
    
    return editions, matches, elo


# ── PARTICIPANTS & PROFILES ───────────────────────────────────────────────────
def build_participants(matches: pd.DataFrame, editions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in matches.groupby("year"):
        teams = set(group["team1"]) | set(group["team2"])
        edition = editions.loc[editions["year"] == year]
        if edition.empty:
            continue
        champion = edition["champion"].iloc[0]
        host_list = edition["host_list"].iloc[0]
        for team in teams:
            rows.append({
                "year": year,
                "team": team,
                "is_champion": int(team == champion),
                "is_host": int(team in host_list),
            })
    return pd.DataFrame(rows)


def attach_profiles(participants: pd.DataFrame, elo: pd.DataFrame) -> pd.DataFrame:
    elo_cols = [
        "rank", "rating", "rank_avg", "rating_avg", "matches_total", 
        "wins", "losses", "draws", "goals_for", "goals_against", "confederation"
    ]
    
    profile_rows = []
    for _, r in participants.iterrows():
        snap = elo[(elo["country"] == r["team"]) & (elo["year"] < r["year"])]
        if snap.empty:
            continue
        snap_latest = snap.sort_values("year").iloc[-1][elo_cols]
        profile_rows.append({**r.to_dict(), **snap_latest.to_dict()})
        
    profiles = pd.DataFrame(profile_rows)
    profiles["win_rate"] = profiles["wins"] / profiles["matches_total"].clip(lower=1)
    profiles["goal_diff_pg"] = (profiles["goals_for"] - profiles["goals_against"]) / profiles["matches_total"].clip(lower=1)
    
    return profiles


def get_2026_profiles(elo: pd.DataFrame, editions: pd.DataFrame) -> pd.DataFrame:
    edition_2026 = editions.loc[editions["year"] == 2026]
    if not edition_2026.empty and "host_list" in edition_2026.columns:
        host_list_2026 = edition_2026["host_list"].iloc[0]
    else:
        host_list_2026 = ["United States", "Mexico", "Canada"]
        
    # Teams with snapshots in 2026
    teams = elo[elo["snapshot_date"].dt.year == 2026]["country"].dropna().unique().tolist()
    participants_2026 = pd.DataFrame({
        "year": 2026,
        "team": teams,
        "is_champion": 0,
        "is_host": [int(t in host_list_2026) for t in teams]
    })
    
    return attach_profiles(participants_2026, elo)


# ── MODEL ─────────────────────────────────────────────────────────────────────
def train_and_predict(train_profiles: pd.DataFrame, profiles_2026: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = ["rank", "rating", "rank_avg", "rating_avg", "win_rate", "goal_diff_pg"]
    categorical_cols = ["confederation"]
    binary_cols = ["is_host"]
    
    feature_cols = numeric_cols + categorical_cols + binary_cols
    
    X = train_profiles[feature_cols]
    y = train_profiles["is_champion"]
    
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("bin", "passthrough", binary_cols),
    ])
    
    model = Pipeline([
        ("prep", preprocess),
        ("clf", RandomForestClassifier(
            n_estimators=300, 
            max_depth=4, 
            class_weight="balanced", 
            random_state=42
        ))
    ])
    
    model.fit(X, y)
    
    X_2026 = profiles_2026[feature_cols]
    profiles_2026["win_probability"] = model.predict_proba(X_2026)[:, 1]
    
    predictions = profiles_2026[["team", "win_probability"]].sort_values("win_probability", ascending=False).reset_index(drop=True)
    return predictions


# ── PIPELINE ──────────────────────────────────────────────────────────────────
def run_pipeline(min_probability: float = MIN_PROB) -> pd.DataFrame:
    print("Loading data ...")
    name_map = build_name_map()
    editions, matches, elo = load_data(name_map)
    
    print("Building historical profiles ...")
    participants = build_participants(matches, editions)
    train_profiles = attach_profiles(participants, elo)
    
    print("Building 2026 profiles ...")
    profiles_2026 = get_2026_profiles(elo, editions)
    
    print("Training model & predicting ...")
    predictions = train_and_predict(train_profiles, profiles_2026)
    
    # Filter candidates with chance > 3%
    candidates = predictions[predictions["win_probability"] > min_probability].copy()
    candidates["probability_pct"] = (candidates["win_probability"] * 100).round(1)
    
    # ── PRINT ─────────────────────────────────────────────────────────────────
    print(f"\n{'='*44}")
    print(f"  2026 CANDIDATES  (>{int(min_probability * 100)}% chance)")
    print(f"{'='*44}")
    print(f"  {'Team':<26} {'Probability':>10}")
    print(f"  {'-'*38}")
    for _, row in candidates.iterrows():
        print(f"  {row['team']:<26} {row['probability_pct']:>8.1f}%")
    print(f"{'='*44}\n")
    
    return candidates

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = run_pipeline(min_probability=MIN_PROB)