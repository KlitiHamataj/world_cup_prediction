"""
World Cup 2026 — Winner Prediction Pipeline
Usage: python feature_engineering.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
FEATURES = [
    "matches_played",
    "avg_scored",
    "avg_conceded",
    "goal_diff_avg",
    "win_percentage",
    "historical_top_3_finishes",
    "elo_rating",
]
FORM_WINDOW = 4
MIN_PROB = 0.03


def build_name_map() -> dict:
    """
    Builds a mapping dictionary to standardize historical and alternate country names.
    
    Returns:
        dict: Mapping of former names to current standard names.
    """
    manual = {
        "USA": "United States",
        "Republic of Ireland": "Ireland",
        "West Germany": "Germany",
        "Soviet Union": "Russia",
        "FR Yugoslavia": "Serbia",
        "Zaire": "DR Congo",
        "Dutch East Indies": "Indonesia",
    }
    try:
        df_former = pd.read_csv(DATA_DIR / "former_names.csv")
        auto = dict(zip(df_former["former"], df_former["current"]))
        auto.update(manual)
        return auto
    except FileNotFoundError:
        return manual


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads core datasets required for feature engineering and modeling.
    
    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Match results, World Cup history, and Elo ratings DataFrames.
    """
    df_results = pd.read_csv(DATA_DIR / "results.csv", parse_dates=["date"])
    df_wc = pd.read_csv(DATA_DIR / "wc_all_editions.csv")
    df_elo = pd.read_csv(DATA_DIR / "elo_ratings_wc2026.csv", parse_dates=["snapshot_date"])
    return df_results, df_wc, df_elo


def normalise(df_results: pd.DataFrame, df_wc: pd.DataFrame, df_elo: pd.DataFrame, name_map: dict) -> tuple:
    """
    Applies the standardized name mapping across all relevant columns in the datasets.
    
    Args:
        df_results (pd.DataFrame): Match results DataFrame.
        df_wc (pd.DataFrame): World Cup history DataFrame.
        df_elo (pd.DataFrame): Elo ratings DataFrame.
        name_map (dict): Mapping dictionary for country names.
        
    Returns:
        tuple: The updated df_results, df_wc, and df_elo DataFrames.
    """
    for col in ("home_team", "away_team"):
        df_results[col] = df_results[col].replace(name_map)
    for col in ("champion", "runner_up", "third_place"):
        df_wc[col] = df_wc[col].replace(name_map)
    df_elo["country"] = df_elo["country"].replace(name_map)
    return df_results, df_wc, df_elo


def calc_form(df_results: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """
    Calculates performance metrics (goals, win rate) over a specific pre-tournament window.
    
    Args:
        df_results (pd.DataFrame): Match results DataFrame.
        target_year (int): The year of the tournament to calculate form leading up to.
        
    Returns:
        pd.DataFrame: Aggregated form statistics per team for the given cycle.
    """
    start = pd.Timestamp(f"{target_year - FORM_WINDOW}-01-01")
    end = pd.Timestamp(f"{target_year}-05-31")
    cycle = df_results[(df_results["date"] >= start) & (df_results["date"] <= end)]

    home = cycle[["home_team", "home_score", "away_score"]].copy()
    home["pts"] = np.where(home["home_score"] > home["away_score"], 3,
                  np.where(home["home_score"] == home["away_score"], 1, 0))
    home.columns = ["team", "scored", "conceded", "pts"]

    away = cycle[["away_team", "away_score", "home_score"]].copy()
    away["pts"] = np.where(away["away_score"] > away["home_score"], 3,
                  np.where(away["away_score"] == away["home_score"], 1, 0))
    away.columns = ["team", "scored", "conceded", "pts"]

    stats = pd.concat([home, away]).groupby("team").agg(
        matches_played=("scored", "count"),
        avg_scored=("scored", "mean"),
        avg_conceded=("conceded", "mean"),
        win_percentage=("pts", lambda x: (x == 3).mean()),
    ).reset_index()

    stats["goal_diff_avg"] = stats["avg_scored"] - stats["avg_conceded"]
    stats["year"] = target_year
    return stats


def attach_elo(df: pd.DataFrame, df_elo: pd.DataFrame) -> pd.DataFrame:
    """
    Merges historical Elo ratings onto the target DataFrame.
    
    Args:
        df (pd.DataFrame): Target DataFrame containing 'year' and 'team' columns.
        df_elo (pd.DataFrame): DataFrame containing Elo ratings over time.
        
    Returns:
        pd.DataFrame: The target DataFrame enriched with 'elo_rating'.
    """
    elo_annual = (
        df_elo.sort_values("snapshot_date")
              .groupby(["year", "country"])
              .last()
              .reset_index()[["year", "country", "rating"]]
              .rename(columns={"country": "team", "rating": "elo_rating"})
    )
    df = df.merge(elo_annual, on=["year", "team"], how="left")
    df["elo_rating"] = df.groupby("year")["elo_rating"].transform(lambda x: x.fillna(x.median()))
    return df


def build_pedigree(df_wc: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates cumulative top-3 World Cup finishes for each team historically.
    
    Args:
        df_wc (pd.DataFrame): World Cup history DataFrame.
        
    Returns:
        pd.DataFrame: Cumulative historical pedigree for each team by year.
    """
    rows = []
    for _, row in df_wc.iterrows():
        for team in (row["champion"], row["runner_up"], row["third_place"]):
            if pd.notna(team):
                rows.append({"year": row["year"], "team": team})

    ped = pd.DataFrame(rows).sort_values("year")
    ped["count"] = 1
    ped["historical_top_3_finishes"] = ped.groupby("team")["count"].cumsum() - 1
    return ped[["year", "team", "historical_top_3_finishes"]]


def attach_pedigree(df: pd.DataFrame, pedigree: pd.DataFrame) -> pd.DataFrame:
    """
    Merges the historical pedigree data onto the target DataFrame, ensuring no data leakage.
    
    Args:
        df (pd.DataFrame): Target DataFrame.
        pedigree (pd.DataFrame): Pedigree DataFrame generated by build_pedigree.
        
    Returns:
        pd.DataFrame: Target DataFrame enriched with 'historical_top_3_finishes'.
    """
    merged = df.merge(pedigree, on="team", suffixes=("", "_ped"))
    merged = merged[merged["year_ped"] < merged["year"]]
    best = merged.groupby(["year", "team"])["historical_top_3_finishes"].max().reset_index()
    df = df.merge(best, on=["year", "team"], how="left")
    df["historical_top_3_finishes"] = df["historical_top_3_finishes"].fillna(0)
    return df


def build_training_data(df_results: pd.DataFrame, df_wc: pd.DataFrame, df_elo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Assembles the historical training dataset incorporating form, Elo, and pedigree features.
    
    Args:
        df_results (pd.DataFrame): Match results DataFrame.
        df_wc (pd.DataFrame): World Cup history DataFrame.
        df_elo (pd.DataFrame): Elo ratings DataFrame.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The final training DataFrame and the computed pedigree DataFrame.
    """
    wc_matches = df_results[df_results["tournament"] == "FIFA World Cup"].copy()
    wc_matches["year"] = wc_matches["date"].dt.year

    t1 = wc_matches[["year", "home_team"]].rename(columns={"home_team": "team"})
    t2 = wc_matches[["year", "away_team"]].rename(columns={"away_team": "team"})
    base = pd.concat([t1, t2]).drop_duplicates().reset_index(drop=True)

    label_rows = []
    for _, row in df_wc.iterrows():
        for team in (row["champion"], row["runner_up"], row["third_place"]):
            if pd.notna(team):
                label_rows.append({"year": row["year"], "team": team, "is_top_3": 1})
                
    labels = pd.DataFrame(label_rows)
    base = base.merge(labels, on=["year", "team"], how="left")
    base["is_top_3"] = base["is_top_3"].fillna(0).astype(int)

    pedigree = build_pedigree(df_wc)
    base = attach_pedigree(base, pedigree)

    df_form = pd.concat([calc_form(df_results, y) for y in base["year"].unique()])
    base = base.merge(df_form, on=["year", "team"], how="left")
    base = attach_elo(base, df_elo)
    return base.fillna(0), pedigree


def build_2026_data(df_results: pd.DataFrame, df_elo: pd.DataFrame, pedigree: pd.DataFrame) -> pd.DataFrame:
    """
    Assembles the feature set specifically for the 2026 tournament prediction.
    
    Args:
        df_results (pd.DataFrame): Match results DataFrame.
        df_elo (pd.DataFrame): Elo ratings DataFrame.
        pedigree (pd.DataFrame): Computed historical pedigree DataFrame.
        
    Returns:
        pd.DataFrame: The complete feature set for 2026 tournament participants.
    """
    teams = df_elo[df_elo["snapshot_date"].dt.year == 2026]["country"].dropna().unique().tolist()
    df_2026 = pd.DataFrame({"year": 2026, "team": teams})

    df_2026 = attach_pedigree(df_2026, pedigree)
    form = calc_form(df_results, 2026)
    df_2026 = df_2026.merge(form, on=["year", "team"], how="left")
    df_2026 = attach_elo(df_2026, df_elo)
    return df_2026.fillna(0)


def train_and_predict(training_df: pd.DataFrame, df_2026: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trains a RandomForestClassifier on historical data and predicts probabilities for 2026.
    
    Args:
        training_df (pd.DataFrame): Historical dataset containing features and labels.
        df_2026 (pd.DataFrame): 2026 feature set to run predictions on.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Predictions DataFrame sorted by probability, and Feature Importances DataFrame.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(training_df[FEATURES], training_df["is_top_3"])

    importance = (
        pd.DataFrame({"feature": FEATURES, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    df_2026 = df_2026.copy()
    df_2026["win_probability"] = model.predict_proba(df_2026[FEATURES])[:, 1]

    predictions = (
        df_2026[["team", "win_probability"]]
        .sort_values("win_probability", ascending=False)
        .reset_index(drop=True)
    )
    return predictions, importance


def run_pipeline(min_probability: float = MIN_PROB) -> dict:
    """
    Executes the end-to-end data processing, training, and prediction pipeline.
    
    Args:
        min_probability (float, optional): The minimum win probability threshold for candidates. Defaults to MIN_PROB.
        
    Returns:
        dict: A dictionary containing candidates, all predictions, and feature importance data.
    """
    print("Loading data ...")
    df_results, df_wc, df_elo = load_data()

    name_map = build_name_map()
    df_results, df_wc, df_elo = normalise(df_results, df_wc, df_elo, name_map)

    print("Building training features ...")
    training_df, pedigree = build_training_data(df_results, df_wc, df_elo)

    print("Building 2026 features ...")
    df_2026 = build_2026_data(df_results, df_elo, pedigree)

    print("Training model & predicting ...")
    predictions, importance = train_and_predict(training_df, df_2026)

    candidates = predictions[predictions["win_probability"] >= min_probability].copy()
    candidates["probability_pct"] = (candidates["win_probability"] * 100).round(1)

    print(f"\n{'='*44}")
    print("   WHAT DRIVES A WORLD CUP WINNER?")
    print(f"{'='*44}")
    for _, row in importance.iterrows():
        bar = "█" * int(row["importance"] * 40)
        print(f"  {row['feature']:<30} {bar}  {row['importance']:.3f}")

    print(f"\n{'='*44}")
    print(f"  2026 CANDIDATES  (>{int(min_probability * 100)}% chance)")
    print(f"{'='*44}")
    print(f"  {'Team':<26} {'Probability':>10}")
    print(f"  {'-'*38}")
    for _, row in candidates.iterrows():
        print(f"  {row['team']:<26} {row['probability_pct']:>8.1f}%")
    print(f"{'='*44}\n")

    return {
        "candidates": candidates[["team", "win_probability", "probability_pct"]],
        "all_predictions": predictions,
        "feature_importance": importance,
    }

if __name__ == "__main__":
    results = run_pipeline(min_probability=MIN_PROB)