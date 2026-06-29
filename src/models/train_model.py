"""Train the match-outcome model and save it to disk.

Run:  python src/train_model.py
Output: models/wc_model.pkl  (model + feature list)
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "training_features.csv"
MODEL_PATH = ROOT / "models" / "wc_model.pkl"

FEATURES = [
    "neutral",
    "home_ranking", "away_ranking", "ranking_diff",
    "home_elo", "away_elo", "elo_diff",
    "home_squad_age", "away_squad_age",
    "home_market_value", "away_market_value", "market_value_ratio",
    "home_form_win_rate", "home_form_pts_rate", "home_form_gf", "home_form_ga", "home_form_gd",
    "away_form_win_rate", "away_form_pts_rate", "away_form_gf", "away_form_ga", "away_form_gd",
    "h2h_matches", "h2h_home_wins", "h2h_away_wins", "h2h_draws",
    "home_days_rest", "away_days_rest", "is_world_cup",
]


def train(save: bool = True) -> HistGradientBoostingClassifier:
    df = pd.read_csv(DATA, parse_dates=["date"])
    X, y = df[FEATURES], df["result"]

    model = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        l2_regularization=1.0,
        random_state=42,
    )
    model.fit(X, y)

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
        print(f"Saved model -> {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train()
