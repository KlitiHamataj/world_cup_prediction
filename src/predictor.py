"""Load the trained model and predict match outcome probabilities.

The model is trained on historical match features. To predict an arbitrary
matchup we rebuild each team's latest known stats (elo, ranking, value, form)
from the training data, plus the head-to-head history between the two teams.
"""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "training_features.csv"
MODEL_PATH = ROOT / "models" / "wc_model.pkl"

# World-Cup team names that differ from the names used in the training data.
NAME_MAP = {
    "Czechia": "Czech Republic",
    "USA": "United States",
    "Türkiye": "Turkey",
    "Ivory Coast": "Côte d'Ivoire",
    "DR Congo": "Congo DR",
}

_TEAM_COLS = ["ranking", "elo", "squad_age", "market_value",
              "form_win_rate", "form_pts_rate", "form_gf", "form_ga", "form_gd"]


class Predictor:
    def __init__(self, model_path: Path = MODEL_PATH, data_path: Path = DATA):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]
        self.classes = list(self.model.classes_)  # ['AWAY', 'DRAW', 'HOME']
        self.df = pd.read_csv(data_path, parse_dates=["date"]).sort_values("date")
        self._long = self._build_long_table(self.df)
        self._median_rest = int(self.df["home_days_rest"].median())

    # ----- helpers -------------------------------------------------------
    @staticmethod
    def _build_long_table(df: pd.DataFrame) -> pd.DataFrame:
        out = []
        for side in ("home", "away"):
            cols = ["date", f"{side}_team"] + [f"{side}_{c}" for c in _TEAM_COLS]
            sub = df[cols].copy()
            sub.columns = ["date", "team"] + _TEAM_COLS
            out.append(sub)
        return pd.concat(out).sort_values("date")

    def _resolve(self, team: str) -> str:
        return NAME_MAP.get(team, team)

    def _team_state(self, team: str):
        name = self._resolve(team)
        sub = self._long[self._long["team"] == name]
        if sub.empty:
            raise ValueError(f"Unknown team: {team!r} (resolved to {name!r})")
        return sub.iloc[-1]

    def _h2h(self, home: str, away: str):
        h, a = self._resolve(home), self._resolve(away)
        d = self.df
        m = d[((d.home_team == h) & (d.away_team == a)) |
              ((d.home_team == a) & (d.away_team == h))]
        hw = ((m.home_team == h) & (m.result == "HOME")).sum() + ((m.away_team == h) & (m.result == "AWAY")).sum()
        aw = ((m.home_team == a) & (m.result == "HOME")).sum() + ((m.away_team == a) & (m.result == "AWAY")).sum()
        return len(m), int(hw), int(aw), int((m.result == "DRAW").sum())

    # ----- public API ----------------------------------------------------
    def predict(self, home: str, away: str, neutral: bool = True,
                is_world_cup: bool = True) -> dict:
        """Return {'home': p, 'draw': p, 'away': p} probabilities.

        On a neutral venue the matchup has no real "home" team, yet the model
        was trained mostly on non-neutral games and keeps a residual bias for
        the home slot. To make the result order-independent we average the two
        orderings (A vs B and B vs A) so predict(A, B) and predict(B, A) are
        exact mirrors of each other.
        """
        if neutral:
            p1 = self._predict_raw(home, away, neutral=True, is_world_cup=is_world_cup)
            p2 = self._predict_raw(away, home, neutral=True, is_world_cup=is_world_cup)
            return {
                "home": (p1["home"] + p2["away"]) / 2,
                "draw": (p1["draw"] + p2["draw"]) / 2,
                "away": (p1["away"] + p2["home"]) / 2,
            }
        return self._predict_raw(home, away, neutral=False, is_world_cup=is_world_cup)

    def _predict_raw(self, home: str, away: str, neutral: bool = True,
                     is_world_cup: bool = True) -> dict:
        """Single-direction model call (home slot vs away slot)."""
        h, a = self._team_state(home), self._team_state(away)
        n, hw, aw, dr = self._h2h(home, away)

        def diff(x, y):
            return x - y if pd.notna(x) and pd.notna(y) else np.nan

        row = {
            "neutral": int(neutral),
            "home_ranking": h.ranking, "away_ranking": a.ranking,
            "ranking_diff": diff(h.ranking, a.ranking),
            "home_elo": h.elo, "away_elo": a.elo, "elo_diff": diff(h.elo, a.elo),
            "home_squad_age": h.squad_age, "away_squad_age": a.squad_age,
            "home_market_value": h.market_value, "away_market_value": a.market_value,
            "market_value_ratio": (h.market_value / a.market_value
                                   if pd.notna(h.market_value) and pd.notna(a.market_value) and a.market_value
                                   else np.nan),
            "home_form_win_rate": h.form_win_rate, "home_form_pts_rate": h.form_pts_rate,
            "home_form_gf": h.form_gf, "home_form_ga": h.form_ga, "home_form_gd": h.form_gd,
            "away_form_win_rate": a.form_win_rate, "away_form_pts_rate": a.form_pts_rate,
            "away_form_gf": a.form_gf, "away_form_ga": a.form_ga, "away_form_gd": a.form_gd,
            "h2h_matches": n, "h2h_home_wins": hw, "h2h_away_wins": aw, "h2h_draws": dr,
            "home_days_rest": self._median_rest, "away_days_rest": self._median_rest,
            "is_world_cup": int(is_world_cup),
        }
        p = self.model.predict_proba(pd.DataFrame([row])[self.features])[0]
        d = dict(zip(self.classes, p))
        return {"home": float(d["HOME"]), "draw": float(d["DRAW"]), "away": float(d["AWAY"])}
