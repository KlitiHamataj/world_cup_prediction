"""Ensemble winner predictor + dashboard insights.

Combines two top-down "who wins the cup" models into a favourites ranking:

  * leaders        -> recent form (4-yr window) + Elo + pedigree   (weight 0.69)
  * winner_profile -> historical "champion DNA" + host status       (weight 0.31)

The blended score is normalised so the "win probability" sums to 100% across
the field. Everything (ranking, stat cards, host info, confederation split and
the model's feature importances) is computed once and cached at module level.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .leaders import run_pipeline as _run_leaders, build_name_map
from .winner_profile import run_pipeline as _run_profiles

WEIGHT_LEADERS = 0.69
WEIGHT_PROFILES = 0.31

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"

HOSTS = ["USA", "Mexico", "Canada"]

# Model outputs use historical data-set names; map them to the display names
# used everywhere else in the app (matching the flag table / team list).
DISPLAY_NAMES = {"United States": "USA", "Turkey": "Türkiye"}

CONFED_NAMES = {
    "UEFA": "Europe",
    "CONMEBOL": "South America",
    "CONCACAF": "N. & C. America",
    "CAF": "Africa",
    "AFC": "Asia",
    "OFC": "Oceania",
}

FEATURE_LABELS = {
    "elo_rating": "Elo rating",
    "avg_scored": "Goals scored / game",
    "goal_diff_avg": "Goal difference / game",
    "win_percentage": "Win rate",
    "historical_top_3_finishes": "Past podiums",
    "avg_conceded": "Goals conceded / game",
    "matches_played": "Matches played",
}

_data: dict | None = None


def _confederation_map() -> dict:
    """team -> confederation (friendly name), from the Elo dataset."""
    elo = pd.read_csv(DATA_DIR / "elo_ratings_wc2026.csv")
    elo["country"] = elo["country"].replace(build_name_map()).replace(DISPLAY_NAMES)
    elo = elo.dropna(subset=["confederation"])
    if "snapshot_date" in elo.columns:
        elo = elo.sort_values("snapshot_date")
    last = elo.groupby("country")["confederation"].last()
    return {team: CONFED_NAMES.get(c, c) for team, c in last.items()}


def _compute() -> dict:
    """Run both pipelines once and build the full dashboard payload."""
    lead = _run_leaders()
    leaders = (
        lead["all_predictions"][["team", "win_probability"]]
        .rename(columns={"win_probability": "prob_leaders"})
    )
    importance = lead["feature_importance"]
    profiles = (
        _run_profiles(min_probability=0.0)[["team", "win_probability"]]
        .rename(columns={"win_probability": "prob_profiles"})
    )

    df = pd.merge(leaders, profiles, on="team", how="inner")

    # Blended score (independent per-team classifiers -> does not sum to 1)...
    df["score"] = df["prob_leaders"] * WEIGHT_LEADERS + df["prob_profiles"] * WEIGHT_PROFILES
    # ...then normalise into a real probability of winning the cup (sums to 100%).
    total = df["score"].sum()
    df["final_probability"] = df["score"] / total if total else 0.0

    df = df.sort_values("final_probability", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    df["final_pct"] = (df["final_probability"] * 100).round(1)
    df["leaders_pct"] = (df["prob_leaders"] * 100).round(1)
    df["profiles_pct"] = (df["prob_profiles"] * 100).round(1)
    df["team"] = df["team"].replace(DISPLAY_NAMES)

    # ---- Stat cards -------------------------------------------------------
    fav = df.iloc[0]
    in_form = df.loc[df["prob_leaders"].idxmax()]
    pedigree = df.loc[df["prob_profiles"].idxmax()]
    # Dark horse: most in-form team that lacks a champion pedigree (low history),
    # excluding the established favourites.
    low_ped = df[df["profiles_pct"] < 10.0]
    dark = (low_ped if not low_ped.empty else df).sort_values(
        "leaders_pct", ascending=False).iloc[0]
    cards = {
        "favourite": {"team": fav["team"], "value": float(fav["final_pct"])},
        "in_form": {"team": in_form["team"], "value": float(in_form["leaders_pct"])},
        "pedigree": {"team": pedigree["team"], "value": float(pedigree["profiles_pct"])},
        "dark_horse": {"team": dark["team"], "form": float(dark["leaders_pct"]),
                       "pedigree": float(dark["profiles_pct"])},
    }

    # ---- Host nations -----------------------------------------------------
    hosts = (
        df[df["team"].isin(HOSTS)][["rank", "team", "final_pct"]]
        .sort_values("rank")
        .to_dict("records")
    )

    # ---- Confederation split ---------------------------------------------
    cmap = _confederation_map()
    df["confed"] = df["team"].map(cmap)
    grp = df.dropna(subset=["confed"]).groupby("confed")
    confeds = (
        pd.DataFrame({
            "name": grp.size().index,
            "pct": (grp["final_probability"].sum() * 100).round(1).values,
            "count": grp.size().values,
        })
        .sort_values("pct", ascending=False)
        .to_dict("records")
    )

    # ---- Feature importances ---------------------------------------------
    imp = importance.copy()
    imp["label"] = imp["feature"].map(lambda f: FEATURE_LABELS.get(f, f))
    imp["pct"] = (imp["importance"] * 100).round(1)
    importances = imp[["label", "pct"]].to_dict("records")

    ranking_cols = ["rank", "team", "leaders_pct", "profiles_pct", "final_pct"]
    return {
        "ranking_df": df[ranking_cols],
        "ranking": df[ranking_cols].to_dict("records"),
        "cards": cards,
        "hosts": hosts,
        "confeds": confeds,
        "importances": importances,
    }


def get_dashboard_data(force: bool = False) -> dict:
    """Cached full dashboard payload (ranking + insights)."""
    global _data
    if _data is None or force:
        _data = _compute()
    return _data


def get_ensemble(force: bool = False) -> pd.DataFrame:
    """Cached favourites ranking as a DataFrame (back-compat)."""
    return get_dashboard_data(force)["ranking_df"]


def run_ensemble() -> pd.DataFrame:
    """Uncached ranking, for standalone use / debugging."""
    return _compute()["ranking_df"]


if __name__ == "__main__":
    d = _compute()
    print(d["ranking_df"].head(8).to_string(index=False))
    print("cards:", d["cards"])
    print("hosts:", d["hosts"])
    print("confeds:", d["confeds"])
    print("importances:", d["importances"])
