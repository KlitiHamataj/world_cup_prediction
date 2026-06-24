"""World Cup 2026 prediction dashboard (Flask).

Run:  python app.py
Then open http://127.0.0.1:5000
"""
from pathlib import Path
import sys

import pandas as pd
from flask import Flask, render_template, request

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from src.predictor import Predictor
from src.simulator import run_full_simulation, COIN_BIAS
from src.flags import flag

app = Flask(__name__)
app.jinja_env.globals["flag"] = flag
_predictor = Predictor()          # loaded once at startup
_cache = {}                        # seed -> simulation result

TEAMS = sorted(pd.read_csv(ROOT / "data" / "raw" / "wc_2026_teams.csv")["team"].tolist())


def get_sim(seed: int):
    if seed not in _cache:
        _cache[seed] = run_full_simulation(_predictor, seed=seed)
    return _cache[seed]


def _seed():
    return request.args.get("seed", default=42, type=int)


@app.route("/")
def groups():
    seed = _seed()
    sim = get_sim(seed)

    matches_by_group = {}
    for m in sim["group_matches"]:
        matches_by_group.setdefault(m["group"], []).append(m)

    groups_data = []
    for g in sorted(sim["standings"].keys()):
        groups_data.append({
            "name": g,
            "standings": sim["standings"][g].to_dict("records"),
            "matches": matches_by_group[g],
        })

    return render_template(
        "groups.html",
        page="groups",
        seed=seed,
        champion=sim["champion"],
        groups=groups_data,
        best_thirds=sim["best_thirds"].to_dict("records"),
    )


def _half(matches, name, ri, lo, hi):
    return {"name": name, "ri": ri, "matches": matches[lo:hi]}


@app.route("/bracket")
def bracket():
    seed = _seed()
    sim = get_sim(seed)

    ko = sim["ko_matches"]
    r32 = [m for m in ko if m["stage"] == "Round of 32"]
    r16 = [m for m in ko if m["stage"] == "Round of 16"]
    qf = [m for m in ko if m["stage"] == "Quarter-final"]
    sf = [m for m in ko if m["stage"] == "Semi-final"]
    final = [m for m in ko if m["stage"] == "Final"]
    third = [m for m in ko if m["stage"] == "3rd Place Match"]

    # Left half feeds Finalist 1, right half feeds Finalist 2.
    left_cols = [
        _half(r32, "Round of 32", 0, 0, 8),
        _half(r16, "Round of 16", 1, 0, 4),
        _half(qf, "Quarter-final", 2, 0, 2),
        _half(sf, "Semi-final", 3, 0, 1),
    ]
    # Right half rendered inner -> outer (Semi-final closest to the centre).
    right_cols = [
        _half(sf, "Semi-final", 3, 1, 2),
        _half(qf, "Quarter-final", 2, 2, 4),
        _half(r16, "Round of 16", 1, 4, 8),
        _half(r32, "Round of 32", 0, 8, 16),
    ]

    return render_template(
        "bracket.html",
        page="bracket",
        seed=seed,
        champion=sim["champion"],
        third_place=sim["third_place"],
        left_cols=left_cols,
        right_cols=right_cols,
        final_match=final[0] if final else None,
        third_match=third[0] if third else None,
        coin_bias=COIN_BIAS,
    )


@app.route("/predict")
def predict():
    seed = _seed()
    sim = get_sim(seed)

    t1 = request.args.get("team1", "")
    t2 = request.args.get("team2", "")
    neutral = request.args.get("neutral") == "on"

    result = None
    error = None
    if t1 and t2:
        if t1 == t2:
            error = "Pick two different teams."
        else:
            p = _predictor.predict(t1, t2, neutral=neutral, is_world_cup=True)
            outcomes = {f"{t1} win": p["home"], "Draw": p["draw"], f"{t2} win": p["away"]}
            verdict = max(outcomes, key=outcomes.get)
            result = {
                "team1": t1, "team2": t2, "neutral": neutral,
                "p_home": p["home"], "p_draw": p["draw"], "p_away": p["away"],
                "verdict": verdict,
            }

    return render_template(
        "predict.html",
        page="predict",
        seed=seed,
        champion=sim["champion"],
        teams=TEAMS,
        team1=t1, team2=t2, neutral=neutral,
        result=result, error=error,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
