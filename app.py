"""World Cup 2026 prediction dashboard (Flask).

Run:  python app.py
Then open http://127.0.0.1:5000
"""
from pathlib import Path
import sys

import pandas as pd
from flask import Flask, render_template, request, jsonify

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from src.models.predictor import Predictor
from src.models.simulator import run_full_simulation, COIN_BIAS
from src.utils.flags import flag, FLAGS, CODES
from src.models.ensemble import get_ensemble, get_dashboard_data, WEIGHT_LEADERS, WEIGHT_PROFILES

app = Flask(__name__)
app.jinja_env.globals["flag"] = flag
_predictor = Predictor()          # loaded once at startup
_cache = {}                        # seed -> simulation result

TEAMS = sorted(pd.read_csv(ROOT / "data" / "raw" / "wc_2026_teams.csv")["team"].tolist())

# Official 2026 Round-of-32 fixtures. Left half = slots 0-7, right half = 8-15.
# Used by the builder's "Official bracket" preset (?preset=official).
OFFICIAL_R32 = [
    # ----- Left half (slots 0-7): adjacent pairs meet in the Round of 16 -----
    {"team1": "Germany", "team2": "Paraguay"},               # N74  -> R16 N89
    {"team1": "France", "team2": "Sweden"},                  # N77  -> R16 N89
    {"team1": "South Africa", "team2": "Canada"},            # N73  -> R16 N90
    {"team1": "Netherlands", "team2": "Morocco"},            # N75  -> R16 N90
    {"team1": "Portugal", "team2": "Croatia"},               # N83  -> R16 N93
    {"team1": "Spain", "team2": "Austria"},                  # N84  -> R16 N93
    {"team1": "USA", "team2": "Bosnia and Herzegovina"},     # N81  -> R16 N94
    {"team1": "Belgium", "team2": "Senegal"},                # N82  -> R16 N94
    # ----- Right half (slots 8-15) -----
    {"team1": "Brazil", "team2": "Japan"},                   # N76  -> R16 N91
    {"team1": "Ivory Coast", "team2": "Norway"},             # N78  -> R16 N91
    {"team1": "Mexico", "team2": "Ecuador"},                 # N79  -> R16 N92
    {"team1": "England", "team2": "DR Congo"},               # N80  -> R16 N92
    {"team1": "Argentina", "team2": "Cape Verde"},           # N86  -> R16 N95
    {"team1": "Australia", "team2": "Egypt"},                # N88  -> R16 N95
    {"team1": "Switzerland", "team2": "Algeria"},            # N85  -> R16 N96
    {"team1": "Colombia", "team2": "Ghana"},                 # N87  -> R16 N96
]


def get_sim(seed: int):
    if seed not in _cache:
        _cache[seed] = run_full_simulation(_predictor, seed=seed)
    return _cache[seed]


def _seed():
    return request.args.get("seed", default=42, type=int)


@app.route("/groups")
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
            verdict = max(outcomes, key=outcomes.get) #type: ignore
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


@app.route("/builder")
def builder():
    """Custom bracket: the user places teams, an animation resolves each tie."""
    seed = _seed()
    sim = get_sim(seed)

    preset = request.args.get("preset")
    if preset == "official":
        first_round = [dict(p) for p in OFFICIAL_R32]
    else:
        # Pre-fill the Round-of-32 with the qualifiers from the current simulation.
        r32 = [m for m in sim["ko_matches"] if m["stage"] == "Round of 32"]
        first_round = [{"team1": m["team1"], "team2": m["team2"]} for m in r32]

    # Teams not currently in the bracket (the "bench") = all WC teams minus the 32.
    in_bracket = {m["team1"] for m in first_round} | {m["team2"] for m in first_round}
    bench = sorted(t for t in TEAMS if t not in in_bracket)

    return render_template(
        "builder.html",
        page="builder",
        seed=seed,
        champion=sim["champion"],
        teams=TEAMS,
        first_round=first_round,
        bench=bench,
        flags=CODES,
        preset=preset,
    )


def _decide_winner(t1: str, t2: str):
    """Resolve a knockout tie at a neutral venue using the model.

    A tie can't end in a draw: when DRAW is the top outcome the winner is the
    team with the higher win probability (a coin flip nudged to the favourite).
    """
    p = _predictor.predict(t1, t2, neutral=True, is_world_cup=True)
    top = max(p, key=p.get) #type: ignore
    coin_flip = top == "draw"
    winner = t1 if p["home"] >= p["away"] else t2
    return winner, p, coin_flip


@app.route("/api/match")
def api_match():
    t1 = request.args.get("team1", "")
    t2 = request.args.get("team2", "")
    if not t1 or not t2 or t1 == t2:
        return jsonify({"error": "Need two different teams."}), 400
    try:
        winner, p, coin_flip = _decide_winner(t1, t2)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({
        "team1": t1, "team2": t2,
        "p_home": p["home"], "p_draw": p["draw"], "p_away": p["away"],
        "winner": winner, "coin_flip": coin_flip,
    })


@app.route("/")
@app.route("/dashboard")
def dashboard():
    """Top-down favourites: ensemble of the 'leaders' and 'winner_profile' models.

    Unlike the simulation pages (which play out matches), this ranks every team
    by an overall probability of winning the cup, blending recent form/Elo with
    historical champion pedigree.
    """
    data = get_dashboard_data()
    ranking = data["ranking"]
    champion = ranking[0]["team"] if ranking else "-"
    return render_template(
        "dashboard.html",
        page="dashboard",
        seed=_seed(),
        champion=champion,
        ranking=ranking,
        cards=data["cards"],
        hosts=data["hosts"],
        confeds=data["confeds"],
        importances=data["importances"],
        weight_leaders=round(WEIGHT_LEADERS * 100),
        weight_profiles=round(WEIGHT_PROFILES * 100),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
