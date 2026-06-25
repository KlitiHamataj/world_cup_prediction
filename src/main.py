from fastapi import FastAPI
import pandas as pd

from src.who_wins import run_ensemble, get_betting_odds, _get_dataframe
from src.leaders import run_pipeline as get_leaders_results
from src.winner_profile import run_pipeline as get_profiles_results

app = FastAPI()

@app.get("/api/ensemble")
def ensemble_predictions():
    df = run_ensemble()
    return df.to_dict(orient="records")

@app.get("/api/leaders")
def leaders_predictions():
    results = get_leaders_results()
    df = _get_dataframe(results)
    return df.to_dict(orient="records")

@app.get("/api/profiles")
def profiles_predictions():
    results = get_profiles_results()
    df = _get_dataframe(results)
    return df.to_dict(orient="records")

@app.get("/api/bets")
def betting_odds():
    df = get_betting_odds()
    return df.to_dict(orient="records")