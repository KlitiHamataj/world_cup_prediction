"""
World Cup 2026 — API Server & Final Ensemble Predictor (main.py)
Usage: python main.py
"""

import pandas as pd
import uvicorn
from fastapi import FastAPI
from src.models.leaders import run_pipeline as get_leaders_results
from src.models.winner_profile import run_pipeline as get_profiles_results

WEIGHT_LEADERS = 0.55
WEIGHT_PROFILES = 0.25
WEIGHT_BETS = 0.20

app = FastAPI(title="World Cup 2026 Predictor API")


def _get_dataframe(obj) -> pd.DataFrame:
    """
    Recursively extracts a pandas DataFrame from any returned pipeline object.

    Args:
        obj (any): A DataFrame, dictionary, tuple, or list containing pipeline data.

    Returns:
        pd.DataFrame: The extracted or newly constructed DataFrame.
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict):
        for val in obj.values():
            if isinstance(val, pd.DataFrame):
                return val
        return pd.DataFrame(obj)
    if isinstance(obj, tuple):
        for item in obj:
            if isinstance(item, pd.DataFrame):
                return item
        return _get_dataframe(obj[0])
    if isinstance(obj, list):
        return pd.DataFrame(obj)
    return pd.DataFrame(obj)


def get_betting_odds() -> pd.DataFrame:
    """
    Retrieves tournament outright betting probabilities.

    Returns:
        pd.DataFrame: A DataFrame containing default baseline betting probabilities per team.
    """
    return pd.DataFrame({
        'team': ['France', 'Brazil', 'England', 'Argentina', 'Spain', 'Germany'],
        'prob_bets': [0.18, 0.16, 0.14, 0.12, 0.10, 0.08]
    })


def run_ensemble() -> pd.DataFrame:
    """
    Combines predictions from multiple pipelines using an ensemble weighting strategy.

    Returns:
        pd.DataFrame: Final calculated ensemble predictions sorted by winning probability.
    """
    print("Running Leaders pipeline (Match Data)...")
    leaders_out = get_leaders_results()
    df_leaders = _get_dataframe(leaders_out)
        
    print("\nRunning Profiles pipeline (Champion DNA)...")
    profiles_out = get_profiles_results()
    df_profiles = _get_dataframe(profiles_out)

    print("\nFetching Betting Odds...")
    df_bets = get_betting_odds()
    
    df_leaders = df_leaders[['team', 'win_probability']].copy()
    df_leaders = df_leaders.rename(columns={'win_probability': 'prob_leaders'})
    
    df_profiles = df_profiles[['team', 'win_probability']].copy()
    df_profiles = df_profiles.rename(columns={'win_probability': 'prob_profiles'})
    
    merged = pd.merge(df_leaders, df_profiles, on='team', how='inner')
    merged = pd.merge(merged, df_bets, on='team', how='left')
    merged['prob_bets'] = merged['prob_bets'].fillna(0)
    
    merged['final_probability'] = (
        (merged['prob_leaders'] * WEIGHT_LEADERS) + 
        (merged['prob_profiles'] * WEIGHT_PROFILES) +
        (merged['prob_bets'] * WEIGHT_BETS)
    )
    
    merged['final_prob_pct'] = (merged['final_probability'] * 100).round(1)
    results = merged.sort_values(by='final_probability', ascending=False).reset_index(drop=True)
    
    top_5 = results.head(5)
    
    print(f"\n{'='*68}")
    print("  WORLD CUP 2026: TOP 5 ENSEMBLE PREDICTIONS")
    print(f"{'='*68}")
    print(f"  {'Team':<18} | {'Leaders':>8} | {'Profile':>8} | {'Bets':>8} | {'FINAL':>7}")
    print(f"  {'-'*64}")
    
    for _, row in top_5.iterrows():
        team = row['team']
        p_lead = f"{row['prob_leaders']*100:.1f}%"
        p_prof = f"{row['prob_profiles']*100:.1f}%"
        p_bets = f"{row['prob_bets']*100:.1f}%"
        p_final = f"{row['final_prob_pct']:.1f}%"
        
        print(f"  {team:<18} | {p_lead:>8} | {p_prof:>8} | {p_bets:>8} | {p_final:>7}")
        
    print(f"{'='*68}\n")
    
    return results


@app.get("/api/ensemble")
def ensemble_predictions():
    """
    HTTP GET endpoint for retrieving integrated ensemble model records.
    """
    df = run_ensemble()
    return df.to_dict(orient="records")


@app.get("/api/leaders")
def leaders_predictions():
    """
    HTTP GET endpoint for retrieving specific match-leader based output records.
    """
    results = get_leaders_results()
    df = _get_dataframe(results)
    return df.to_dict(orient="records")


@app.get("/api/profiles")
def profiles_predictions():
    """
    HTTP GET endpoint for retrieving profile/pedigree-based prediction records.
    """
    results = get_profiles_results()
    df = _get_dataframe(results)
    return df.to_dict(orient="records")


@app.get("/api/bets")
def betting_odds():
    """
    HTTP GET endpoint for retrieving reference betting probabilities data.
    """
    df = get_betting_odds()
    return df.to_dict(orient="records")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)