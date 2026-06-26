"""
World Cup 2026 — API Server & Final Ensemble Predictor (main.py)
"""

import uvicorn
import pandas as pd
from fastapi import FastAPI

from src.models.leaders import run_pipeline as get_leaders_results
from src.models.winner_profile import run_pipeline as get_profiles_results

# ── CONFIG & WEIGHTS ──────────────────────────────────────────────────────────
WEIGHT_LEADERS = 0.55
WEIGHT_PROFILES = 0.25
WEIGHT_BETS = 0.20

app = FastAPI(title="World Cup 2026 Predictor API")

# ── ENSEMBLE CORE LOGIC ───────────────────────────────────────────────────────

def _get_dataframe(obj) -> pd.DataFrame:
    """Recursively hunts down the pandas DataFrame from any returned object."""
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
    """Retrieves tournament outright betting probabilities."""
    return pd.DataFrame({
        'team': ['France', 'Brazil', 'England', 'Argentina', 'Spain', 'Germany'],
        'prob_bets': [0.18, 0.16, 0.14, 0.12, 0.10, 0.08]
    })


def run_ensemble() -> pd.DataFrame:
    print("Running Leaders pipeline (Match Data)...")
    leaders_out = get_leaders_results()
    df_leaders = _get_dataframe(leaders_out)
        
    print("\nRunning Profiles pipeline (Champion DNA)...")
    profiles_out = get_profiles_results()
    df_profiles = _get_dataframe(profiles_out)

    print("\nFetching Betting Odds...")
    df_bets = get_betting_odds()
    
    # Standardize column names
    df_leaders = df_leaders[['team', 'win_probability']].copy()
    df_leaders = df_leaders.rename(columns={'win_probability': 'prob_leaders'})
    
    df_profiles = df_profiles[['team', 'win_probability']].copy()
    df_profiles = df_profiles.rename(columns={'win_probability': 'prob_profiles'})
    
    # Merge pipelines
    merged = pd.merge(df_leaders, df_profiles, on='team', how='inner')
    
    # Merge betting odds
    merged = pd.merge(merged, df_bets, on='team', how='left')
    merged['prob_bets'] = merged['prob_bets'].fillna(0)
    
    # Calculate weighted ensemble probability
    merged['final_probability'] = (
        (merged['prob_leaders'] * WEIGHT_LEADERS) + 
        (merged['prob_profiles'] * WEIGHT_PROFILES) +
        (merged['prob_bets'] * WEIGHT_BETS)
    )
    
    # Format and sort
    merged['final_prob_pct'] = (merged['final_probability'] * 100).round(1)
    results = merged.sort_values(by='final_probability', ascending=False).reset_index(drop=True)
    
    # Get Top 5
    top_5 = results.head(5)
    
    # ── CONSOLE OUTPUT ────────────────────────────────────────────────────────
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


# ── FASTAPI ENDPOINTS ─────────────────────────────────────────────────────────

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


if __name__ == "__main__":
    # Runs the web server locally when main.py is executed directly
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)