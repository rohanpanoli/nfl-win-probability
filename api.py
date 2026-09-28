import joblib
import pandas as pd
import nflreadpy as nfl
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from utils import compute_current_elo, compute_current_team_stats

SEASON = 2026
FEATURES = [
    'home_roll_pf', 'home_roll_pa', 'home_roll_winpct',
    'away_roll_pf', 'away_roll_pa', 'away_roll_winpct',
    'home_elo', 'away_elo'
]

state = {}

def load_state():
    schedule = nfl.load_schedules([SEASON]).to_pandas()
    completed = schedule.dropna(subset=['home_score','away_score'])
    state['upcoming'] = schedule[schedule['home_score'].isna()].copy()
    state['stats'] = compute_current_team_stats(completed).set_index('team')
    state['elo'] = compute_current_elo(completed).set_index('team')['elo']
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    state['model'] = joblib.load('models/logreg_win_probability.pkl')
    load_state()
    yield
    
app = FastAPI(title='NFL Win Probability API', lifespan=lifespan)

def build_row(home: str, away: str) -> pd.DataFrame:
    for team in (home, away):
        if team not in state['stats'].index:
            raise HTTPException(404, f"Unknown team '{team}'. Valid {sorted(state['stats'].index)}")
        if home == away:
            raise HTTPException(400, "Home and Away teams must differ.")
        h, a = state['stats'].loc[home], state['stats'].loc[away]
        row = {
            'home_roll_pf': h['roll_pf'],
            'home_roll_pa': h['roll_pa'],
            'home_roll_winpct': h['roll_winpct'],
            'away_roll_pf': a['roll_pf'],
            'away_roll_pa': a['roll_pa'],
            'away_roll_winpct': a['roll_winpct'],
            'home_elo': state['elo'][home],
            'away_elo': state['elo'][away],
        }
        return pd.DataFrame([row])[FEATURES]
    
@app.get("/health")
def health():
    return {"status": "ok", "season": SEASON}

@app.get('/predict')
def predict(home: str, away: str):
    """Win probability for any hypothetical matchup."""
    prob = float(state['model'].predict_proba(build_row(home.upper(), away.upper()))[0,1])
    return {"home": home.upper(), "away": away.upper(),
            "home_win_prob": round(prob, 4), "away_win_prob": round(1-prob,4)}
    
@app.get("/predictions/{week}")
def week_predictions(week:int):
    """Predictions for every unplayed game in a given week."""
    games = state['upcoming'][state['upcoming']['week'] == week]
    if games.empty:
        raise HTTPException(404, f"No upcoming games found for week {week}.")
    results=[]
    for _, g in games.iterrows():
        prob = float(state['model'].predict_proba(build_row(g['home_team'], g['away_team']))[0,1])
        results.append({"home": g["home_team"], "away": g["away_team"],
                        "home_win_prob": round(prob,4)})
        return {"week": week, "games": results}
    
@app.get("/elo")
def elo_rankings():
    """Current Elo ratings."""
    ranked = state['elo'].sort_values(ascending=False)
    return [{"team": t, "elo": round(float(e), 1)} for t, e in ranked.items()]

@app.post("/refresh")
def refresh():
    """Re-pull schedule and recompute stats after new games are played."""
    load_state()
    return{"status": "refreshed"}