# ⚽ World Cup 2026 — Predictions & Simulation

A Python / Flask web app that predicts FIFA World Cup 2026 outcomes. It blends
historical match data, Elo ratings, squad value and recent form into machine
learning models, then surfaces everything through an interactive dashboard:
tournament favourites, a full group-stage and knockout simulation, an
interactive "build your own bracket" page, and a single-match predictor.

---

## Overview

The project runs **two complementary prediction systems**:

1. **Match-outcome model** — a `HistGradientBoostingClassifier` trained on
   2010–2026 international matches. Given any two teams it returns
   `HOME / DRAW / AWAY` probabilities. This model drives the single-match
   predictor and the full tournament simulation (group stage + knockout bracket).

2. **Top-down "who wins the cup" ensemble** — ranks every team by an overall
   probability of lifting the trophy, blending two models:
   - `leaders` — recent form (4-year window) + Elo + pedigree *(weight 0.69)*
   - `winner_profile` — historical "champion DNA" + host-nation status *(weight 0.31)*

A data pipeline keeps the underlying data fresh from external APIs, automated
locally with APScheduler and in CI with GitHub Actions.

---

## The dashboard

Run the app and open `http://127.0.0.1:5000`. Five pages:

| Page | Route | What it shows |
|---|---|---|
| **Favourites** | `/` | Top-down ensemble ranking of every team's title odds, with stat cards, host info, confederation split and model feature importances. |
| **Group stage** | `/groups` | Simulated group results, standings and best third-placed teams. |
| **Knockout bracket** | `/bracket` | Full simulated Round-of-32 → Final bracket. Knockout ties that the model calls a draw are resolved by a coin flip slightly biased toward the favourite. |
| **Build your bracket** | `/builder` | Place teams into the Round of 32 (simulated qualifiers or the official 2026 fixtures), drag to tweak, then watch the model resolve each tie up to the champion. |
| **Predict a match** | `/predict` | Pick any two teams and get win / draw / loss probabilities. |

Every simulation is seeded (`?seed=`), so results are reproducible and you can
re-simulate with a different seed. Flags are rendered as real images from
[flagcdn.com](https://flagcdn.com/) for consistent display across browsers.

---

## Project Structure

```
world_cup_prediction/
├── app.py                      # Flask app — all routes & page rendering
├── models/
│   └── wc_model.pkl            # Trained match-outcome model (gitignored)
├── data/
│   ├── raw/                    # Source CSVs (results, fixtures, teams, Elo, …)
│   ├── training_features.csv   # ML-ready feature table
│   └── football.db             # SQLite database (gitignored)
├── notebooks/                  # EDA, winner profiles, model training
├── src/
│   ├── data_pipeline/
│   │   ├── scraper.py          # football-data.org + the-odds-api.com clients
│   │   ├── pipeline.py         # ETL: scrape → transform → load into SQLite
│   │   └── db.py               # SQLite schema, connection & queries
│   ├── models/
│   │   ├── features.py         # Feature engineering from raw match data
│   │   ├── train_model.py      # Trains & saves the match-outcome model
│   │   ├── predictor.py        # Loads the model, predicts a matchup
│   │   ├── simulator.py        # Full tournament simulation (groups + knockout)
│   │   ├── leaders.py          # Form/Elo/pedigree title-odds model
│   │   ├── winner_profile.py   # "Champion DNA" + host title-odds model
│   │   └── ensemble.py         # Blends the two into the Favourites ranking
│   ├── scripts/
│   │   └── ingest_historical.py # Loads Kaggle CSVs into SQLite
│   ├── utils/
│   │   ├── config.py           # .env loader
│   │   ├── flags.py            # Team → flag (flagcdn image) helper
│   │   └── team_names.py       # Name normalisation across data sources
│   └── scheduler.py            # APScheduler — periodic data refresh
├── static/                     # style.css, bracket.js, builder.js
├── templates/                  # base + dashboard / groups / bracket / builder / predict
├── .github/workflows/
│   └── update_data.yml         # CI job: refresh data twice daily
├── env.example                 # API key template — copy to .env
└── requirements.txt
```

---

## Tech Stack

| Layer | Library |
|---|---|
| Web app / dashboard | Flask ≥ 3.0 |
| Data manipulation | pandas ≥ 2.0 |
| Numerical computation | numpy ≥ 1.24 |
| Machine learning | scikit-learn ≥ 1.3 |
| Model persistence | joblib ≥ 1.3 |
| Scheduling | APScheduler ≥ 3.10 |
| API calls | requests ≥ 2.31 |
| Dataset download | kaggle ≥ 1.6 |

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/KlitiHamataj/world_cup_prediction.git
cd world_cup_prediction
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables (optional)

API keys are only needed to refresh data from the live APIs. The app runs on the
data already in `data/` without them.

```bash
cp env.example .env
```

```env
# football-data.org
FOOTBALL_DATA_API_KEY=your_key_here

# the-odds-api.com
ODDS_API_KEY=your_key_here
```

A Kaggle API token (`~/.kaggle/kaggle.json`) is required only for downloading the
historical datasets. See the [Kaggle docs](https://www.kaggle.com/docs/api).

---

## Usage

### Run the dashboard

```bash
python app.py
```

Then open `http://127.0.0.1:5000`.

### (Re)train the match-outcome model

```bash
python -m src.models.train_model     # writes models/wc_model.pkl
```

### Refresh the data

```bash
# Ingest the Kaggle CSVs into SQLite
python -m src.scripts.ingest_historical

# Run the ETL pipeline (full or per-job)
python -m src.data_pipeline.pipeline --all
python -m src.data_pipeline.pipeline --fixtures   # or --results / --odds / --stats

# Run the scheduler (refreshes on a timer, blocks until Ctrl+C)
python -m src.scheduler
```

### Notebooks

```bash
jupyter notebook notebooks/
```

---

## How prediction works

```text
                         ┌─────────────────────────────────────────────┐
Match-outcome model      │  Predictor.predict(team1, team2)            │
(HistGradientBoosting)   │      → P(home) / P(draw) / P(away)          │
                         └──────────────┬──────────────────────────────┘
                                        │
                 ┌──────────────────────┼───────────────────────────┐
                 ▼                      ▼                           ▼
          /predict page         Group simulation            Knockout bracket
                                (xPts ranking)        (winners advance; draws →
                                                       biased coin flip, bias 0.5)


Top-down title odds        Leaders model (form + Elo + pedigree)  ── weight 0.69 ─┐
(Favourites page)                                                                 ├─► Ensemble ─► ranking
                           Winner-profile model (champion DNA + host) ─ weight 0.31 ┘
```

- **Group stage** ranks teams by expected points: `xPts = 3·P(win) + 1·P(draw)`.
- **Knockout** ties can't end in a draw — when the model's top outcome is a draw,
  the winner is decided by a coin flip biased toward the higher win probability
  (`COIN_BIAS = 0.5`).
- **Host advantage**: co-hosts USA, Canada and Mexico get home advantage only when
  playing in their own country.

---

## Environment Variables

| Variable | Description | Source |
|---|---|---|
| `FOOTBALL_DATA_API_KEY` | Match results and fixtures | [football-data.org](https://www.football-data.org/) |
| `ODDS_API_KEY` | Betting odds for upcoming matches | [the-odds-api.com](https://the-odds-api.com/) |

---

## Automated data refresh

`.github/workflows/update_data.yml` runs the pipeline twice a day (06:00 and
18:00 UTC) and on manual dispatch, committing refreshed data back to the repo.
API keys are provided through GitHub Actions secrets.

---

## Contributing

Pull requests are welcome. For larger changes, please open an issue first.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is open source. See [LICENSE](https://polyformproject.org/licenses/noncommercial/1.0.0) for details.
