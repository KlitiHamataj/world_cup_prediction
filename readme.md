# ⚽ World Cup Prediction

A Python-based machine learning project that fetches live football data, builds predictive models, and serves match outcome predictions through an interactive Streamlit dashboard — with automated data refresh via APScheduler and GitHub Actions.

---

## Overview

This project combines historical match data from Kaggle with live results and betting odds from external APIs to train scikit-learn models capable of predicting World Cup match outcomes. Predictions are surfaced through a Streamlit web app that updates automatically on a schedule.

**Key capabilities:**
- Pulls live match data from [football-data.org](https://www.football-data.org/) and odds from [the-odds-api.com](https://the-odds-api.com/)
- Downloads historical datasets via the Kaggle API
- Stores and queries data from a local SQLite database
- Trains and evaluates ML models for match outcome prediction
- Serves an interactive prediction dashboard with Streamlit
- Automates data ingestion on a schedule using APScheduler
- CI/CD pipeline via GitHub Actions

---

## Project Structure

```
world_cup_prediction/
├── .github/
│   └── workflows/          # GitHub Actions CI/CD pipelines
├── data/
│   └── football.db         # SQLite database (gitignored)
├── notebooks/              # Exploratory data analysis notebooks
├── src/                    # Core Python source modules
├── .gitignore
├── env.example             # API key template — copy to .env
└── requirements.txt
```

---

## Tech Stack

| Layer | Library |
|---|---|
| Data manipulation | pandas ≥ 2.0 |
| Numerical computation | numpy ≥ 1.24 |
| Machine learning | scikit-learn ≥ 1.3 |
| Dashboard | Streamlit ≥ 1.30 |
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

### 4. Configure environment variables

```bash
cp env.example .env
```

Open `.env` and fill in your API keys:

```env
# football-data.org
FOOTBALL_DATA_API_KEY=your_key_here

# the-odds-api.com
ODDS_API_KEY=your_key_here
```

You'll also need a Kaggle API token (`~/.kaggle/kaggle.json`) for dataset downloads. See the [Kaggle docs](https://www.kaggle.com/docs/api) for setup instructions.

---

## Usage

### Run the Streamlit dashboard

```bash
streamlit run src/app.py
```

### Run notebooks

Open the `notebooks/` directory in Jupyter to explore EDA and model development:

```bash
jupyter notebook notebooks/
```

---

## Data Flow

```
Kaggle (historical data)
        │
        ▼
football-data.org API ──► Data ingestion (src/) ──► football.db (SQLite)
        │                                                    │
the-odds-api.com API ────────────────────────────────────────
                                                             │
                                              Feature engineering
                                                             │
                                              ML model (scikit-learn)
                                                             │
                                              Streamlit dashboard
```

1. **Ingestion** — match results, fixtures, and odds are fetched from APIs and stored in SQLite
2. **Feature engineering** — historical stats, form, head-to-head records, and odds-derived features are computed
3. **Model training** — a scikit-learn classifier is trained on the processed features
4. **Prediction** — the dashboard displays predicted outcomes for upcoming matches
5. **Scheduling** — APScheduler (and optionally GitHub Actions) keeps the data fresh automatically

---

## API Server & Usage (`main.py`)

The `main.py` file serves as the entry point for the World Cup 2026 Predictor API. It utilizes FastAPI to expose prediction data via HTTP endpoints and acts as an ensemble aggregator, combining multiple models into a final weighted win probability.

### Run the API Server

Start the server locally using Python:

```bash
python src/main.py
```

*Alternatively, run it directly via Uvicorn:*

```bash
uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

### API Endpoints

Once the server is running (default: `http://127.0.0.1:8000`), you can access the interactive Swagger documentation at `http://127.0.0.1:8000/docs`. 

| Endpoint | Description |
|---|---|
| `GET /api/ensemble` | Returns the final integrated predictions, applying the weighted ensemble formula across all data sources. |
| `GET /api/leaders` | Returns prediction probabilities based solely on 4-year pre-tournament match form and Elo ratings. |
| `GET /api/profiles` | Returns prediction probabilities based solely on historical pedigree, "Champion DNA", and host nation status. |
| `GET /api/bets` | Returns the baseline outright betting probabilities used as a control weight in the ensemble. |

### Ensemble Data Flow

```text
Leaders Pipeline (55% weight) ────────┐
                                      │
Profiles Pipeline (25% weight) ───────┼──► Ensemble Aggregator ──► Final Probabilities ──► FastAPI Endpoints
                                      │
Betting Odds (20% weight) ────────────┘
```

---

## Environment Variables

| Variable | Description | Source |
|---|---|---|
| `FOOTBALL_DATA_API_KEY` | Match results and fixture data | [football-data.org](https://www.football-data.org/) |
| `ODDS_API_KEY` | Betting odds for upcoming matches | [the-odds-api.com](https://the-odds-api.com/) |

---

## Contributing

Pull requests are welcome. For larger changes, please open an issue first to discuss what you'd like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add your feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## License

This project is open source. See [LICENSE](https://polyformproject.org/licenses/noncommercial/1.0.0) for details.