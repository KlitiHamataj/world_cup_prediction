"""
Database layer — schema, connection, and helper queries.

Usage:
    from src.data_pipeline.db import get_db, init_db, queries

    # Initialize (creates tables if they don't exist)
    init_db()

    # Get a connection
    with get_db() as conn:
        df = queries.upcoming_matches(conn)
"""

import sqlite3
import pathlib
import pandas as pd
from contextlib import contextmanager

DB_PATH = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "football.db"


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

@contextmanager
def get_db(db_path: str | pathlib.Path = DB_PATH):
    """Start a SQLite connection"""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
-- Historical international match results (Kaggle base dataset)
CREATE TABLE IF NOT EXISTS matches_historical (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    date            TEXT NOT NULL,               -- YYYY-MM-DD
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    home_score      INTEGER,
    away_score      INTEGER,
    tournament      TEXT,
    city            TEXT,
    country         TEXT,
    neutral         INTEGER DEFAULT 0,           -- 0/1 flag
    UNIQUE(date, home_team, away_team)
);

-- World Cup 2026 fixtures & results
CREATE TABLE IF NOT EXISTS matches_wc2026 (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    api_id          TEXT UNIQUE,                  -- external ID from API
    date            TEXT NOT NULL,
    stage           TEXT,                          -- Group A, Round of 16, etc.
    group_name      TEXT,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    home_score      INTEGER,                      -- NULL if not played yet
    away_score      INTEGER,
    status          TEXT DEFAULT 'SCHEDULED',      -- SCHEDULED / LIVE / FINISHED
    venue           TEXT,
    city            TEXT,
    country         TEXT,
    updated_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(date, home_team, away_team)
);

-- Teams metadata
CREATE TABLE IF NOT EXISTS teams (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT UNIQUE NOT NULL,
    fifa_code       TEXT,
    confederation   TEXT,                          -- UEFA, CONMEBOL, etc.
    fifa_ranking     INTEGER,
    elo_rating      REAL,
    wc_group        TEXT,                          -- Group letter for WC2026
    coach           TEXT,
    squad_avg_age   REAL,                          -- average squad age
    market_value    REAL,                          -- total squad market value (millions)
    updated_at      TEXT DEFAULT (datetime('now'))
);

-- Betting odds snapshots
CREATE TABLE IF NOT EXISTS odds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    match_date      TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    source          TEXT,                          -- bookmaker name
    home_win        REAL,
    draw            REAL,
    away_win        REAL,
    scraped_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(match_date, home_team, away_team, source, scraped_at)
);

-- Team recent stats (last N matches, WC campaign stats)
CREATE TABLE IF NOT EXISTS team_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    team            TEXT NOT NULL,
    stat_type       TEXT NOT NULL,                 -- 'recent_form' or 'wc2026'
    matches_played  INTEGER DEFAULT 0,
    wins            INTEGER DEFAULT 0,
    draws           INTEGER DEFAULT 0,
    losses          INTEGER DEFAULT 0,
    goals_for       INTEGER DEFAULT 0,
    goals_against   INTEGER DEFAULT 0,
    clean_sheets    INTEGER DEFAULT 0,
    avg_possession  REAL,
    avg_shots       REAL,
    computed_at     TEXT DEFAULT (datetime('now')),
    UNIQUE(team, stat_type)
);

-- Model predictions (where the DS's model stores its output with the three probabilities home win, draw, away win)
CREATE TABLE IF NOT EXISTS predictions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    match_date      TEXT NOT NULL,
    home_team       TEXT NOT NULL,
    away_team       TEXT NOT NULL,
    prob_home_win   REAL,
    prob_draw       REAL,
    prob_away_win   REAL,
    predicted_outcome TEXT,                        -- HOME / DRAW / AWAY
    model_version   TEXT,
    predicted_at    TEXT DEFAULT (datetime('now')),
    UNIQUE(match_date, home_team, away_team, model_version)
);

-- Pipeline run log (tracks what ran and when)
CREATE TABLE IF NOT EXISTS pipeline_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_name        TEXT NOT NULL,
    status          TEXT NOT NULL,                 -- SUCCESS / FAILED
    rows_affected   INTEGER DEFAULT 0,
    message         TEXT,
    started_at      TEXT,
    finished_at     TEXT DEFAULT (datetime('now'))
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_hist_date ON matches_historical(date);
CREATE INDEX IF NOT EXISTS idx_hist_teams ON matches_historical(home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_wc_date ON matches_wc2026(date);
CREATE INDEX IF NOT EXISTS idx_wc_status ON matches_wc2026(status);
CREATE INDEX IF NOT EXISTS idx_odds_match ON odds(match_date, home_team, away_team);
CREATE INDEX IF NOT EXISTS idx_pred_match ON predictions(match_date, home_team, away_team);
"""


def init_db(db_path: str | pathlib.Path = DB_PATH):
    """Create all tables and indexes."""
    db_path = pathlib.Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_db(db_path) as conn:
        conn.executescript(SCHEMA)
    print(f"✓ Database initialized at {db_path}")


# ---------------------------------------------------------------------------
# Query helpers — what teammates actually call
# ---------------------------------------------------------------------------

class queries:
    """Static namespace of ready-made queries returning DataFrames."""

    @staticmethod
    def upcoming_matches(conn, limit=20) -> pd.DataFrame:
        """Next N unplayed WC2026 matches."""
        sql = """
            SELECT date, stage, group_name, home_team, away_team, venue, city
            FROM matches_wc2026
            WHERE status = 'SCHEDULED'
            ORDER BY date
            LIMIT ?
        """
        return pd.read_sql_query(sql, conn, params=(limit,))

    @staticmethod
    def finished_matches(conn, tournament=None) -> pd.DataFrame:
        """All finished WC2026 matches, optionally filtered by stage."""
        sql = """
            SELECT date, stage, group_name, home_team, away_team,
                   home_score, away_score, venue
            FROM matches_wc2026
            WHERE status = 'FINISHED'
        """
        params = ()
        if tournament:
            sql += " AND stage = ?"
            params = (tournament,)
        sql += " ORDER BY date"
        return pd.read_sql_query(sql, conn, params=params)

    @staticmethod
    def team_recent_form(conn, team: str, n=5) -> pd.DataFrame:
        """Last N matches for a team (from historical + WC2026 combined)."""
        sql = """
            SELECT date, home_team, away_team, home_score, away_score, tournament
            FROM (
                SELECT date, home_team, away_team, home_score, away_score, tournament
                FROM matches_historical
                WHERE (home_team = ? OR away_team = ?)
                  AND home_score IS NOT NULL

                UNION ALL

                SELECT date, home_team, away_team, home_score, away_score, stage AS tournament
                FROM matches_wc2026
                WHERE (home_team = ? OR away_team = ?)
                  AND status = 'FINISHED'
            )
            ORDER BY date DESC
            LIMIT ?
        """
        return pd.read_sql_query(sql, conn, params=(team, team, team, team, n))

    @staticmethod
    def head_to_head(conn, team_a: str, team_b: str, limit=10) -> pd.DataFrame:
        """Historical head-to-head between two teams."""
        sql = """
            SELECT date, home_team, away_team, home_score, away_score, tournament
            FROM matches_historical
            WHERE (home_team = ? AND away_team = ?)
               OR (home_team = ? AND away_team = ?)
            ORDER BY date DESC
            LIMIT ?
        """
        return pd.read_sql_query(
            sql, conn, params=(team_a, team_b, team_b, team_a, limit)
        )

    @staticmethod
    def latest_odds(conn, home_team: str = None, away_team: str = None) -> pd.DataFrame:
        """Latest odds, optionally filtered by match."""
        sql = """
            SELECT o.match_date, o.home_team, o.away_team,
                   o.source, o.home_win, o.draw, o.away_win, o.scraped_at
            FROM odds o
            INNER JOIN (
                SELECT match_date, home_team, away_team, source, MAX(scraped_at) AS max_ts
                FROM odds
                GROUP BY match_date, home_team, away_team, source
            ) latest ON o.match_date = latest.match_date
                    AND o.home_team = latest.home_team
                    AND o.away_team = latest.away_team
                    AND o.source = latest.source
                    AND o.scraped_at = latest.max_ts
        """
        params = []
        if home_team and away_team:
            sql += " WHERE o.home_team = ? AND o.away_team = ?"
            params = [home_team, away_team]
        sql += " ORDER BY o.match_date"
        return pd.read_sql_query(sql, conn, params=params)

    @staticmethod
    def team_info(conn, team: str = None) -> pd.DataFrame:
        """Team metadata. Pass team=None for all teams."""
        sql = "SELECT * FROM teams"
        params = ()
        if team:
            sql += " WHERE name = ?"
            params = (team,)
        sql += " ORDER BY fifa_ranking"
        return pd.read_sql_query(sql, conn, params=params)

    @staticmethod
    def predictions_for_match(conn, home_team: str, away_team: str) -> pd.DataFrame:
        """Get model predictions for a specific match."""
        sql = """
            SELECT * FROM predictions
            WHERE home_team = ? AND away_team = ?
            ORDER BY predicted_at DESC
            LIMIT 1
        """
        return pd.read_sql_query(sql, conn, params=(home_team, away_team))

    @staticmethod
    def standings(conn, group: str = None) -> pd.DataFrame:
        """Compute WC2026 group standings from finished matches."""
        sql = """
            WITH results AS (
                SELECT home_team AS team,
                       CASE WHEN home_score > away_score THEN 3
                            WHEN home_score = away_score THEN 1
                            ELSE 0 END AS pts,
                       home_score AS gf, away_score AS ga
                FROM matches_wc2026
                WHERE status = 'FINISHED' AND group_name IS NOT NULL

                UNION ALL

                SELECT away_team AS team,
                       CASE WHEN away_score > home_score THEN 3
                            WHEN away_score = home_score THEN 1
                            ELSE 0 END AS pts,
                       away_score AS gf, home_score AS ga
                FROM matches_wc2026
                WHERE status = 'FINISHED' AND group_name IS NOT NULL
            )
            SELECT t.wc_group, r.team,
                   COUNT(*) AS played,
                   SUM(CASE WHEN r.pts = 3 THEN 1 ELSE 0 END) AS won,
                   SUM(CASE WHEN r.pts = 1 THEN 1 ELSE 0 END) AS drawn,
                   SUM(CASE WHEN r.pts = 0 THEN 1 ELSE 0 END) AS lost,
                   SUM(r.gf) AS gf,
                   SUM(r.ga) AS ga,
                   SUM(r.gf) - SUM(r.ga) AS gd,
                   SUM(r.pts) AS points
            FROM results r
            LEFT JOIN teams t ON t.name = r.team
            GROUP BY r.team
            ORDER BY t.wc_group, points DESC, gd DESC, gf DESC
        """
        df = pd.read_sql_query(sql, conn)
        if group:
            df = df[df["wc_group"] == group]
        return df

    @staticmethod
    def training_data(conn) -> pd.DataFrame:
        """Full historical dataset for the DS to train on."""
        sql = """
            SELECT h.date, h.home_team, h.away_team,
                   h.home_score, h.away_score, h.tournament,
                   h.neutral,
                   th.fifa_ranking AS home_ranking,
                   th.elo_rating AS home_elo,
                   ta.fifa_ranking AS away_ranking,
                   ta.elo_rating AS away_elo
            FROM matches_historical h
            LEFT JOIN teams th ON th.name = h.home_team
            LEFT JOIN teams ta ON ta.name = h.away_team
            WHERE h.home_score IS NOT NULL
            ORDER BY h.date
        """
        return pd.read_sql_query(sql, conn)


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------

def upsert_matches_historical(conn, df: pd.DataFrame) -> int:
    """Insert historical matches, skipping duplicates. Returns rows inserted."""
    sql = """
        INSERT OR IGNORE INTO matches_historical
            (date, home_team, away_team, home_score, away_score,
             tournament, city, country, neutral)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    rows = df[["date", "home_team", "away_team", "home_score", "away_score",
               "tournament", "city", "country", "neutral"]].values.tolist()
    cursor = conn.executemany(sql, rows)
    return cursor.rowcount


def upsert_wc2026_match(conn, match: dict) -> None:
    """Insert or update a single WC2026 match from API/scraper data."""
    sql = """
        INSERT INTO matches_wc2026
            (api_id, date, stage, group_name, home_team, away_team,
             home_score, away_score, status, venue, city, country, updated_at)
        VALUES (:api_id, :date, :stage, :group_name, :home_team, :away_team,
                :home_score, :away_score, :status, :venue, :city, :country, datetime('now'))
        ON CONFLICT(date, home_team, away_team) DO UPDATE SET
            home_score = excluded.home_score,
            away_score = excluded.away_score,
            status = excluded.status,
            updated_at = datetime('now')
    """
    conn.execute(sql, match)


def upsert_odds(conn, odds_list: list[dict]) -> int:
    """Insert odds snapshots."""
    sql = """
        INSERT OR IGNORE INTO odds
            (match_date, home_team, away_team, source, home_win, draw, away_win)
        VALUES (:match_date, :home_team, :away_team, :source, :home_win, :draw, :away_win)
    """
    cursor = conn.executemany(sql, odds_list)
    return cursor.rowcount


def upsert_team(conn, team: dict) -> None:
    """Insert or update team metadata. Only overwrites fields that are not NULL."""
    # Fill missing keys with None so the query doesn't fail
    defaults = {
        "name": None, "fifa_code": None, "confederation": None,
        "fifa_ranking": None, "elo_rating": None, "wc_group": None,
        "coach": None, "squad_avg_age": None, "market_value": None,
    }
    data = {**defaults, **team}
    sql = """
        INSERT INTO teams (name, fifa_code, confederation, fifa_ranking,
                           elo_rating, wc_group, coach, squad_avg_age,
                           market_value, updated_at)
        VALUES (:name, :fifa_code, :confederation, :fifa_ranking,
                :elo_rating, :wc_group, :coach, :squad_avg_age,
                :market_value, datetime('now'))
        ON CONFLICT(name) DO UPDATE SET
            fifa_code = COALESCE(excluded.fifa_code, teams.fifa_code),
            confederation = COALESCE(excluded.confederation, teams.confederation),
            fifa_ranking = COALESCE(excluded.fifa_ranking, teams.fifa_ranking),
            elo_rating = COALESCE(excluded.elo_rating, teams.elo_rating),
            wc_group = COALESCE(excluded.wc_group, teams.wc_group),
            coach = COALESCE(excluded.coach, teams.coach),
            squad_avg_age = COALESCE(excluded.squad_avg_age, teams.squad_avg_age),
            market_value = COALESCE(excluded.market_value, teams.market_value),
            updated_at = datetime('now')
    """
    conn.execute(sql, data)


def log_pipeline_run(conn, job_name: str, status: str,
                     rows_affected: int = 0, message: str = ""):
    """Log a pipeline execution."""
    conn.execute(
        """INSERT INTO pipeline_log (job_name, status, rows_affected, message, started_at)
           VALUES (?, ?, ?, ?, datetime('now'))""",
        (job_name, status, rows_affected, message),
    )


if __name__ == "__main__":
    init_db()
