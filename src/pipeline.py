"""
ETL Pipeline.

Orchestrates: scrape → transform → load into SQLite.
Can be run manually or triggered by the scheduler.

Usage:
    # Full refresh
    python -m src.pipeline --all

    # Individual jobs
    python -m src.pipeline --fixtures
    python -m src.pipeline --results
    python -m src.pipeline --odds
    python -m src.pipeline --stats
"""

import argparse
import logging
from datetime import datetime
from src.db import (
    get_db, init_db, upsert_wc2026_match, upsert_odds,
    upsert_team, log_pipeline_run,
)
from src.scraper import FootballDataAPI, OddsScraper
from src.config import get_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def sync_wc2026_fixtures(api: FootballDataAPI) -> int:
    """Fetch all WC2026 fixtures/results and upsert into DB."""
    logger.info("Syncing WC2026 fixtures...")
    try:
        matches = api.get_wc2026_fixtures()
    except Exception as e:
        logger.error(f"Failed to fetch fixtures: {e}")
        with get_db() as conn:
            log_pipeline_run(conn, "sync_fixtures", "FAILED", 0, str(e))
        return 0

    with get_db() as conn:
        count = 0
        skipped = 0
        for m in matches:
            # Skip knockout matches where teams are TBD
            if not m.get("home_team") or m["home_team"] == "Unknown":
                skipped += 1
                continue
            try:
                upsert_wc2026_match(conn, m)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to upsert match {m.get('home_team')} vs {m.get('away_team')}: {e}")

        log_pipeline_run(conn, "sync_fixtures", "SUCCESS", count,
                         f"{count}/{len(matches)} fixtures synced")

    logger.info(f"Synced {count} fixtures")
    return count


def sync_recent_results(api: FootballDataAPI, days_back: int = 3) -> int:
    """Fetch recent results and update match statuses."""
    logger.info(f"Syncing results from last {days_back} days...")
    try:
        results = api.get_recent_results(days_back)
    except Exception as e:
        logger.error(f"Failed to fetch results: {e}")
        with get_db() as conn:
            log_pipeline_run(conn, "sync_results", "FAILED", 0, str(e))
        return 0

    with get_db() as conn:
        count = 0
        for r in results:
            if r["status"] == "FINISHED" and r["home_score"] is not None:
                upsert_wc2026_match(conn, r)
                count += 1
        log_pipeline_run(conn, "sync_results", "SUCCESS", count,
                         f"{count} results updated")

    logger.info(f"Updated {count} match results")
    return count


def sync_odds(odds_api_key: str = None) -> int:
    """Fetch latest odds and store snapshots."""
    logger.info("Syncing odds...")
    scraper = OddsScraper()

    try:
        odds_data = scraper.fetch_odds(api_key=odds_api_key)
    except Exception as e:
        logger.error(f"Failed to fetch odds: {e}")
        with get_db() as conn:
            log_pipeline_run(conn, "sync_odds", "FAILED", 0, str(e))
        return 0

    if not odds_data:
        logger.info("No odds data returned (missing API key or no upcoming matches)")
        return 0

    with get_db() as conn:
        count = upsert_odds(conn, odds_data)
        log_pipeline_run(conn, "sync_odds", "SUCCESS", count,
                         f"{count} odds entries stored")

    logger.info(f"Stored {count} odds entries")
    return count


def sync_team_stats(api: FootballDataAPI) -> int:
    """Update team metadata from the API."""
    logger.info("Syncing team data...")
    try:
        teams = api.get_wc2026_teams()
    except Exception as e:
        logger.error(f"Failed to fetch teams: {e}")
        with get_db() as conn:
            log_pipeline_run(conn, "sync_teams", "FAILED", 0, str(e))
        return 0

    with get_db() as conn:
        for t in teams:
            upsert_team(conn, t)
        log_pipeline_run(conn, "sync_teams", "SUCCESS", len(teams),
                         f"{len(teams)} teams updated")

    logger.info(f"Updated {len(teams)} teams")
    return len(teams)


def compute_team_form_stats() -> int:
    """
    Compute and store recent form stats for all WC2026 teams.
    The DA uses this for the dashboard "last 5 matches" cards.
    """
    import pandas as pd

    logger.info("Computing team form stats...")

    with get_db() as conn:
        teams = pd.read_sql_query("SELECT name FROM teams", conn)
        hist = pd.read_sql_query(
            "SELECT date, home_team, away_team, home_score, away_score FROM matches_historical "
            "WHERE home_score IS NOT NULL ORDER BY date", conn
        )
        wc = pd.read_sql_query(
            "SELECT date, home_team, away_team, home_score, away_score FROM matches_wc2026 "
            "WHERE status='FINISHED'", conn
        )

    all_matches = pd.concat([hist, wc], ignore_index=True)
    all_matches["date"] = pd.to_datetime(all_matches["date"])
    all_matches = all_matches.sort_values("date")

    count = 0
    with get_db() as conn:
        for _, row in teams.iterrows():
            team = row["name"]

            # Get last 5 matches for this team
            mask = (all_matches["home_team"] == team) | (all_matches["away_team"] == team)
            recent = all_matches[mask].tail(5)

            wins, draws, losses, gf, ga = 0, 0, 0, 0, 0
            for _, m in recent.iterrows():
                if m["home_team"] == team:
                    scored, conceded = m["home_score"], m["away_score"]
                else:
                    scored, conceded = m["away_score"], m["home_score"]
                gf += scored
                ga += conceded
                if scored > conceded:
                    wins += 1
                elif scored < conceded:
                    losses += 1
                else:
                    draws += 1

            conn.execute(
                """INSERT OR REPLACE INTO team_stats
                   (team, stat_type, matches_played, wins, draws, losses,
                    goals_for, goals_against, computed_at)
                   VALUES (?, 'recent_form', ?, ?, ?, ?, ?, ?, datetime('now'))""",
                (team, len(recent), wins, draws, losses, gf, ga),
            )
            count += 1
        log_pipeline_run(conn, "compute_form_stats", "SUCCESS", count,
                         f"Form stats for {count} teams")

    logger.info(f"Computed form stats for {count} teams")
    return count


def promote_wc_results() -> int:
    """Copy finished WC2026 matches into matches_historical so training picks them up."""
    logger.info("Promoting finished WC2026 matches to historical...")

    with get_db() as conn:
        cursor = conn.execute("""
            INSERT OR IGNORE INTO matches_historical
                (date, home_team, away_team, home_score, away_score, tournament, neutral)
            SELECT date, home_team, away_team, home_score, away_score, 'FIFA World Cup', 1
            FROM matches_wc2026
            WHERE status = 'FINISHED'
              AND home_score IS NOT NULL
        """)
        count = cursor.rowcount
        log_pipeline_run(conn, "promote_wc_results", "SUCCESS", count,
                         f"{count} WC matches added to historical")

    logger.info(f"Promoted {count} WC2026 results to matches_historical")
    return count


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_pipeline():
    """Run all pipeline jobs in sequence."""
    cfg = get_config()
    init_db()

    api = FootballDataAPI(api_key=cfg.get("football_data_api_key"))

    logger.info("=" * 60)
    logger.info(f"Pipeline started at {datetime.now().isoformat()}")
    logger.info("=" * 60)

    sync_wc2026_fixtures(api)
    sync_recent_results(api, days_back=3)
    promote_wc_results()
    sync_team_stats(api)
    sync_odds(odds_api_key=cfg.get("odds_api_key"))
    compute_team_form_stats()

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Football data pipeline")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--fixtures", action="store_true", help="Sync fixtures only")
    parser.add_argument("--results", action="store_true", help="Sync results only")
    parser.add_argument("--odds", action="store_true", help="Sync odds only")
    parser.add_argument("--stats", action="store_true", help="Compute team stats only")
    args = parser.parse_args()

    cfg = get_config()
    init_db()

    if args.all or not any([args.fixtures, args.results, args.odds, args.stats]):
        run_full_pipeline()
    else:
        api = FootballDataAPI(api_key=cfg.get("football_data_api_key"))
        if args.fixtures:
            sync_wc2026_fixtures(api)
        if args.results:
            sync_recent_results(api)
        if args.odds:
            sync_odds(odds_api_key=cfg.get("odds_api_key"))
        if args.stats:
            compute_team_form_stats()