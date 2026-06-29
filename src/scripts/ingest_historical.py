"""
Ingest all Kaggle datasets into SQLite.

  results.csv              matches_historical for ML training
  wc_2026_fixtures.csv     matches_wc2026 (fixtures + results)
  wc_2026_teams.csv        teams (group, rank, coach, confederation)
  elo_ratings_wc2026.csv   teams.elo_rating
  train.csv + test.csv     teams.squad_avg_age + market_value

Run:
    python -m src.ingest_historical
"""

import pathlib
import pandas as pd
from src.data_pipeline.db import (
    get_db, init_db, upsert_matches_historical,
    upsert_wc2026_match, upsert_team, log_pipeline_run,
)
from src.utils.team_names import normalize, normalize_all_team_columns

RAW_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "raw"


def ingest_international_results() -> int:
    """results.csv into matches_historical"""
    path = RAW_DIR / "results.csv"
    if not path.exists():
        print("results.csv not found")
        return 0

    print(f"\n Dataset 1: {path.name}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df = df.dropna(subset=["date", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df["neutral"] = df["neutral"].map({True: 1, False: 0, "TRUE": 1, "FALSE": 0}).fillna(0).astype(int)

    normalize_all_team_columns(df)
    print(f"Loaded {len(df):,} matches")

    with get_db() as conn:
        n = upsert_matches_historical(conn, df)
        log_pipeline_run(conn, "ingest_results_csv", "SUCCESS", n, path.name)

    print(f"Inserted {n:,} rows into matches_historical")
    return n


def ingest_wc2026_fixtures() -> int:
    """wc_2026_fixtures.csv into matches_wc2026"""
    path = RAW_DIR / "wc_2026_fixtures.csv"
    if not path.exists():
        print("wc_2026_fixtures.csv not found")
        return 0

    print(f"\n Dataset 2a: {path.name}")

    df = pd.read_csv(path)
    count = 0
    with get_db() as conn:
        for _, row in df.iterrows():
            match = {
                "api_id": None,
                "date": pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                "stage": row["stage"],
                "group_name": row["group"],
                "home_team": normalize(str(row["team1"]).strip()),
                "away_team": normalize(str(row["team2"]).strip()),
                "home_score": None,
                "away_score": None,
                "status": "SCHEDULED",
                "venue": row["venue"],
                "city": row["city"],
                "country": row["country"],
            }
            upsert_wc2026_match(conn, match)
            count += 1

        log_pipeline_run(conn, "ingest_wc2026_fixtures", "SUCCESS", count, path.name)

    print(f"Loaded {count} fixtures into matches_wc2026")
    return count


def ingest_wc2026_teams() -> int:
    """wc_2026_teams.csv into teams"""
    path = RAW_DIR / "wc_2026_teams.csv"
    if not path.exists():
        print("wc_2026_teams.csv not found")
        return 0

    print(f"\n Dataset 2b: {path.name}")

    df = pd.read_csv(path)
    count = 0
    with get_db() as conn:
        for _, row in df.iterrows():
            upsert_team(conn, {
                "name": normalize(str(row["team"]).strip()),
                "confederation": row["confederation"],
                "fifa_ranking": int(row["fifa_rank"]),
                "wc_group": row["group"],
                "coach": row["coach"],
            })
            count += 1

        log_pipeline_run(conn, "ingest_wc2026_teams", "SUCCESS", count, path.name)

    print(f"Loaded {count} teams")
    return count


def ingest_elo_ratings() -> int:
    """elo_ratings_wc2026.csv into patches teams.elo_rating"""
    path = RAW_DIR / "elo_ratings_wc2026.csv"
    if not path.exists():
        print("elo_ratings_wc2026.csv not found")
        return 0

    print(f"\n Dataset 3: {path.name}")

    df = pd.read_csv(path)

    # Take the latest snapshot per team
    df = df.sort_values("snapshot_date").groupby("country").tail(1)

    count = 0
    with get_db() as conn:
        for _, row in df.iterrows():
            name = normalize(str(row["country"]).strip())
            elo = float(row["rating"])

            existing = conn.execute(
                "SELECT id FROM teams WHERE name = ?", (name,)
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE teams SET elo_rating = ?, updated_at = datetime('now') WHERE name = ?",
                    (elo, name),
                )
            else:
                upsert_team(conn, {"name": name, "elo_rating": elo})
            count += 1

        log_pipeline_run(conn, "ingest_elo_ratings", "SUCCESS", count, path.name)

    print(f" Patched Elo ratings for {count} teams")
    return count


def ingest_squad_data() -> int:
    """train.csv + test.csv into teams.squad_avg_age + market_value"""
    train_path = RAW_DIR / "train.csv"
    test_path = RAW_DIR / "test.csv"

    files = [p for p in [train_path, test_path] if p.exists()]
    if not files:
        print("train.csv / test.csv not found")
        return 0

    print(f"\n Dataset 4: {', '.join(p.name for p in files)}")

    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    print(f"  Combined {len(df)} rows from {len(files)} file(s)")

    # Take the latest edition (version) per team
    df = df.sort_values("version", ascending=False).drop_duplicates(
        subset=["team"], keep="first"
    )

    count = 0
    with get_db() as conn:
        for _, row in df.iterrows():
            name = normalize(str(row["team"]).strip())

            age = float(row["squad_avg_age"]) if pd.notna(row["squad_avg_age"]) else None
            value = float(row["squad_total_market_value_eur"]) if pd.notna(row["squad_total_market_value_eur"]) else None

            if age is None and value is None:
                continue

            upsert_team(conn, {
                "name": name,
                "squad_avg_age": age,
                "market_value": value,
            })
            count += 1

        log_pipeline_run(conn, "ingest_squad_data", "SUCCESS", count,
                         ", ".join(p.name for p in files))

    print(f"Patched squad data for {count} teams")
    return count


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ingest_all():
    """Run the full ingestion pipeline in the correct order."""
    print("=" * 60)
    print("Kaggle Data Ingestion")
    print("=" * 60)
    print(f"\n Looking for CSVs in: {RAW_DIR}/")

    csvs = sorted(RAW_DIR.glob("*.csv"))
    if not csvs:
        print("\n No CSV files found in data/raw/")
        return

    print(f"Found {len(csvs)} CSV(s):")
    for f in csvs:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name:40s} ({size_mb:.1f} MB)")

    init_db()

    ingest_international_results()
    ingest_wc2026_fixtures()
    ingest_wc2026_teams()
    ingest_elo_ratings()
    ingest_squad_data()

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    with get_db() as conn:
        for table in ["matches_historical", "matches_wc2026", "teams"]:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  {table:25s} {count:>8,} rows")

        total_teams = conn.execute("SELECT COUNT(*) FROM teams").fetchone()[0]
        teams_with_elo = conn.execute(
            "SELECT COUNT(*) FROM teams WHERE elo_rating IS NOT NULL"
        ).fetchone()[0]
        teams_with_squad = conn.execute(
            "SELECT COUNT(*) FROM teams WHERE squad_avg_age IS NOT NULL OR market_value IS NOT NULL"
        ).fetchone()[0]
        print(f"Teams with Elo ratings:  {teams_with_elo}/{total_teams}")
        print(f"Teams with squad data:   {teams_with_squad}/{total_teams}")

    print(f"\n Database ready at data/football.db")


if __name__ == "__main__":
    ingest_all()
