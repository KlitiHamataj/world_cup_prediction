"""
Scraper / API client.

Two data sources:
  - football-data.org   fixtures, results, standings, teams
  - the-odds-api.com    betting odds from multiple bookmakers

Usage:
    from src.data_pipeline.scraper import FootballDataAPI, OddsScraper

    api = FootballDataAPI(api_key="YOUR_KEY")  # optional for free tier
    fixtures = api.get_wc2026_fixtures()

    odds = OddsScraper()
    odds_data = odds.fetch_odds(api_key="YOUR_KEY")
"""

import time
import logging
import requests
from datetime import datetime, timedelta
from src.utils.team_names import normalize

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# football-data.org client
# ---------------------------------------------------------------------------

class FootballDataAPI:
    """
    Client for football-data.org API.

    Docs: https://www.football-data.org/documentation/api

    The 2026 World Cup competition ID may vary, we search for it dynamically.
    """

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, api_key: str = None):
        self.session = requests.Session()
        if api_key:
            self.session.headers["X-Auth-Token"] = api_key
        self.session.headers["Accept"] = "application/json"
        self._rate_limit_remaining = 10
        self._wc_id = None  # cached competition ID

    def _request(self, endpoint: str, params: dict = None) -> dict:
        """Make a rate-limited request."""
        if self._rate_limit_remaining <= 1:
            logger.info("Rate limit approaching, sleeping 60s...")
            time.sleep(60)

        url = f"{self.BASE_URL}/{endpoint}"
        resp = self.session.get(url, params=params, timeout=15)

        # Track rate limits from headers
        self._rate_limit_remaining = int(
            resp.headers.get("X-Requests-Available-Minute", 10)
        )

        if resp.status_code == 429:
            wait = int(resp.headers.get("X-RequestCounter-Reset", 60))
            logger.warning(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            return self._request(endpoint, params)

        resp.raise_for_status()
        return resp.json()

    # -- Competition discovery --

    def find_wc2026_id(self) -> int:
        """Find the competition ID for the 2026 World Cup."""
        if self._wc_id:
            return self._wc_id

        # WC is usually competition 2000 or 2001 on football-data.org
        # Try the known ID first
        for comp_id in [2000, 2001]:
            try:
                data = self._request(f"competitions/{comp_id}")
                name = data.get("name", "").lower()
                if "world cup" in name:
                    season = data.get("currentSeason", {})
                    start = season.get("startDate", "")
                    if "2026" in start:
                        self._wc_id = comp_id
                        return comp_id
            except requests.HTTPError:
                continue

        # Fallback: list all and search
        data = self._request("competitions")
        for comp in data.get("competitions", []):
            if "world cup" in comp.get("name", "").lower():
                self._wc_id = comp["id"]
                return comp["id"]

        raise ValueError("Could not find WC2026 competition on football-data.org")

    # -- Fixtures & results --

    def get_wc2026_fixtures(self, status: str = None) -> list[dict]:
        """
        Get all WC2026 matches.
        status: SCHEDULED, LIVE, IN_PLAY, PAUSED, FINISHED, POSTPONED, CANCELLED
        """
        comp_id = self.find_wc2026_id()
        params = {}
        if status:
            params["status"] = status
        data = self._request(f"competitions/{comp_id}/matches", params)
        return [self._normalize_match(m) for m in data.get("matches", [])]

    def get_live_matches(self) -> list[dict]:
        """Get currently live matches across all competitions."""
        data = self._request("matches", {"status": "LIVE"})
        return [self._normalize_match(m) for m in data.get("matches", [])]

    def get_match_details(self, match_id: int) -> dict:
        """Get detailed info for a specific match."""
        data = self._request(f"matches/{match_id}")
        return self._normalize_match(data)

    def get_recent_results(self, days_back: int = 3) -> list[dict]:
        """Get recent WC2026 results."""
        comp_id = self.find_wc2026_id()
        date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        date_to = datetime.now().strftime("%Y-%m-%d")
        data = self._request(
            f"competitions/{comp_id}/matches",
            {"dateFrom": date_from, "dateTo": date_to, "status": "FINISHED"},
        )
        return [self._normalize_match(m) for m in data.get("matches", [])]

    # -- Teams & standings --

    def get_wc2026_teams(self) -> list[dict]:
        """Get all teams in the WC2026 competition."""
        comp_id = self.find_wc2026_id()
        data = self._request(f"competitions/{comp_id}/teams")
        teams = []
        for t in data.get("teams", []):
            teams.append({
                "name": normalize(t.get("name")),
                "fifa_code": t.get("tla"),
                "confederation": t.get("area", {}).get("name"),
                "fifa_ranking": None,  # not in this endpoint
                "elo_rating": None,
                "wc_group": None,      # comes from standings
                "coach": t.get("coach", {}).get("name") if t.get("coach") else None,
            })
        return teams

    def get_wc2026_standings(self) -> list[dict]:
        """Get group standings."""
        comp_id = self.find_wc2026_id()
        data = self._request(f"competitions/{comp_id}/standings")
        standings = []
        for group in data.get("standings", []):
            group_name = group.get("group", "")
            for entry in group.get("table", []):
                standings.append({
                    "group": group_name,
                    "position": entry.get("position"),
                    "team": normalize(entry.get("team", {}).get("name")),
                    "played": entry.get("playedGames"),
                    "won": entry.get("won"),
                    "draw": entry.get("draw"),
                    "lost": entry.get("lost"),
                    "goals_for": entry.get("goalsFor"),
                    "goals_against": entry.get("goalsAgainst"),
                    "goal_difference": entry.get("goalDifference"),
                    "points": entry.get("points"),
                })
        return standings

    # -- Normalization --

    def _normalize_match(self, m: dict) -> dict:
        """Normalize a football-data.org match object to our schema."""
        score = m.get("score", {})
        full_time = score.get("fullTime", {})
        home_team = m.get("homeTeam", {})
        away_team = m.get("awayTeam", {})

        # Map API status to our status
        status_map = {
            "TIMED": "SCHEDULED",
            "SCHEDULED": "SCHEDULED",
            "LIVE": "LIVE",
            "IN_PLAY": "LIVE",
            "PAUSED": "LIVE",
            "FINISHED": "FINISHED",
            "POSTPONED": "SCHEDULED",
            "CANCELLED": "CANCELLED",
        }

        return {
            "api_id": str(m.get("id", "")),
            "date": m.get("utcDate", "")[:10],
            "stage": m.get("stage", "").replace("_", " ").title(),
            "group_name": m.get("group", "").replace("_", " ") if m.get("group") else None,
            "home_team": normalize(home_team.get("name", "Unknown")),
            "away_team": normalize(away_team.get("name", "Unknown")),
            "home_score": full_time.get("home"),
            "away_score": full_time.get("away"),
            "status": status_map.get(m.get("status", ""), m.get("status", "")),
            "venue": None,  # not always in this API
            "city": None,
            "country": None,
        }


# ---------------------------------------------------------------------------
# Odds client (the-odds-api.com)
# ---------------------------------------------------------------------------

class OddsScraper:
    """
    Client for the-odds-api.com.
    Returns odds from multiple bookmakers
    (Bet365, Unibet, William Hill, etc.) in clean JSON.
    Docs: https://the-odds-api.com/liveapi/guides/v4/
    """

    def __init__(self):
        self.session = requests.Session()

    def fetch_odds(self, api_key: str = None) -> list[dict]:
        """Fetch World Cup match odds from multiple bookmakers."""
        if not api_key:
            logger.info("No Odds API key provided. Skipping odds fetch.")
            return []

        url = "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds"
        params = {
            "apiKey": api_key,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
        }
        resp = self.session.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            logger.warning(f"Odds API returned {resp.status_code}")
            return []

        data = resp.json()
        results = []
        for event in data:
            home = normalize(event.get("home_team", ""))
            away = normalize(event.get("away_team", ""))
            match_date = event.get("commence_time", "")[:10]

            for bookmaker in event.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market.get("key") == "h2h":
                        outcomes = {o["name"]: o["price"] for o in market.get("outcomes", [])}
                        results.append({
                            "match_date": match_date,
                            "home_team": home,
                            "away_team": away,
                            "source": bookmaker.get("title", "unknown"),
                            "home_win": outcomes.get(event.get("home_team")),
                            "draw": outcomes.get("Draw"),
                            "away_win": outcomes.get(event.get("away_team")),
                        })
        logger.info(f"Fetched {len(results)} odds entries from The Odds API")
        return results
