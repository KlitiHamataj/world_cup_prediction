"""
Create a .env file in the project root (git-ignored) with:

    FOOTBALL_DATA_API_KEY=your_key_here
    ODDS_API_KEY=your_key_here
"""

import os
import pathlib

ENV_PATH = pathlib.Path(__file__).resolve().parent.parent / ".env"


def _load_dotenv():
    """Minimal .env loader (no dependency needed)."""
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value


def get_config() -> dict:
    """Return a dict of all config values."""
    _load_dotenv()
    return {
        "football_data_api_key": os.environ.get("FOOTBALL_DATA_API_KEY"),
        "odds_api_key": os.environ.get("ODDS_API_KEY"),
    }
