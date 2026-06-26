"""
Team name normalization.

Football use different names for the same team.
This maps every known variant to one name so that joins
across datasets always work.

The names follow the main Kaggle results.csv conventions
(the biggest dataset), since that's what the model trains on.

Usage:
    from src.team_names import normalize

    normalize("Korea Republic")   # → "South Korea"
"""

import pandas as pd

# Canonical name - list of known variants
# The key is what we store in the DB. The values are what we might encounter.
_ALIASES: dict[str, list[str]] = {
    "Bolivia": ["Bolivia (Plurinational State of)"],
    "Bosnia and Herzegovina": ["Bosnia-Herzegovina", "Bosnia & Herzegovina", "Bosnia"],
    "Brunei": ["Brunei Darussalam"],
    "Cape Verde": ["Cabo Verde", "Cape Verde Islands"],
    "China PR": ["China", "China People's Republic", "People's Republic of China"],
    "Congo": ["Congo Republic", "Republic of the Congo"],
    "Congo DR": [
        "DR Congo", "Congo Democratic Republic",
        "Democratic Republic of the Congo", "Zaire",
    ],
    "Côte d'Ivoire": ["Ivory Coast", "Cote d'Ivoire", "Cote d Ivoire"],
    "Curaçao": ["Curacao"],
    "Czech Republic": ["Czechia"],
    "East Timor": ["Timor-Leste"],
    "Eswatini": ["Swaziland"],
    "Germany": ["Germany FR", "West Germany"],
    "Indonesia": ["Dutch East Indies"],
    "Iran": ["IR Iran", "Iran (Islamic Republic of)"],
    "Ireland": ["Republic of Ireland"],
    "Kyrgyzstan": ["Kyrgyz Republic"],
    "Laos": ["Lao People's Democratic Republic"],
    "Macau": ["Macao"],
    "Moldova": ["Moldova (Republic of)", "Republic of Moldova"],
    "Myanmar": ["Burma"],
    "Netherlands Antilles": ["Antilles"],
    "North Korea": ["Korea DPR", "DPR Korea", "Democratic People's Republic of Korea"],
    "North Macedonia": ["Macedonia", "FYR Macedonia", "FYROM"],
    "Palestine": ["Palestinian Territory"],
    "Russia": ["Soviet Union", "USSR"],
    "São Tomé and Príncipe": ["Sao Tome and Principe", "São Tomé e Príncipe"],
    "Serbia": ["Serbia and Montenegro", "Yugoslavia", "FR Yugoslavia"],
    "South Korea": [
        "Korea Republic", "Republic of Korea", "Korea, Republic of", "Korea South",
    ],
    "St. Kitts and Nevis": [
        "Saint Kitts and Nevis", "St Kitts and Nevis", "St. Kitts & Nevis",
    ],
    "St. Lucia": ["Saint Lucia", "St Lucia"],
    "St. Vincent and the Grenadines": [
        "Saint Vincent and the Grenadines", "St Vincent and the Grenadines",
        "St. Vincent / Grenadines",
    ],
    "Taiwan": ["Chinese Taipei"],
    "Tanzania": ["Tanzania (United Republic of)"],
    "Trinidad and Tobago": ["Trinidad & Tobago", "T&T"],
    "Turkey": ["Türkiye", "Turkiye"],
    "United States": [
        "USA", "US", "United States of America", "U.S.A.", "U.S.",
    ],
    "Venezuela": ["Venezuela (Bolivarian Republic of)"],
    "Vietnam": ["Viet Nam"],
}

# Build the reverse lookup: variant - canonical
_LOOKUP: dict[str, str] = {}
for canonical, variants in _ALIASES.items():
    for v in variants:
        _LOOKUP[v.strip().lower()] = canonical
    # Also map the canonical name to itself (for case normalization)
    _LOOKUP[canonical.strip().lower()] = canonical


def normalize(name: str) -> str:
    """
    Normalize a team name to its canonical form.
    Returns the original name (stripped) if no mapping is found.
    """
    if not name or not isinstance(name, str):
        return name
    stripped = name.strip()
    return _LOOKUP.get(stripped.lower(), stripped)


def normalize_column(df: pd.DataFrame, column: str = "home_team") -> pd.DataFrame:
    """Normalize a team name column in a DataFrame (in-place)."""
    df[column] = df[column].apply(normalize)
    return df


def normalize_all_team_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize all team-related columns in a DataFrame."""
    for col in df.columns:
        if "team" in col.lower():
            df[col] = df[col].apply(normalize)
    return df