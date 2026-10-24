"""
Which venue market is which CFBD game, and which of its listings is the home team's.

Kalshi lists a game as an event with one market per team (`yes_sub_title` names the school; the event ticker carries
the US Eastern date). Polymarket US lists it as an event whose full-game-winner market has a long and a short side,
each naming its team (`team.name`, the school; `description` is only the nickname). A market is matched to the game
whose two schools both of its names fit, on the same Eastern date for Kalshi and within two hours of kickoff for
Polymarket US, and oriented only when the names tell home from away by a clear margin: two "Tigers" never decide which
is LSU. Names are compared as spelled after folding accents, punctuation and the venues' known spellings of CFBD's
schools (ALIASES), never by a shared word: "Tennessee Tech" and "SE Louisiana" (FCS) matched Tennessee and Louisiana's
FBS games on Oct 24 when one word was enough. A game two markets of one venue claim, and anything else unclear, is left
out, never guessed.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from zoneinfo import ZoneInfo

import pandas as pd

MIN_TEAM_SIM = 0.85
MIN_RUNNER_UP_GAP = 0.1
MIN_ORIENTATION_MARGIN = 0.2
KICKOFF_TOLERANCE = pd.Timedelta(hours=2)
EASTERN = ZoneInfo("America/New_York")
KALSHI_EVENT = re.compile(r"^KXNCAAFGAME-(\d{2})([A-Z]{3})(\d{2})")
MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}
WINNER = "full_game_winner"
LEG_COLUMNS = ["game_id", "venue", "esid_home", "esid_away", "score", "margin"]
# The venues' spellings of CFBD's school names, normalised (norm), mapped to CFBD's.
ALIASES = {
    "appalachian state": "app state", "southern mississippi": "southern miss", "louisiana monroe": "ul monroe",
    "ulm": "ul monroe", "louisiana lafayette": "louisiana", "ul lafayette": "louisiana", "ut san antonio": "utsa",
    "texas san antonio": "utsa", "miami fl": "miami", "miami florida": "miami", "massachusetts": "umass",
    "connecticut": "uconn", "florida international": "fiu", "sam houston state": "sam houston",
    "mississippi": "ole miss", "north carolina state": "nc state", "southern california": "usc", "southern cal": "usc",
    "louisiana state": "lsu", "texas christian": "tcu", "southern methodist": "smu", "brigham young": "byu",
    "central florida": "ucf", "alabama birmingham": "uab", "texas el paso": "utep", "nevada las vegas": "unlv",
    "pitt": "pittsburgh", "middle tennessee state": "middle tennessee", "army west point": "army",
    "miami ohio": "miami oh", "k state": "kansas state", "la tech": "louisiana tech",
}


def norm(name: str) -> str:
    x = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    x = x.replace("&", "and").replace("st.", "state").replace("univ.", "university").replace("-", " ")
    x = x.replace("university at ", "").replace("ualbany", "albany")
    x = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", x)).strip()
    x = x.removeprefix("the ")
    return ALIASES.get(x, x)


def sim(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a or not b:
        return 0.0
    return 1.0 if a == b else SequenceMatcher(None, a, b).ratio()


def _fit(first: str, second: str, home: str, away: str) -> tuple[float, float]:
    """The weaker school's fit with first=home, second=away, and with them swapped: both names must fit."""
    return min(sim(first, home), sim(second, away)), min(sim(first, away), sim(second, home))


def _match(first: str, second: str, candidates: pd.DataFrame) -> tuple[pd.Series, bool, float, float] | None:
    """The candidate game the two names fit, whether `first` is its home team, the fit and the orientation margin."""
    scored = []
    for g in candidates.itertuples():
        straight, swapped = _fit(first, second, g.home_team, g.away_team)
        scored.append((max(straight, swapped), straight - swapped, g.Index))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < MIN_TEAM_SIM:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < MIN_RUNNER_UP_GAP:
        return None
    score, diff, index = scored[0]
    if abs(diff) < MIN_ORIENTATION_MARGIN:
        return None
    return candidates.loc[index], diff > 0, score, abs(diff)


def _eastern_date(kickoff) -> object:
    return pd.Timestamp(kickoff).tz_convert(EASTERN).date()


def _kalshi_date(event_ticker: str):
    m = KALSHI_EVENT.match(event_ticker)
    if m is None or m.group(2) not in MONTHS:
        return None
    yy, mon, dd = m.groups()
    return pd.Timestamp(year=2000 + int(yy), month=MONTHS[mon], day=int(dd)).date()


def kalshi_legs(games: pd.DataFrame, markets: list[dict]) -> pd.DataFrame:
    """Legs for the games among `games` (game_id, kickoff, home_team, away_team) that Kalshi's KXNCAAFGAME markets
    list; esids are the team markets' tickers."""
    dates = games["kickoff"].map(_eastern_date)
    by_event: dict[str, list[dict]] = {}
    for m in markets:
        by_event.setdefault(m["event_ticker"], []).append(m)
    rows = []
    for event, teams in by_event.items():
        day = _kalshi_date(event)
        if day is None or len(teams) != 2:
            continue
        teams = sorted(teams, key=lambda m: m["ticker"])
        hit = _match(teams[0].get("yes_sub_title") or "", teams[1].get("yes_sub_title") or "", games[dates == day])
        if hit is None:
            continue
        game, first_is_home, score, margin = hit
        home, away = (teams[0], teams[1]) if first_is_home else (teams[1], teams[0])
        rows.append({"game_id": int(game.game_id), "venue": "kalshi", "esid_home": home["ticker"],
                     "esid_away": away["ticker"], "score": score, "margin": margin})
    return _unclaimed_twice(rows)


def _unclaimed_twice(rows: list[dict]) -> pd.DataFrame:
    legs = pd.DataFrame(rows, columns=LEG_COLUMNS)
    return legs[~legs["game_id"].duplicated(keep=False)].reset_index(drop=True)


def _side_name(side: dict) -> str:
    return (side.get("team") or {}).get("name") or side.get("description") or ""


def pmus_legs(games: pd.DataFrame, events: list[dict]) -> pd.DataFrame:
    """Legs for the games among `games` that Polymarket US's CFB events list; esids are "<market slug>:long" and
    ":short", as the registry has them."""
    kickoffs = pd.to_datetime(games["kickoff"], utc=True)
    rows = []
    for e in events:
        if not e.get("startTime"):
            continue
        market = next((m for m in e.get("markets", []) if (m.get("sportsMarketType") or "").endswith(WINNER)), None)
        sides = (market or {}).get("marketSides") or []
        long = next((s for s in sides if s.get("long")), None)
        short = next((s for s in sides if not s.get("long")), None)
        if long is None or short is None:
            continue
        near = games[(kickoffs - pd.Timestamp(e["startTime"])).abs() <= KICKOFF_TOLERANCE]
        hit = _match(_side_name(long), _side_name(short), near)
        if hit is None:
            continue
        game, long_is_home, score, margin = hit
        slug = market["slug"]
        home, away = (f"{slug}:long", f"{slug}:short") if long_is_home else (f"{slug}:short", f"{slug}:long")
        rows.append({"game_id": int(game.game_id), "venue": "polymarket_us", "esid_home": home, "esid_away": away,
                     "score": score, "margin": margin})
    return _unclaimed_twice(rows)
