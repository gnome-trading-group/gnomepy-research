"""
Match HLTV matches to Polymarket markets, and markets to registry listings.

The rules are the ones the research backtest had to learn:
- an h2h "event" holds every market between two teams across several meetings,
  so markets are identified by slug, never picked by volume;
- the series moneyline slug ends in the match date, map winners end -gameN;
- which token is team A is read from the market's `outcomes`, not assumed;
- a settled market whose result disagrees with HLTV is a mismatch, not data.

Registry listings carry exchange_security_id = "<conditionId>:<tokenId>". Gamma
returns the conditionId with each market, so each token resolves with one exact
lookup - far sturdier than paging through every Polymarket listing.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import time

import pandas as pd
import requests

from gnomepy.registry import RegistryClient

GAMMA = "https://gamma-api.polymarket.com"
POLYMARKET_EXCHANGE_ID = 4
SLUG = re.compile(r"-(\d{4}-\d{2}-\d{2})(?:-game(\d))?$")
MARKET_OF_GAME = {"0": "series", "1": "game1", "2": "game2"}


def norm(name: str | None) -> str:
    n = (name or "").lower().strip()
    n = re.sub(r"\b(esports|esport|team|gaming|club|academy|nxt|fe)\b", "", n)
    return re.sub(r"[^a-z0-9]", "", n)


def fetch_markets(start: pd.Timestamp, end: pd.Timestamp, session: requests.Session | None = None) -> pd.DataFrame:
    """CS2 head-to-head markets whose events were created between start and end."""
    s = session or requests.Session()
    s.headers.setdefault("User-Agent", "research/1.0")
    rows, off = [], 0
    while off <= 3900:
        r = s.get(f"{GAMMA}/events", params={"limit": 100, "offset": off, "tag_slug": "counter-strike-2",
                                             "start_date_min": start.strftime("%Y-%m-%dT00:00:00Z"),
                                             "start_date_max": end.strftime("%Y-%m-%dT00:00:00Z")}, timeout=30)
        data = r.json() if r.status_code == 200 else []
        if not data:
            break
        for e in data:
            for m in e.get("markets", []):
                rows.append({"market_slug": m.get("slug"), "condition_id": m.get("conditionId"),
                             "outcomes": json.loads(m.get("outcomes") or "[]"),
                             "tokens": json.loads(m.get("clobTokenIds") or "[]"),
                             "final": json.loads(m.get("outcomePrices") or "[]"),
                             "closed": bool(m.get("closed")),
                             "kickoff": _kickoff(m)})
        off += 100
        time.sleep(0.2)
    return index_markets(pd.DataFrame(rows))


def _kickoff(market: dict) -> pd.Timestamp | None:
    """Scheduled start: gameStartTime, else the event start, else endDate - 6h (Polymarket's convention)."""
    for key in ("gameStartTime", "eventStartTime"):
        if market.get(key):
            return pd.Timestamp(market[key]).tz_convert("UTC")
    if market.get("endDate"):
        return pd.Timestamp(market["endDate"]).tz_convert("UTC") - pd.Timedelta(hours=6)
    return None


def index_markets(pm: pd.DataFrame) -> pd.DataFrame:
    pm = pm.drop_duplicates("market_slug").copy()
    ext = pm.market_slug.str.extract(SLUG)
    pm["slug_date"], pm["game"] = ext[0], ext[1].fillna("0")
    pm = pm[pm.slug_date.notna() & pm.game.isin(MARKET_OF_GAME)
            & (pm.outcomes.map(len) == 2) & (pm.tokens.map(len) == 2)].copy()
    pm["slug_date"] = pd.to_datetime(pm.slug_date).dt.date
    pm["market"] = pm.game.map(MARKET_OF_GAME)
    pm["o0"] = [norm(o[0]) for o in pm.outcomes]
    pm["o1"] = [norm(o[1]) for o in pm.outcomes]
    pm["pair"] = [tuple(sorted(p)) for p in zip(pm.o0, pm.o1)]
    return pm


def match_market(pm: pd.DataFrame, team_a: str, team_b: str, match_date: dt.date, market: str,
                 team_a_won: int | None = None) -> tuple[str | None, str | None, str | None, str | None]:
    """
    (condition id, token for team A, token for team B, reason if unmatched) for one market.

    Exact date first, then +/-1 day; more than one candidate is ambiguous (two
    meetings between the same teams) and is refused rather than guessed.
    """
    a, b = norm(team_a), norm(team_b)
    pair = tuple(sorted([a, b]))
    pool = pm[(pm.pair == pair) & (pm.market == market)]
    cands = pool[pool.slug_date == match_date]
    if cands.empty:
        for off in (-1, 1):
            cands = pool[pool.slug_date == match_date + dt.timedelta(days=off)]
            if not cands.empty:
                break
    if cands.empty:
        return None, None, None, "no market"
    if len(cands) > 1:
        return None, None, None, "ambiguous"
    mk = cands.iloc[0]
    a_is_o0 = mk.o0 == a
    fin = [float(x) for x in mk.final] if len(mk.final) == 2 else []
    if team_a_won is not None and fin and max(fin) > 0.9 and int((fin[0] > fin[1]) == a_is_o0) != int(team_a_won):
        return None, None, None, "settlement disagrees with HLTV"
    tok_a, tok_b = (mk.tokens[0], mk.tokens[1]) if a_is_o0 else (mk.tokens[1], mk.tokens[0])
    return mk.get("condition_id"), str(tok_a), str(tok_b), None


def resolve_listing(condition_id: str | None, token: str, registry: RegistryClient,
                    retries: int = 3) -> int | None:
    """Registry listing id for one outcome token, or None if it is not registered."""
    if not condition_id:
        return None
    for attempt in range(retries):
        try:
            hits = registry.get_listing(exchange_security_id=f"{condition_id}:{token}")
            return int(hits[0].listing_id) if hits else None
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(1 + attempt)
    return None
