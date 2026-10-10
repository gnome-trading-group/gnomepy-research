"""
Which Polymarket market is this HLTV match? Resolved through HLTV team ids, not names.

Normalised-name matching linked only 74% of HLTV BO3 series to their market
(Aug-Sep 2026); another 20% had a market whose opponent was spelled differently -
sponsor suffixes ("Fluxo W7M"), short forms ("NIP"), rebrands ("Copenhagen" vs
"EAC") - and were silently dropped. A market outcome name resolves to an HLTV
team id by, in order:

1. the normalised HLTV name of one of the match's two teams;
2. the alias table (confirmed aliases before provisional ones) - second, because
   HLTV reuses names like "ex-RUSTEC" for different rosters, so an alias learned
   for one of them must never override the teams actually on this match's page;
3. anchor + kickoff: the other outcome resolved to one of the teams, the market
   starts within KICKOFF_TOLERANCE of the HLTV match, and no other candidate
   market does - the unresolved name is then the opponent, learned as a
   provisional alias.

Anything still unresolved or ambiguous is skipped with a reason, never guessed.
Settlement later confirms a provisional alias or removes it. One agreeing result
is weak evidence - a wrong mapping still agrees half the time - so confirmation
takes MIN_CONFIRMATIONS agreeing settlements; any disagreement removes it. Kalshi uses the same table with source="kalshi".
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gnomepy_research.sessions.cs2_prematch.polymarket import norm

KICKOFF_TOLERANCE = pd.Timedelta(minutes=60)
ALIAS_COLUMNS = ["source", "source_name_norm", "hltv_team_id", "status", "evidence_count", "first_seen", "last_seen"]
CONFIRMED, PROVISIONAL = "confirmed", "provisional"
MIN_CONFIRMATIONS = 2


def empty_aliases() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(
        ALIAS_COLUMNS, ["object", "object", "int64", "object", "int64", "datetime64[ns, UTC]", "datetime64[ns, UTC]"])})


@dataclass(frozen=True)
class Resolution:
    market_slug: str | None
    team_a_outcome: int | None
    learned: tuple[tuple[str, int], ...] = ()
    reason: str | None = None


def _alias_lookup(aliases: pd.DataFrame, source: str) -> dict[str, int]:
    rows = aliases[aliases.source == source]
    rows = rows.assign(_rank=rows.status.map({CONFIRMED: 0, PROVISIONAL: 1})).sort_values("_rank", kind="stable")
    out: dict[str, int] = {}
    for name, team_id in zip(rows.source_name_norm, rows.hltv_team_id):
        out.setdefault(name, int(team_id))
    return out


def _resolve_name(name_norm: str, team_ids: dict[str, int], lookup: dict[str, int]) -> int | None:
    if name_norm in team_ids:
        return team_ids[name_norm]
    return lookup.get(name_norm)


def resolve_match(match: dict, markets: pd.DataFrame, aliases: pd.DataFrame, source: str = "polymarket",
                  market: str = "series") -> Resolution:
    """
    The market of one kind for one HLTV match.

    match: team_a_id, team_a_name, team_b_id, team_b_name, match_time (UTC).
    markets: polymarket.index_markets output with a kickoff column.
    """
    a_id, b_id = int(match["team_a_id"]), int(match["team_b_id"])
    team_ids = {norm(match["team_a_name"]): a_id, norm(match["team_b_name"]): b_id}
    lookup = _alias_lookup(aliases, source)
    kickoff = pd.Timestamp(match["match_time"])
    pool = markets[(markets.market == market) & markets.kickoff.notna()]
    pool = pool[(pool.kickoff - kickoff).abs() <= KICKOFF_TOLERANCE]

    full, anchored = [], []
    for row in pool.itertuples():
        ids = [_resolve_name(row.o0, team_ids, lookup), _resolve_name(row.o1, team_ids, lookup)]
        known = [i for i in ids if i is not None]
        if any(i not in (a_id, b_id) for i in known):
            continue
        if sorted(known) == sorted([a_id, b_id]):
            full.append((row, ids))
        elif len(known) == 1:
            anchored.append((row, ids))

    if len(full) > 1:
        return Resolution(None, None, reason="ambiguous: several markets resolve to both teams")
    if full:
        row, ids = full[0]
        return Resolution(row.market_slug, ids.index(a_id))
    if not anchored:
        return Resolution(None, None, reason="no market")
    if len(anchored) > 1:
        return Resolution(None, None, reason="ambiguous: several anchored markets")
    row, ids = anchored[0]
    known = next(i for i in ids if i is not None)
    opponent = b_id if known == a_id else a_id
    unresolved = row.o1 if ids[0] is not None else row.o0
    ids = [i if i is not None else opponent for i in ids]
    return Resolution(row.market_slug, ids.index(a_id), learned=((unresolved, opponent),))


def learn(aliases: pd.DataFrame, learned: tuple[tuple[str, int], ...], now: pd.Timestamp,
          source: str = "polymarket") -> pd.DataFrame:
    """Record provisional aliases; seeing an existing alias again only refreshes last_seen."""
    out = aliases.copy()
    for name, team_id in learned:
        hit = (out.source == source) & (out.source_name_norm == name) & (out.hltv_team_id == team_id)
        if hit.any():
            out.loc[hit, "last_seen"] = now
            continue
        new = pd.DataFrame([{"source": source, "source_name_norm": name, "hltv_team_id": int(team_id),
                             "status": PROVISIONAL, "evidence_count": 0, "first_seen": now, "last_seen": now}])
        out = pd.concat([out, new], ignore_index=True) if len(out) else new
    return out.astype({"hltv_team_id": "int64", "evidence_count": "int64"})


def settle(aliases: pd.DataFrame, source: str, name_norm: str, hltv_team_id: int, agrees: bool) -> pd.DataFrame:
    """
    Fold one settled market into an alias: agreement with HLTV's result is evidence,
    disagreement removes the alias outright.
    """
    hit = (aliases.source == source) & (aliases.source_name_norm == name_norm) & (aliases.hltv_team_id == hltv_team_id)
    if not hit.any():
        return aliases
    if not agrees:
        return aliases[~hit].reset_index(drop=True)
    out = aliases.copy()
    out.loc[hit, "evidence_count"] += 1
    out.loc[hit & (out.evidence_count >= MIN_CONFIRMATIONS), "status"] = CONFIRMED
    return out


def sibling_market(markets: pd.DataFrame, series_slug: str, market: str) -> str | None:
    """The map market (game1, game2) of the same meeting as a resolved series market."""
    base = markets[markets.market_slug == series_slug]
    if base.empty:
        return None
    hit = markets[markets.market_slug == f"{series_slug}-game{market[-1]}"]
    return hit.market_slug.iloc[0] if len(hit) == 1 else None
