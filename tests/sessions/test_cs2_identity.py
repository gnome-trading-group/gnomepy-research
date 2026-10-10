import pandas as pd
import pytest

from gnomepy_research.sessions.cs2_prematch.identity import (
    CONFIRMED,
    MIN_CONFIRMATIONS,
    PROVISIONAL,
    empty_aliases,
    learn,
    resolve_match,
    settle,
    sibling_market,
)
from gnomepy_research.sessions.cs2_prematch.polymarket import index_markets

KICK = pd.Timestamp("2026-09-20 15:00", tz="UTC")
NOW = pd.Timestamp("2026-09-20 03:00", tz="UTC")
FLUXO = {"team_a_id": 9996, "team_a_name": "Fluxo", "team_b_id": 4608, "team_b_name": "Natus Vincere",
         "match_time": KICK}


def _markets(*specs):
    rows = []
    for slug, outcomes, kickoff in specs:
        rows.append({"market_slug": slug, "condition_id": "0x" + slug, "outcomes": list(outcomes),
                     "tokens": ["1", "2"], "final": [], "closed": False, "kickoff": kickoff})
    return index_markets(pd.DataFrame(rows))


def test_exact_names_resolve_and_orient_team_a():
    pm = _markets(("cs2-navi-flx-2026-09-20", ("Natus Vincere", "Fluxo"), KICK))
    r = resolve_match(FLUXO, pm, empty_aliases())
    assert r.market_slug == "cs2-navi-flx-2026-09-20"
    assert r.team_a_outcome == 1
    assert r.learned == () and r.reason is None


def test_sponsor_suffix_is_learned_from_the_anchor():
    pm = _markets(("cs2-flx-navi-2026-09-20", ("Fluxo W7M", "Natus Vincere"), KICK + pd.Timedelta(minutes=30)))
    r = resolve_match(FLUXO, pm, empty_aliases())
    assert r.market_slug == "cs2-flx-navi-2026-09-20" and r.team_a_outcome == 0
    assert r.learned == (("fluxow7m", 9996),)


def test_learned_alias_then_resolves_directly():
    pm = _markets(("cs2-flx-navi-2026-09-20", ("Fluxo W7M", "Natus Vincere"), KICK))
    aliases = learn(empty_aliases(), (("fluxow7m", 9996),), NOW)
    r = resolve_match(FLUXO, pm, aliases)
    assert r.market_slug is not None and r.learned == ()


def test_kickoff_outside_tolerance_is_another_meeting():
    pm = _markets(("cs2-flx-navi-2026-09-20", ("Fluxo W7M", "Natus Vincere"), KICK + pd.Timedelta(hours=3)))
    assert resolve_match(FLUXO, pm, empty_aliases()).reason == "no market"


def test_two_anchored_candidates_are_refused():
    pm = _markets(("cs2-flx-navi-2026-09-20", ("Fluxo W7M", "Natus Vincere"), KICK),
                  ("cs2-xyz-navi-2026-09-20", ("Some Other", "Natus Vincere"), KICK))
    r = resolve_match(FLUXO, pm, empty_aliases())
    assert r.market_slug is None and r.reason.startswith("ambiguous")


def test_alias_to_a_third_team_rules_the_market_out():
    pm = _markets(("cs2-flx-navi-2026-09-20", ("Fluxo W7M", "Natus Vincere"), KICK))
    aliases = learn(empty_aliases(), (("fluxow7m", 1234),), NOW)
    assert resolve_match(FLUXO, pm, aliases).market_slug is None


def test_confirmed_alias_beats_provisional():
    pm = _markets(("cs2-flx-navi-2026-09-20", ("Fluxo W7M", "Natus Vincere"), KICK))
    aliases = learn(empty_aliases(), (("fluxow7m", 1234),), NOW)
    aliases = learn(aliases, (("fluxow7m", 9996),), NOW)
    for _ in range(MIN_CONFIRMATIONS):
        aliases = settle(aliases, "polymarket", "fluxow7m", 9996, agrees=True)
    assert resolve_match(FLUXO, pm, aliases).market_slug == "cs2-flx-navi-2026-09-20"


def test_settlement_confirms_or_removes():
    aliases = learn(empty_aliases(), (("fluxow7m", 9996), ("nip", 4411)), NOW)
    assert set(aliases.status) == {PROVISIONAL}
    aliases = settle(aliases, "polymarket", "fluxow7m", 9996, agrees=True)
    assert aliases.status.iloc[0] == PROVISIONAL, "one agreeing result is a coin flip, not confirmation"
    aliases = settle(aliases, "polymarket", "fluxow7m", 9996, agrees=True)
    aliases = settle(aliases, "polymarket", "nip", 4411, agrees=False)
    assert aliases.source_name_norm.tolist() == ["fluxow7m"]
    assert aliases.status.iloc[0] == CONFIRMED and aliases.evidence_count.iloc[0] == 2


def test_relearning_an_alias_does_not_duplicate_it():
    aliases = learn(empty_aliases(), (("fluxow7m", 9996),), NOW)
    aliases = learn(aliases, (("fluxow7m", 9996),), NOW + pd.Timedelta(days=1))
    assert len(aliases) == 1 and aliases.last_seen.iloc[0] == NOW + pd.Timedelta(days=1)


def test_map_market_is_found_from_the_series_slug():
    pm = _markets(("cs2-flx-navi-2026-09-20", ("Fluxo W7M", "Natus Vincere"), KICK),
                  ("cs2-flx-navi-2026-09-20-game1", ("Fluxo W7M", "Natus Vincere"), KICK))
    assert sibling_market(pm, "cs2-flx-navi-2026-09-20", "game1") == "cs2-flx-navi-2026-09-20-game1"
    assert sibling_market(pm, "cs2-flx-navi-2026-09-20", "game2") is None


def test_this_matchs_own_team_names_beat_an_alias():
    """HLTV reuses "ex-" names; an alias for another ex-RUSTEC roster must not hijack this one."""
    match = {"team_a_id": 13001, "team_a_name": "ex-RUSTEC", "team_b_id": 9001, "team_b_name": "SAW Youngsters",
             "match_time": KICK}
    pm = _markets(("cs2-exr-saw-2026-09-20", ("ex-RUSTEC", "SAW Youngsters"), KICK))
    aliases = learn(empty_aliases(), (("exrustec", 12000),), NOW)
    assert resolve_match(match, pm, aliases).market_slug == "cs2-exr-saw-2026-09-20"
