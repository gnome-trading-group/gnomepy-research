"""CFB pregame launch: matching venue markets to CFBD games, orienting their legs, resolving listings, the cross-check
that drops likely flips, and the direct_launch triggers and launch-queue messages."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import pytest

from gnomepy_research.sessions.cfb_pregame import launch
from gnomepy_research.sessions.cfb_pregame.slate import kalshi_legs, pmus_legs

KICK = pd.Timestamp("2026-10-24T19:30:00Z")
GAMES = pd.DataFrame({
    "game_id": [1, 2, 3],
    "kickoff": [KICK, KICK, pd.Timestamp("2026-10-25T02:15:00Z")],
    "home_team": ["Miami", "Auburn", "San José State"],
    "away_team": ["Pittsburgh", "LSU", "Hawai'i"],
})


def _kalshi(event, teams):
    return [{"ticker": f"{event}-{code}", "event_ticker": event, "yes_sub_title": name} for code, name in teams]


def _pmus(slug, start, long_team, short_team, title=None):
    side = lambda team, long: {"long": long, "description": team.split()[-1], "team": {"name": team}}
    return {"slug": slug, "title": title or f"{long_team} vs. {short_team}", "startTime": start.isoformat(),
            "markets": [{"slug": f"aec-{slug}", "sportsMarketType": "football_team_full_game_winner",
                         "marketSides": [side(long_team, True), side(short_team, False)]},
                        {"slug": f"spread-{slug}", "sportsMarketType": "football_team_spread", "marketSides": []}]}


def test_kalshi_legs_orient_by_team_name():
    markets = _kalshi("KXNCAAFGAME-26OCT24PITTMIA", [("MIA", "Miami (FL)"), ("PITT", "Pittsburgh")])
    legs = kalshi_legs(GAMES, markets).set_index("game_id")
    assert legs.loc[1, "esid_home"] == "KXNCAAFGAME-26OCT24PITTMIA-MIA"
    assert legs.loc[1, "esid_away"] == "KXNCAAFGAME-26OCT24PITTMIA-PITT"
    assert legs.loc[1, "venue"] == "kalshi"


def test_kalshi_dates_are_eastern_so_a_late_game_keeps_its_saturday_ticker():
    markets = _kalshi("KXNCAAFGAME-26OCT24HAWSJSU", [("SJSU", "San Jose State"), ("HAW", "Hawaii")])
    legs = kalshi_legs(GAMES, markets).set_index("game_id")
    assert legs.loc[3, "esid_home"].endswith("-SJSU"), "02:15 UTC Sunday is 22:15 ET Saturday"


def test_pmus_long_side_orients_by_team_name():
    legs = pmus_legs(GAMES, [_pmus("cfb-pitt-mia-2026-10-24", KICK, "Pittsburgh", "Miami (FL)")]).set_index("game_id")
    assert legs.loc[1, "esid_home"] == "aec-cfb-pitt-mia-2026-10-24:short", "the long side is Pitt, the away team"
    assert legs.loc[1, "esid_away"] == "aec-cfb-pitt-mia-2026-10-24:long"
    assert legs.loc[1, "venue"] == "polymarket_us"


def test_both_tigers_orient_by_school_not_nickname():
    event = _pmus("cfb-lsu-aub-2026-10-24", KICK, "LSU", "Auburn")
    for side in event["markets"][0]["marketSides"]:
        side["description"] = "Tigers"
    legs = pmus_legs(GAMES, [event]).set_index("game_id")
    assert legs.loc[2, "esid_home"].endswith(":short") and legs.loc[2, "esid_away"].endswith(":long")


def test_a_side_that_names_neither_school_is_ambiguous_and_dropped():
    event = _pmus("cfb-lsu-aub-2026-10-24", KICK, "LSU", "Auburn")
    for side in event["markets"][0]["marketSides"]:
        side["team"] = {"name": "Tigers"}
    assert pmus_legs(GAMES, [event]).empty


def test_unmatched_and_far_kickoff_events_are_ignored():
    far = _pmus("cfb-pitt-mia-2026-10-31", KICK + pd.Timedelta(days=7), "Pittsburgh", "Miami (FL)")
    other = _pmus("cfb-ore-wash-2026-10-24", KICK, "Oregon", "Washington")
    assert pmus_legs(GAMES, [far, other]).empty
    assert kalshi_legs(GAMES, _kalshi("KXNCAAFGAME-26OCT31PITTMIA", [("MIA", "Miami (FL)"), ("PITT", "Pittsburgh")])).empty


LOOKALIKES = pd.DataFrame({
    "game_id": [10, 11, 12],
    "kickoff": [KICK, KICK, KICK],
    "home_team": ["South Carolina", "Southern Miss", "Kansas"],
    "away_team": ["Tennessee", "Louisiana", "Baylor"],
})


@pytest.mark.parametrize("first, second", [
    ("Tennessee Tech", "The Citadel"),
    ("Southeastern Louisiana", "Stephen F. Austin"),
    ("SE Louisiana", "SF Austin"),
    ("Kansas State", "Arizona State"),
])
def test_a_game_between_lookalike_schools_matches_nothing(first, second):
    """Measured Oct 24: FCS games whose schools share a word with an FBS game's matched it on both venues."""
    assert pmus_legs(LOOKALIKES, [_pmus("cfb-x-y-2026-10-24", KICK, first, second)]).empty
    assert kalshi_legs(LOOKALIKES, _kalshi("KXNCAAFGAME-26OCT24XY", [("X", first), ("Y", second)])).empty


def test_venue_spellings_of_cfbd_schools_still_match():
    games = pd.DataFrame({"game_id": [20, 21, 22, 23, 24], "kickoff": [KICK] * 5,
                          "home_team": ["App State", "UL Monroe", "Miami (OH)", "Arizona State", "Louisiana Tech"],
                          "away_team": ["Southern Miss", "UTSA", "Miami", "Kansas State", "Old Dominion"]})
    events = [_pmus("a", KICK, "Southern Mississippi", "Appalachian State"),
              _pmus("b", KICK, "UTSA", "Louisiana-Monroe"),
              _pmus("c", KICK, "Miami (FL)", "Miami (OH)"),
              _pmus("d", KICK, "K-State", "Arizona State"),
              _pmus("e", KICK, "Old Dominion", "LA Tech")]
    legs = pmus_legs(games, events).set_index("game_id")
    assert legs.loc[20, "esid_home"] == "aec-a:short" and legs.loc[21, "esid_home"] == "aec-b:short"
    assert legs.loc[23, "esid_home"] == "aec-d:short" and legs.loc[24, "esid_home"] == "aec-e:short", "Oct 24 spellings"
    assert legs.loc[22, "esid_home"] == "aec-c:short", "Miami (OH) is the home team, never Miami (FL)"


def test_a_game_two_venue_markets_claim_is_left_out():
    events = [_pmus("cfb-pitt-mia-2026-10-24", KICK, "Pittsburgh", "Miami (FL)"),
              _pmus("cfb-pitt-mia-2026-10-24-b", KICK, "Pittsburgh", "Miami (FL)")]
    assert 1 not in set(pmus_legs(GAMES, events)["game_id"])


@dataclass
class _Exchange:
    exchange_name: str
    exchange_id: int


@dataclass
class _Listing:
    listing_id: int


class _Registry:
    def __init__(self, known):
        self.known = known

    def get_exchange(self):
        return [_Exchange("Kalshi", 5), _Exchange("Polymarket (US)", 6)]

    def get_listing(self, *, exchange_id, exchange_security_id):
        lid = self.known.get((exchange_id, exchange_security_id))
        return [_Listing(lid)] if lid else []


def _legs():
    return pd.DataFrame([
        {"game_id": 1, "venue": "kalshi", "esid_home": "K1H", "esid_away": "K1A"},
        {"game_id": 1, "venue": "polymarket_us", "esid_home": "P1:short", "esid_away": "P1:long"},
        {"game_id": 2, "venue": "kalshi", "esid_home": "K2H", "esid_away": "K2A"},
    ])


def test_resolve_adds_listings_and_reports_games_missing_one():
    reg = _Registry({(5, "K1H"): 11, (5, "K1A"): 12, (6, "P1:short"): 21, (6, "P1:long"): 22, (5, "K2H"): 31})
    kept, drops = launch.resolve(_legs(), reg)
    assert kept.set_index(["game_id", "venue"]).loc[(1, "polymarket_us"), ["listing_home", "listing_away"]].tolist() == [21, 22]
    assert len(kept) == 2 and drops == [(2, "kalshi", "K2A not in the registry")]


def _preds(p1=0.70, p2=0.40):
    return pd.DataFrame({"game_id": [1, 2], "p_home": [p1, p2], "kickoff": [KICK, KICK + pd.Timedelta(hours=1)]})


def test_cross_check_drops_disagreeing_venues_and_likely_flips():
    legs = _legs()
    kept, drops = launch.cross_check(legs, {(1, "kalshi"): 0.66, (1, "polymarket_us"): 0.64, (2, "kalshi"): 0.80},
                                     _preds())
    assert set(zip(kept.game_id, kept.venue)) == {(1, "kalshi"), (1, "polymarket_us")}
    assert drops == [(2, "kalshi", "model 0.40 vs market 0.80: likely flipped")]
    kept, drops = launch.cross_check(legs, {(1, "kalshi"): 0.66, (1, "polymarket_us"): 0.34, (2, "kalshi"): 0.42},
                                     _preds())
    assert set(zip(kept.game_id, kept.venue)) == {(2, "kalshi")}
    assert {d[0] for d in drops} == {1} and all("venues disagree" in d[2] for d in drops)


def test_cross_check_drops_a_leg_with_no_mid():
    kept, drops = launch.cross_check(_legs(), {(1, "kalshi"): 0.66, (1, "polymarket_us"): 0.64}, _preds())
    assert drops == [(2, "kalshi", "no two-sided book")]


def _resolved():
    return pd.DataFrame([
        {"game_id": 1, "venue": "kalshi", "listing_home": 11, "listing_away": 12},
        {"game_id": 2, "venue": "kalshi", "listing_home": 31, "listing_away": 32},
        {"game_id": 1, "venue": "polymarket_us", "listing_home": 21, "listing_away": 22},
    ])


def test_triggers_one_per_variant_and_venue_with_paper_mode_and_deadline():
    now = KICK - pd.Timedelta(hours=5)
    out = launch.triggers(_resolved(), _preds(), strategy_ids={"anchor": 7, "raw": 8}, research_commit="abc123",
                          bankroll_usd=10_000.0, now=now)
    assert len(out) == 4
    by = {(t["data"]["strategy_id"], t["venue"]): t for t in out}
    k = by[(7, "kalshi")]
    d = k["data"]
    assert d["mode"] == "paper" and d["strategy_type"] == "python" and d["research_commit"] == "abc123"
    assert d["strategy_class"] == "gnomepy_research.sessions.cfb_pregame.strategy:CFBPregame"
    assert d["instance_type"] == "t3.small" and sorted(d["listings"]) == [11, 12, 31, 32]
    args = d["strategy_args"]
    assert args["anchor_weight"] == 0.25 and args["min_edge"] == 0.01 and args["bankroll_usd"] == 10_000.0
    assert [g["game_id"] for g in args["games"]] == [1, 2]
    assert args["games"][1] == {"game_id": 2, "listing_home": 31, "listing_away": 32,
                                "kickoff": (KICK + pd.Timedelta(hours=1)).isoformat()}
    assert k["deadline"] == (KICK + pd.Timedelta(minutes=30)).isoformat(), "the last game's entry cutoff"
    assert by[(8, "kalshi")]["data"]["strategy_args"]["anchor_weight"] == 1.0
    assert by[(8, "kalshi")]["data"]["strategy_args"]["min_edge"] == 0.03


def test_games_inside_the_cutoff_are_left_out():
    now = KICK - pd.Timedelta(minutes=40)
    out = launch.triggers(_resolved(), _preds(), strategy_ids={"anchor": 7, "raw": 8}, research_commit="abc",
                          bankroll_usd=10_000.0, now=now)
    kalshi = [t for t in out if t["venue"] == "kalshi"]
    assert all([g["game_id"] for g in t["data"]["strategy_args"]["games"]] == [2] for t in kalshi)
    assert not [t for t in out if t["venue"] == "polymarket_us"], "its only game closes in 10 minutes"


class _Sqs:
    def __init__(self):
        self.sent = []

    def send_message(self, **kwargs):
        self.sent.append(kwargs)


def test_sqs_message_shape_matches_the_launcher():
    out = launch.triggers(_resolved(), _preds(), strategy_ids={"anchor": 7, "raw": 8}, research_commit="abc",
                          bankroll_usd=10_000.0, now=KICK - pd.Timedelta(hours=5))
    sqs = _Sqs()
    ids = launch.send(out[:1], sqs, "https://queue.fifo", now=KICK - pd.Timedelta(hours=5))
    msg = sqs.sent[0]
    body = json.loads(msg["MessageBody"])
    assert msg["QueueUrl"] == "https://queue.fifo" and msg["MessageGroupId"] == "direct_launch"
    assert msg["MessageDeduplicationId"] == body["trigger_id"] == ids[0]
    assert body["rule_type"] == "direct_launch" and body["action"] == "launch"
    assert body["data"] == out[0]["data"] and body["deadline"] == out[0]["deadline"]
    assert set(body) == {"trigger_id", "rule_type", "action", "timestamp", "data", "deadline"}


@pytest.mark.parametrize("book, mid", [
    ({"orderbook_fp": {"yes_dollars": [["0.40", "10"], ["0.45", "5"]], "no_dollars": [["0.50", "3"]]}}, 0.475),
    ({"orderbook_fp": {"yes_dollars": [["0.45", "5"]], "no_dollars": []}}, None),
])
def test_kalshi_mid_from_bids_on_both_sides(book, mid):
    got = launch.kalshi_mid(book)
    assert (got is None) if mid is None else got == pytest.approx(mid)


def test_pmus_mid_is_in_long_terms():
    body = {"marketData": {"bids": [{"px": {"value": "0.30"}, "qty": "5"}],
                           "offers": [{"px": {"value": "0.34"}, "qty": "5"}]}}
    assert launch.pmus_mid(body) == pytest.approx(0.32)
    assert launch.pmus_mid({"marketData": {"bids": [], "offers": []}}) is None
