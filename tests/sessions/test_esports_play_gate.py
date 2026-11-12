"""The arb sessions' play gate: no pairs before the match is live, exit once it is over or never starts, completion
never held back, quotes pulled near a decided result; and the live feeds' parsing (game_state.py)."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from gnomepy import OrderStatus
from gnomepy_research.sessions.esports_cross_venue_arb import game_state as gs
from gnomepy_research.sessions.esports_cross_venue_arb import strategy as mod
from gnomepy_research.sessions.esports_cross_venue_arb__pm_maker import strategy as maker_mod
from tests.sessions.conftest import FakePositions, FakeStreams
from tests.sessions.test_esports_cross_venue_arb import (
    CHEAP_B,
    LEGS,
    LISTINGS,
    QUIET,
    RATES,
    SEC,
    _Registry,
    feed,
    report,
)
from tests.sessions.test_esports_cross_venue_arb_pm_maker import BOOKS, quotes

START = "2026-11-12T16:00:00+00:00"
START_NS = 1_794_499_200 * SEC  # 2026-11-12T16:00:00Z
NOT_STARTED = [START_NS - 600 * SEC, gs.NOT_STARTED, "polymarket"]
LIVE = [START_NS + 60 * SEC, gs.LIVE, "polymarket"]
ENDED = [START_NS + 7200 * SEC, gs.ENDED, "kalshi"]


def _session(cls=mod.EsportsCrossVenueArb, **kwargs):
    kwargs.setdefault("taker_delay_seconds", {})
    kwargs.setdefault("legs", LEGS)
    kwargs.setdefault("scheduled_start", START)
    with patch.object(mod, "RegistryClient", lambda *a, **k: _Registry()):
        s = cls(**kwargs)
    s._position_view = FakePositions(fee_rates=RATES)
    s._streams = FakeStreams(LISTINGS)
    s.exits = []
    s.request_exit = lambda reason, **kw: s.exits.append(reason) or True
    return s


def test_no_pair_before_the_match_is_live_then_pairs_once_it_is():
    s = _session(game_states=[NOT_STARTED, LIVE])
    assert feed(s, QUIET + [CHEAP_B], t=START_NS) == []
    assert len(feed(s, [CHEAP_B], t=START_NS + 61 * SEC)) == 2
    assert s.exits == []


def test_ends_once_the_match_is_over_and_fills_racing_the_end_have_had_time():
    s = _session(game_states=[NOT_STARTED, LIVE, ENDED])
    feed(s, QUIET, t=ENDED[0] + 1 * SEC)
    assert feed(s, [CHEAP_B], t=ENDED[0] + 2 * SEC) == [] and s.exits == [], "no new pair, and not gone yet"
    feed(s, [CHEAP_B], t=ENDED[0] + 31 * SEC)
    assert s.exits == ["the match is over (kalshi)"]


def test_after_the_end_held_legs_still_go_out_and_a_gap_is_completed_before_leaving():
    s = _session(game_states=[LIVE, ENDED], taker_delay_seconds={"polymarket": 1.0})
    sent = feed(s, QUIET + [CHEAP_B], t=ENDED[0] - 1 * SEC)
    assert len(sent) == 1, "the Polymarket US leg goes now, Polymarket International's is held for its delay"
    held = feed(s, [CHEAP_B], t=ENDED[0] + 1 * SEC)
    assert len(held) == 1 and s.exits == [], "the held leg is sent though the match has ended"
    s.positions.set(LISTINGS[302], 30 * 10**9)
    s.on_execution_report(report(302, OrderStatus.FILLED, t=ENDED[0] + 40 * SEC))
    completion = s.on_execution_report(report(101, OrderStatus.CANCELED, t=ENDED[0] + 40 * SEC))
    assert completion and s.exits == [], "an unequal position is completed before the session leaves"


def test_leaves_ended_max_seconds_after_the_end_whatever_it_holds():
    s = _session(game_states=[LIVE, ENDED], ended_max_seconds=600)
    s.positions.set(LISTINGS[302], 30 * 10**9)
    feed(s, QUIET, t=ENDED[0] + 601 * SEC)
    assert s.exits == ["the match is over (kalshi); left after ended_max_seconds"]


def test_a_match_that_was_live_is_never_given_up_on():
    blip = [START_NS + 4 * 3600 * SEC, gs.NOT_STARTED, "kalshi"]
    s = _session(game_states=[NOT_STARTED, LIVE, blip], never_live_hours=3)
    feed(s, QUIET, t=START_NS + 3600 * SEC)
    assert feed(s, QUIET + [CHEAP_B], t=START_NS + 5 * 3600 * SEC) == []
    assert s.exits == [], "a feed saying not started after live closes new pairs, it doesn't end the session"


def test_ends_a_match_never_live_past_its_start_plus_never_live_hours():
    s = _session(game_states=[NOT_STARTED], never_live_hours=3)
    assert feed(s, QUIET, t=START_NS + 2 * 3600 * SEC) == [] and s.exits == []
    feed(s, QUIET, t=START_NS + 3 * 3600 * SEC)
    assert s.exits == ["the match wasn't live by its scheduled start plus never_live_hours"]


def test_ends_at_once_when_the_match_ends_without_being_played():
    forfeit = [START_NS - 120 * SEC, gs.ENDED, "polymarket_event"]
    s = _session(game_states=[NOT_STARTED, forfeit], never_live_hours=3)
    assert feed(s, QUIET + [CHEAP_B], t=START_NS) == []
    assert s.exits == ["the match ended without being played (polymarket_event)"]


def test_a_restarted_session_still_completes_what_it_holds_when_it_first_hears_the_end():
    s = _session(game_states=[ENDED], never_live_hours=3)
    s.positions.set(LISTINGS[101], 20 * mod.SIZE_SCALE)
    out = feed(s, QUIET + [(302, [(0.45, 30)], [(0.43, 30)])], t=ENDED[0] + 1 * SEC)
    assert out and s.exits == [], "it completes the gap rather than leaving"


def test_a_match_live_between_two_callbacks_still_counts_as_played():
    watch = gs.ReplayWatch([tuple(NOT_STARTED), tuple(LIVE), tuple(ENDED)])
    assert not watch.was_live(LIVE[0] - 1) and watch.was_live(ENDED[0] + 1)
    live = gs.GameWatch()
    live._set("polymarket", gs.LIVE)
    live._set("polymarket", gs.ENDED)
    assert live.was_live()


def test_polymarkets_event_is_polled_until_it_says_the_match_is_over():
    answers = iter([[{"ended": False, "closed": False}], [{"ended": True, "closed": False}]])
    watch = gs.GameWatch()
    with patch.object(gs, "_get_json", side_effect=lambda url: next(answers)), \
            patch.object(gs, "EVENT_POLL_SECONDS", 0.01):
        watch._poll_polymarket_event("cs2-zerote-nicetr-2026-11-12")
    assert watch.state().status == gs.ENDED and watch.state().source == "polymarket_event"


def test_completion_is_never_held_back_by_the_match_state():
    s = _session(game_states=[NOT_STARTED, LIVE, [START_NS + 300 * SEC, gs.NOT_STARTED, "polymarket"]])
    feed(s, QUIET + [CHEAP_B], t=START_NS + 100 * SEC)
    s.positions.set(LISTINGS[101], 30 * 10**9)
    s.positions.set(LISTINGS[302], 10 * 10**9)
    s.on_execution_report(report(101, OrderStatus.FILLED, t=START_NS + 400 * SEC))
    intents = s.on_execution_report(report(302, OrderStatus.CANCELED, t=START_NS + 400 * SEC))
    assert intents, "the 20 short on team B are bought though the feed no longer says live"


class _QuietWatch:
    sources = 1

    def state(self, now_ns=None):
        return gs.GameState(gs.LIVE, "polymarket", START_NS)

    def heard_ns(self, now_ns=None):
        return START_NS

    def was_live(self, now_ns=None):
        return True

    def close(self):
        pass


def test_a_feed_gone_quiet_stops_new_pairs():
    s = _session(stale_seconds=60)
    s._watch = _QuietWatch()
    assert feed(s, QUIET + [CHEAP_B], t=START_NS + 61 * SEC) == []
    s2 = _session(stale_seconds=60)
    s2._watch = _QuietWatch()
    assert len(feed(s2, QUIET + [CHEAP_B], t=START_NS + 30 * SEC)) == 2


def test_without_a_feed_the_session_trades_at_once():
    s = _session(scheduled_start=None)
    assert s._watch is None
    assert len(feed(s, QUIET + [CHEAP_B])) == 2


def test_the_live_feeds_are_followed_when_the_launcher_names_them():
    with patch.object(gs, "GameWatch") as watch:
        _session(polymarket_slug="dota2-pari-konca-2026-11-12", kalshi_event_ticker="KXDOTA2GAME-26NOV121100KONCAPARI")
    watch.assert_called_once_with("dota2-pari-konca-2026-11-12", "KXDOTA2GAME-26NOV121100KONCAPARI")


# ---- the maker variant -------------------------------------------------------------------------------------------

def test_the_maker_bids_only_while_the_match_is_live():
    s = _session(maker_mod.EsportsCrossVenueArbPMMaker, game_states=[NOT_STARTED, LIVE])
    assert quotes(feed(s, BOOKS, t=START_NS)) == {}
    assert 1101 in quotes(feed(s, BOOKS, t=START_NS + 61 * SEC))


def test_the_maker_pulls_its_bids_for_good_near_a_decided_result():
    s = _session(maker_mod.EsportsCrossVenueArbPMMaker, game_states=[LIVE])
    assert 1101 in quotes(feed(s, BOOKS, t=START_NS + 61 * SEC))
    decided = [(201, [(0.995, 100)], [(0.985, 100)])]
    pulled = feed(s, decided, t=START_NS + 62 * SEC)
    assert pulled and all(i.bid_price is None or i.bid_size is None or i.bid_size == 0 for i in pulled)
    assert quotes(feed(s, BOOKS, t=START_NS + 63 * SEC)) == {}


# ---- game_state ----------------------------------------------------------------------------------------------------

def test_combined_state_is_ended_then_live_then_not_started():
    a, b = gs.GameState(gs.LIVE, "polymarket", 5), gs.GameState(gs.NOT_STARTED, "kalshi", 3)
    assert gs.combine([a, b]) == a
    assert gs.combine([a, gs.GameState(gs.ENDED, "kalshi", 9)]).status == gs.ENDED
    assert gs.combine([]) == gs.GameState()


def test_venue_states_read_as_one_vocabulary():
    assert gs.polymarket_status({"live": True, "ended": False}) == gs.LIVE
    assert gs.polymarket_status({"live": False, "ended": True}) == gs.ENDED
    assert gs.polymarket_status({"status": "not_started", "live": False}) == gs.NOT_STARTED
    assert gs.kalshi_status({"status": "unplayed", "is_live": False}) == gs.NOT_STARTED
    assert gs.kalshi_status({"status": "live", "is_live": True}) == gs.LIVE
    assert gs.kalshi_status({"status": "complete", "is_live": False}) == gs.ENDED


def test_replay_reads_the_state_as_of_each_moment():
    watch = gs.ReplayWatch([tuple(r) for r in (NOT_STARTED, LIVE)])
    assert watch.state(START_NS - 3600 * SEC) == gs.GameState()
    assert watch.state(START_NS).status == gs.NOT_STARTED
    assert watch.state(START_NS + 61 * SEC) == gs.GameState(gs.LIVE, "polymarket", LIVE[0])


def test_freshness_counts_only_sources_that_have_reported_the_match():
    watch = gs.GameWatch()
    watch._set("kalshi", gs.LIVE)
    kalshi_heard = watch.heard_ns()
    watch._set("polymarket", None)  # the socket is alive but has said nothing about this match
    assert watch.heard_ns() == kalshi_heard


class _Socket:
    def __init__(self):
        self.sent = []

    def send(self, text):
        self.sent.append(text)


def test_a_text_heartbeat_is_answered():
    watch, socket = gs.GameWatch(), _Socket()
    watch._on_sports_message("ping", 1793317, socket)
    assert socket.sent == ["pong"] and watch.state().status == gs.UNKNOWN


def test_the_sports_socket_is_read_for_the_matchs_game_only():
    watch = gs.GameWatch()
    watch._on_sports_message('{"gameId": 1, "live": true}', 1793317)
    assert watch.state().status == gs.UNKNOWN and watch.heard_ns() > 0
    watch._on_sports_message('[{"gameId": 1793317, "live": true, "ended": false}]', 1793317)
    assert watch.state() == gs.GameState(gs.LIVE, "polymarket", watch.state().observed_ns)


def test_kalshi_live_data_is_found_by_ticker_and_polled():
    ticker = "KXDOTA2GAME-26NOV121100KONCAPARI"
    answers = {
        f"{gs.KALSHI_API}/milestones?related_event_ticker={ticker}&limit=5":
            {"milestones": [{"id": "m1", "type": "esports_match", "related_event_tickers": [ticker]}]},
        f"{gs.KALSHI_API}/live_data/esports_match/milestone/m1":
            {"live_data": {"details": {"status": "live", "is_live": True}}},
    }
    watch = gs.GameWatch.__new__(gs.GameWatch)
    gs.GameWatch.__init__(watch)
    watch._poll_seconds = 0.01
    with patch.object(gs, "_get_json", side_effect=lambda url: answers[url]):
        polled = []
        original = watch._set

        def stop_after_first(source, status):
            original(source, status)
            polled.append(status)
            watch.close()
        watch._set = stop_after_first
        watch._follow_kalshi(ticker)
    assert polled == [gs.LIVE] and watch.state().status == gs.LIVE


@pytest.mark.parametrize("row", [NOT_STARTED, LIVE, ENDED, [0, gs.ENDED, "polymarket_event"]])
def test_recorded_codes_cover_every_state(row):
    assert row[1] in gs.STATUS_CODES and row[2] in gs.SOURCE_CODES


def test_polymarkets_taker_delay_is_read_from_the_market():
    event = [{"markets": [{"conditionId": "0xabc", "secondsDelay": 3, "sportsMarketType": "child_moneyline"},
                          {"conditionId": "0xdef", "secondsDelay": 1, "sportsMarketType": "moneyline"}]}]
    with patch.object(gs, "_get_json", return_value=event):
        assert gs.polymarket_seconds_delay("cs2-x", "0xabc") == 3
        assert gs.polymarket_seconds_delay("cs2-x", None) == 1
    with patch.object(gs, "_get_json", side_effect=OSError("down")):
        s = _session(taker_delay_seconds=None, polymarket_slug="cs2-x", game_states=[LIVE])
    assert s._delay_ns == {"polymarket": 1_000_000_000}


def test_one_stray_quote_doesnt_take_the_bids_down():
    s = _session(maker_mod.EsportsCrossVenueArbPMMaker, game_states=[LIVE])
    feed(s, BOOKS, t=START_NS + 61 * SEC)
    feed(s, [(301, [(0.56, 100)], [(0.99, 1)])], t=START_NS + 62 * SEC)
    assert not s._decided


def test_the_maker_pulls_its_bids_at_the_end_then_leaves():
    s = _session(maker_mod.EsportsCrossVenueArbPMMaker, game_states=[LIVE, ENDED])
    assert 1101 in quotes(feed(s, BOOKS, t=ENDED[0] - SEC))
    pulled = feed(s, BOOKS[:1], t=ENDED[0] + SEC)
    assert pulled and s.exits == []
    feed(s, BOOKS[:1], t=ENDED[0] + 31 * SEC)
    assert s.exits == ["the match is over (kalshi)"]


def test_a_guard_of_two_cents_leaves_a_97_cent_favourite_quoted():
    s = _session(maker_mod.EsportsCrossVenueArbPMMaker, game_states=[LIVE])
    feed(s, BOOKS, t=START_NS + 61 * SEC)
    feed(s, [(201, [(0.975, 100)], [(0.97, 100)])], t=START_NS + 62 * SEC)
    assert not s._decided
