"""
Whether a match is being played, from the venues' own live feeds, for a session launched before play that must not
trade until it starts (a match never played settles differently by venue) and must leave once it is over.

Three sources, each followed by a daemon thread that swaps in an immutable state, so trading callbacks never wait on
the network:

- Polymarket's sports websocket (wss://sports-api.polymarket.com/ws, public): pushes every game's `live` and `ended`
  by `gameId`; the match's gameId is read from its Polymarket event (gamma, by slug). Measured on CS2 (2026-11-11) at
  11-20 s behind PandaScore's low-latency feed. A forfeit never appears on it.
- Polymarket's event (gamma, by slug), polled every EVENT_POLL_SECONDS until it says the match is over (`ended` or
  `closed`): the only Polymarket signal for a match never played (a forfeit or a cancellation), which the sports
  websocket never sends. Reported as its own source, and only once it says ended.
- Kalshi's live data for the match's milestone (`GET /live_data/esports_match/milestone/{id}`, public, polled): its
  `status` goes unplayed -> live -> complete. The milestone is found by the event ticker.

Combined: ended once any source says so, else live once any says so, else not started once any has answered, else
unknown. A source that fails keeps its last state and retries with backoff (reset once it is answering again);
`heard_ns` is when a source that has reported this match last answered (any answer: Polymarket only pushes a game when
it changes), so the strategy can stop opening positions on a feed gone quiet. Polymarket's documented text heartbeat
(`ping`, answered `pong`) is answered, though the socket hasn't been seen to send one.

Replays read the states a session recorded (`ReplayWatch`), as of each event's time.
"""
from __future__ import annotations

import bisect
import json
import logging
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

import websocket

logger = logging.getLogger(__name__)

UNKNOWN, NOT_STARTED, LIVE, ENDED = "unknown", "not_started", "live", "ended"
STATUS_CODES = {UNKNOWN: 0, NOT_STARTED: 1, LIVE: 2, ENDED: 3}
SOURCE_CODES = {"": 0, "polymarket": 1, "kalshi": 2, "polymarket_event": 3}

SPORTS_WS = "wss://sports-api.polymarket.com/ws"
GAMMA_EVENT = "https://gamma-api.polymarket.com/events?slug={slug}"
KALSHI_API = "https://api.elections.kalshi.com/trade-api/v2"
USER_AGENT = "gnome-session"
TIMEOUT_SECONDS = 15
MAX_BACKOFF_SECONDS = 60
EVENT_POLL_SECONDS = 60


@dataclass(frozen=True)
class GameState:
    status: str = UNKNOWN
    source: str = ""
    observed_ns: int = 0


def _get_json(url: str) -> dict | list:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return json.load(response)


def combine(states: list[GameState]) -> GameState:
    """Ended once any source says so, else live once any does, else not started once any has answered."""
    for status in (ENDED, LIVE, NOT_STARTED):
        found = [s for s in states if s.status == status]
        if found:
            return min(found, key=lambda s: s.observed_ns)
    return GameState()


def polymarket_status(message: dict) -> str:
    if message.get("ended"):
        return ENDED
    return LIVE if message.get("live") else NOT_STARTED


def polymarket_event_ended(event: dict) -> bool:
    return bool(event.get("ended") or event.get("closed"))


def kalshi_status(details: dict) -> str:
    status = (details.get("status") or "").lower()
    if status in ("complete", "completed", "closed", "final"):
        return ENDED
    if details.get("is_live") or status == "live":
        return LIVE
    return NOT_STARTED


def polymarket_seconds_delay(slug: str, condition_id: str | None) -> float | None:
    """How long Polymarket holds a taker order on the event's market with this condition id (its secondsDelay), or on
    its match winner when no id is given; None when the event doesn't say."""
    events = _get_json(GAMMA_EVENT.format(slug=urllib.parse.quote(slug)))
    markets = events[0].get("markets") or [] if events else []
    market = next((m for m in markets if condition_id and m.get("conditionId") == condition_id), None) \
        or next((m for m in markets if m.get("sportsMarketType") == "moneyline"), None)
    return None if market is None or market.get("secondsDelay") is None else float(market["secondsDelay"])


class GameWatch:
    """Follows one match live on the sources it is given; `state()` is the combined state as last heard."""

    def __init__(self, polymarket_slug: str | None = None, kalshi_event_ticker: str | None = None,
                 poll_seconds: float = 3.0):
        self._states: dict[str, GameState] = {}
        self._heard: dict[str, int] = {}
        self._was_live = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._poll_seconds = poll_seconds
        self._threads = []
        if polymarket_slug:
            self._threads.append(threading.Thread(target=self._follow_polymarket, args=(polymarket_slug,),
                                                  name="game-state-polymarket", daemon=True))
            self._threads.append(threading.Thread(target=self._poll_polymarket_event, args=(polymarket_slug,),
                                                  name="game-state-polymarket-event", daemon=True))
        if kalshi_event_ticker:
            self._threads.append(threading.Thread(target=self._follow_kalshi, args=(kalshi_event_ticker,),
                                                  name="game-state-kalshi", daemon=True))
        for thread in self._threads:
            thread.start()

    @property
    def sources(self) -> int:
        return len(self._threads)

    def state(self, now_ns: int | None = None) -> GameState:
        with self._lock:
            return combine(list(self._states.values()))

    def was_live(self, now_ns: int | None = None) -> bool:
        """Whether any source has reported the match live, though the strategy may not have looked while it was."""
        with self._lock:
            return self._was_live

    def heard_ns(self, now_ns: int | None = None) -> int:
        """When a source that has reported the match last answered; any source's answer before one has."""
        with self._lock:
            reporting = [self._heard[s] for s in self._states if s in self._heard]
            return max(reporting or self._heard.values() or [0])

    def close(self) -> None:
        self._stop.set()

    def _set(self, source: str, status: str | None) -> None:
        now = time.time_ns()
        with self._lock:
            self._heard[source] = now
            if status is None:
                return
            self._was_live = self._was_live or status == LIVE
            old = self._states.get(source)
            if old is None or old.status != status:
                self._states[source] = GameState(status, source, now)
                logger.info("game state from %s: %s", source, status)

    def _retry(self, attempt: int) -> None:
        self._stop.wait(min(MAX_BACKOFF_SECONDS, 2 ** attempt))

    # ---- Polymarket ----------------------------------------------------------------------------------------------

    def _follow_polymarket(self, slug: str) -> None:
        attempt = 0
        game_id = None
        while game_id is None and not self._stop.is_set():
            try:
                events = _get_json(GAMMA_EVENT.format(slug=urllib.parse.quote(slug)))
                game_id = events[0].get("gameId") if events else None
                if game_id is None:
                    logger.warning("Polymarket event %s has no gameId yet", slug)
            except Exception:
                logger.warning("Polymarket event %s unreadable", slug, exc_info=True)
            if game_id is None:
                attempt += 1
                self._retry(attempt)
        self._attempt = 0
        while not self._stop.is_set():
            app = websocket.WebSocketApp(
                SPORTS_WS,
                on_open=lambda _: setattr(self, "_attempt", 0),
                on_message=lambda ws, raw: self._on_sports_message(raw, int(game_id), ws),
                on_pong=lambda *_: self._set("polymarket", None),
            )
            app.run_forever(ping_interval=20, ping_timeout=10)
            if self._stop.is_set():
                return
            self._attempt += 1
            logger.warning("Polymarket sports websocket closed; reconnecting")
            self._retry(self._attempt)

    def _poll_polymarket_event(self, slug: str) -> None:
        while not self._stop.is_set():
            try:
                events = _get_json(GAMMA_EVENT.format(slug=urllib.parse.quote(slug)))
                if events and polymarket_event_ended(events[0]):
                    self._set("polymarket_event", ENDED)
                    return
            except Exception:
                logger.warning("Polymarket event %s unreadable", slug, exc_info=True)
            self._stop.wait(EVENT_POLL_SECONDS)

    def _on_sports_message(self, raw: str, game_id: int, ws=None) -> None:
        if raw == "ping":
            if ws is not None:
                ws.send("pong")
            self._set("polymarket", None)
            return
        try:
            message = json.loads(raw)
        except ValueError:
            self._set("polymarket", None)
            return
        for m in message if isinstance(message, list) else [message]:
            if isinstance(m, dict) and m.get("gameId") is not None and int(m["gameId"]) == game_id:
                self._set("polymarket", polymarket_status(m))
                return
        # Any message proves the socket is alive, whichever game it was about.
        self._set("polymarket", None)

    # ---- Kalshi --------------------------------------------------------------------------------------------------

    def _follow_kalshi(self, ticker: str) -> None:
        attempt = 0
        milestone = None
        while milestone is None and not self._stop.is_set():
            try:
                found = _get_json(f"{KALSHI_API}/milestones?related_event_ticker={urllib.parse.quote(ticker)}"
                                  "&limit=5").get("milestones") or []
                milestone = next((m for m in found if m.get("id")
                                  and ticker in (m.get("related_event_tickers") or [])), None)
                if milestone is None:
                    logger.warning("Kalshi has no milestone for %s yet", ticker)
            except Exception:
                logger.warning("Kalshi milestone for %s unreadable", ticker, exc_info=True)
            if milestone is None:
                attempt += 1
                self._retry(attempt)
        attempt = 0
        url = f"{KALSHI_API}/live_data/{milestone.get('type') or 'esports_match'}/milestone/{milestone['id']}"
        while not self._stop.is_set():
            try:
                details = (_get_json(url).get("live_data") or {}).get("details") or {}
                self._set("kalshi", kalshi_status(details))
                attempt = 0
                self._stop.wait(self._poll_seconds)
            except Exception:
                attempt += 1
                logger.warning("Kalshi live data for %s unreadable", ticker, exc_info=True)
                self._retry(attempt)


class ReplayWatch:
    """The states a session recorded, as of each moment of a replay: [(observed_ns, status, source), ...]."""

    def __init__(self, rows: list[tuple[int, str, str]]):
        self._rows = sorted(rows)
        self._at = [r[0] for r in self._rows]
        self.sources = 1

    def state(self, now_ns: int | None = None) -> GameState:
        i = bisect.bisect_right(self._at, now_ns if now_ns is not None else 2 ** 63 - 1)
        if i == 0:
            return GameState()
        at, status, source = self._rows[i - 1]
        return GameState(status, source, at)

    def was_live(self, now_ns: int | None = None) -> bool:
        i = bisect.bisect_right(self._at, now_ns if now_ns is not None else 2 ** 63 - 1)
        return any(status == LIVE for _, status, _ in self._rows[:i])

    def heard_ns(self, now_ns: int | None = None) -> int:
        return now_ns or 0

    def close(self) -> None:
        pass
