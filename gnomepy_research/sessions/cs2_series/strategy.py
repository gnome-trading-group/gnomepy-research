"""
CS2 series winner at kickoff, on every venue listing one HLTV match.

gnome-launcher starts one session per match holding each venue's two team listings (Polymarket International,
Kalshi, Polymarket US), so the match is sized and capped once, wherever its liquidity is. The session prices the match
from each cs2_upcoming_features row as the pipeline writes it (match_feed.py): e12-wide-mm, the weekly series model,
prices the two announced fives as q, P(team A wins). On every book update the fair price is the market's own mid
blended with q (MarketBlend, weights fitted on past kickoff prices and outcomes), so the model only moves the market's
price by as much as it has earned.

Inside the entry window - the last window_hours before HLTV's latest kickoff, while HLTV still lists the match as
upcoming - it buys whichever listing, on any venue, is underpriced by at least min_edge after that listing's taker
fee, sized by the Kelly criterion on kelly_fraction of bankroll_usd and never past max_stake_usd across the whole
match. Every book update re-checks the edge and tops the holding up to the Kelly size the current book justifies. An
order walks the ask book only as deep as every share still clears the threshold and the Kelly size; it is a limit at
the deepest level it costed, so it only ever takes liquidity it has already priced. It never holds both teams.

A held position is sold only after the model's price has changed since it last bought (a lineup or stand-in change),
and only into bids whose proceeds after the fee beat the new fair price by min_edge: selling then is worth more than
holding. A move in the market alone never sells, since the market moving is what the blend already prices. Once the
held team is sold out, the other team is bought on the usual rule if it now clears the edge.

The mid is Polymarket International's when it has a two-sided book, since the blend was fitted on its prices, else
the tightest two-sided book elsewhere; team B's book stands in (as 1 - its mid) where team A's is one-sided.

At kickoff (entry_cutoff_hours before it) the session ends itself, keeping what it holds, once no order is in flight;
it also ends early if HLTV lists the match live or drops it. Positions settle in the ledger. A stalled pipeline still
ends the session at the cutoff of the last kickoff it saw.

It does not trade until both lineups are announced, nor on stale match state (the newest cs2_upcoming row older than
max_state_age_minutes: a match that has since started or moved would look upcoming), nor on a model fitted more than
max_model_age_days before now or after it (a replay must not price with a model that knows the future). The model and
blend load once at start, so a session never changes models mid-match.

State that must survive a restart - what it holds, money committed - is read from the engine's positions rather than
kept here, so a restarted session resumes instead of buying again. The model price a position was last bought at is
not kept: a restarted session takes the price it first sees as that reference, so a restart alone never sells.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from gnomepy import ExecutionReport, Intent, OrderStatus, OrderType, Scales, Side, Strategy
from gnomepy.data import ArtifactQuery, ArtifactStore, resolve_artifact_path
from gnomepy.java.schemas import StreamRecord
from gnomepy.registry import RegistryClient

from gnomepy_research.fees import registry_fee_rates
from gnomepy_research.models.cs2.e12.blend import MarketBlend
from gnomepy_research.models.cs2.e12.model import E12Model
from gnomepy_research.sessions.cs2_series.decision import (
    Entry,
    Level,
    Sale,
    cost_per_share,
    evaluate_side,
    kelly_take,
    sell_down,
)
from gnomepy_research.sessions.cs2_series.match_feed import FEATURES, MATCHES, MatchFeed, Quote, lineup_ids

logger = logging.getLogger(__name__)

PRICE_SCALE = Scales.PRICE
SIZE_SCALE = Scales.SIZE
HOUR_NS = 3_600_000_000_000
DAY_NS = 24 * HOUR_NS
TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
UPCOMING = "upcoming"
ARTIFACT = "artifact://"
DEFAULT_MODEL = "artifact://e12_model/cs2_series"
DEFAULT_BLEND = "artifact://blend/cs2_series_e12"
# The blend's weights were fitted on this venue's prices.
MID_VENUE = "polymarket"
BUY, SELL = 1, -1


@dataclass
class _Leg:
    venue: str
    team: int  # 0 = HLTV's team A, 1 = team B
    listing_id: int
    exchange_id: int
    security_id: int
    tick: float
    lot: int
    min_size: int
    asks: list[Level] = field(default_factory=list)
    bids: list[Level] = field(default_factory=list)
    in_flight: bool = False
    last_limit: float = 0.0


def _fetch(uri: str) -> tuple[Path, str]:
    """A local copy of an artifact URI or path, and the exact version it resolved to (an unpinned URI or an alias
    is resolved first)."""
    if uri.startswith(ARTIFACT):
        store = ArtifactStore()
        ref = store.find(ArtifactQuery.parse(uri))
        return Path(store.resolve(ref)), str(ref)
    return Path(resolve_artifact_path(uri)), uri


@dataclass(frozen=True)
class Models:
    model: E12Model
    blend: MarketBlend
    versions: dict[str, str]


def load_models(model_path: str, blend_path: str) -> Models:
    (m, mv), (b, bv) = _fetch(model_path), _fetch(blend_path)
    return Models(E12Model.load(m), MarketBlend.load(b), {"model": mv, "blend": bv})


def _team_id(value) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def _mid(leg: _Leg) -> tuple[float, float] | None:
    """(mid, spread) of a two-sided book."""
    if not leg.asks or not leg.bids:
        return None
    return (leg.asks[0].price + leg.bids[0].price) / 2, leg.asks[0].price - leg.bids[0].price


class CS2Series(Strategy):
    """
    Args:
        match_id: the HLTV match.
        legs: one {venue, listing_team_a, listing_team_b} per venue with a series market for the match: the venue's
            registry listings for each HLTV team winning the series. Set by the launcher.
        bankroll_usd: the money the strategy sizes its bets on one match against.
        model_path: the e12-wide-mm model (cs2_e12_fit, weekly), an artifact URI or path; unpinned means the latest.
        blend_path: the weights blending the market's mid with the model (JSON), likewise.
        max_model_age_days: the oldest model it trades on, counted from the model's cutoff.
        min_edge: the least edge after the fee worth buying, and the least a sale must beat holding by.
        kelly_fraction: the share of the Kelly size to hold; under 1 because the fair price is an estimate, and
            betting full Kelly on an estimate over-bets.
        window_hours: how long before kickoff the entry window opens.
        entry_cutoff_hours: how long before kickoff the window closes and the session ends.
        max_stake_usd: most the match may commit across every venue, fees included, however large the Kelly size.
        min_order_usd: smallest order worth sending.
        max_state_age_minutes: the oldest match state it trades on; the pipeline writes one every 15 minutes.
        reload_minutes: how often to re-read the match's rows (0 = once, for replays).
        near_kickoff_reload_minutes, near_kickoff_hours: re-read this often inside this long before the cutoff,
            so a lineup change, a kickoff HLTV brings forward or a match it lists live reaches the session within a
            minute of the pipeline writing it.
        features_source, matches_source: DatasetStore tables, or parquet extracts of them.
    """

    def __init__(
        self,
        match_id: int,
        legs: list[dict],
        bankroll_usd: float,
        model_path: str = DEFAULT_MODEL,
        blend_path: str = DEFAULT_BLEND,
        max_model_age_days: float = 8.0,
        min_edge: float = 0.02,
        kelly_fraction: float = 0.5,
        window_hours: float = 1.0,
        entry_cutoff_hours: float = 0.0,
        max_stake_usd: float = 250.0,
        min_order_usd: float = 1.0,
        book_depth: int = 10,
        processing_time_ns: int = 0,
        max_state_age_minutes: float = 30.0,
        reload_minutes: float = 5.0,
        near_kickoff_reload_minutes: float = 1.0,
        near_kickoff_hours: float = 1.0,
        features_source: str = FEATURES,
        matches_source: str = MATCHES,
    ):
        self.match_id = int(match_id)
        self._fee_rates = registry_fee_rates(self)
        if bankroll_usd <= 0:
            raise ValueError(f"bankroll_usd must be positive, got {bankroll_usd}")
        if not 0 <= entry_cutoff_hours < window_hours:
            raise ValueError(f"entry_cutoff_hours must be in [0, window_hours), got {entry_cutoff_hours}")
        venues = [leg["venue"] for leg in legs]
        if not venues or len(set(venues)) != len(venues):
            raise ValueError(f"legs must name each venue once, got {venues}")
        self.bankroll_usd = bankroll_usd
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.window_ns = int(window_hours * HOUR_NS)
        self.cutoff_ns = int(entry_cutoff_hours * HOUR_NS)
        self.max_stake_usd = max_stake_usd
        self.min_order_usd = min_order_usd
        self.book_depth = book_depth
        self._processing_time_ns = processing_time_ns
        self.max_state_age_ns = int(max_state_age_minutes * 60 * 1e9)
        self._metrics_buf = None
        self._model_buf = None
        # The model price the position was last bought at; a change from it is what lets the session sell.
        self._entry_q: float | None = None
        self._recorded_quote_at: int | None = None

        registry = RegistryClient()
        self._venues = [(leg["venue"], (self._resolve(registry, leg["venue"], 0, leg["listing_team_a"]),
                                        self._resolve(registry, leg["venue"], 1, leg["listing_team_b"])))
                        for leg in legs]
        self._legs = [leg for _, pair in self._venues for leg in pair]
        self._leg_of = {leg.listing_id: leg for leg in self._legs}

        models = load_models(model_path, blend_path)
        logger.info("match %d: pricing with %s", self.match_id, models.versions)
        self._blend = models.blend
        self._model_cutoff_ns = pd.Timestamp(models.model.cutoff).value
        self.max_model_age_ns = int(max_model_age_days * DAY_NS)
        model = models.model

        def pricer(rows: pd.DataFrame) -> list[float]:
            prices = []
            for r in rows.itertuples():
                a, b = _team_id(r.team_a_id), _team_id(r.team_b_id)
                prices.append(math.nan if a is None or b is None else
                              model.price(a, lineup_ids(r.team_a_lineup_ids), b, lineup_ids(r.team_b_lineup_ids)))
            return prices

        self._feed = MatchFeed(self.match_id, pricer, features_source, matches_source, reload_minutes * 60,
                               near_kickoff_reload_minutes * 60, (near_kickoff_hours + entry_cutoff_hours) * 3600)

    @staticmethod
    def _resolve(registry, venue: str, team: int, listing_id: int) -> _Leg:
        listings = registry.get_listing(listing_id=int(listing_id))
        if not listings:
            raise ValueError(f"listing {listing_id} not found in registry")
        specs = registry.get_listing_spec(listing_id=int(listing_id))
        spec = specs[0] if specs else None
        tick = int(spec.tick_size) / PRICE_SCALE if spec and spec.tick_size else 0.01
        lot = int(spec.lot_size) if spec and spec.lot_size else 1
        min_size = int(spec.min_size) if spec and spec.min_size else 0
        return _Leg(venue=venue, team=team, listing_id=int(listing_id), exchange_id=listings[0].exchange_id,
                    security_id=listings[0].security_id, tick=tick, lot=lot, min_size=min_size)

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def register_metrics(self) -> None:
        buf = self.metrics.create_stream("cs2_series_orders")
        self._m_ts = buf.add_long_column("timestamp")
        self._m_exchange = buf.add_long_column("exchange_id")
        self._m_listing = buf.add_long_column("listing_id")
        self._m_team = buf.add_long_column("team")
        self._m_direction = buf.add_long_column("direction")
        self._m_q = buf.add_double_column("q")
        self._m_mid = buf.add_double_column("mid")
        self._m_fair = buf.add_double_column("fair")
        self._m_best = buf.add_double_column("best_price")
        self._m_spread = buf.add_double_column("spread")
        self._m_edge = buf.add_double_column("edge_at_best")
        self._m_limit = buf.add_double_column("limit_price")
        self._m_shares = buf.add_double_column("shares")
        self._m_usd = buf.add_double_column("usd")
        self._m_kelly_bankroll = buf.add_double_column("kelly_bankroll")
        self._m_held_shares = buf.add_double_column("held_shares")
        buf.freeze()
        self._metrics_buf = buf

        model = self.metrics.create_stream("cs2_series_model")
        self._q_ts = model.add_long_column("timestamp")
        self._q_at = model.add_long_column("generated_at")
        self._q_q = model.add_double_column("q")
        self._q_lineups = model.add_long_column("lineups_hash")
        self._q_known = model.add_long_column("lineup_known")
        self._q_cutoff = model.add_long_column("model_cutoff")
        model.freeze()
        self._model_buf = model

    def _taker_rate(self, leg: _Leg) -> float:
        """The leg's taker rate in the registry; the fee per share is rate * p * (1 - p)."""
        return self._fee_rates(leg.listing_id)[0]

    # ---- state read from the engine's positions --------------------------------------

    def _leg_state(self, leg: _Leg) -> tuple[float, float, float]:
        """(filled shares, shares bought but not yet filled, committed USD incl. fees and pending)."""
        pos = self.positions.get_position(leg.exchange_id, leg.security_id)
        filled = pos.net_quantity / SIZE_SCALE if pos else 0.0
        avg = pos.avg_entry_price / PRICE_SCALE if pos and pos.net_quantity else 0.0
        fees = pos.total_fees / PRICE_SCALE if pos else 0.0
        pending = max(0.0, self.positions.get_effective_quantity(leg.exchange_id, leg.security_id) / SIZE_SCALE - filled)
        return filled, pending, filled * avg + fees + pending * cost_per_share(leg.last_limit, self._taker_rate(leg))

    # ---- engine callbacks ----------------------------------------------------------------

    def on_market_data(self, data: StreamRecord) -> list[Intent]:
        leg = self._leg_of.get(self.streams.listing_id(data.stream_id))
        if leg is None:
            return []
        leg.asks = [Level(data.ask_price(i) / PRICE_SCALE, data.ask_size(i) / SIZE_SCALE)
                    for i in range(self.book_depth) if data.ask_size(i) > 0]
        leg.bids = [Level(data.bid_price(i) / PRICE_SCALE, data.bid_size(i) / SIZE_SCALE)
                    for i in range(self.book_depth) if data.bid_size(i) > 0]
        return self._evaluate(data.event_timestamp)

    def _evaluate(self, now_ns: int) -> list[Intent]:
        state, quote = self._feed.state(now_ns), self._feed.quote(now_ns)
        in_flight = any(leg.in_flight for leg in self._legs)
        if state is not None and (state.status != UPCOMING or now_ns >= state.kickoff_ns - self.cutoff_ns):
            # Waits for an in-flight order's final report, so the session never ends with an order working. Judged
            # on the last state even when it's stale: a stalled pipeline mustn't keep the session up forever.
            if not in_flight:
                self.request_exit(f"match {self.match_id} is {state.status}" if state.status != UPCOMING
                                  else f"match {self.match_id} reached its entry cutoff")
            return []
        if state is None or quote is None:
            return []
        self._record_quote(now_ns, quote)
        if now_ns - state.at_ns > self.max_state_age_ns:
            return []
        if now_ns < state.kickoff_ns - self.window_ns:
            return []
        if not quote.lineup_known or in_flight or math.isnan(quote.q):
            return []
        if not 0 <= now_ns - self._model_cutoff_ns <= self.max_model_age_ns:
            return []
        mid = self._mid_team_a()
        if mid is None:
            return []
        fair_a = self._blend.fair(mid, quote.q)
        fairs = (fair_a, 1.0 - fair_a)

        states = {leg.listing_id: self._leg_state(leg) for leg in self._legs}
        held = [sum(states[leg.listing_id][0] + states[leg.listing_id][1] for leg in self._legs if leg.team == team)
                for team in (0, 1)]
        held_cost = [sum(states[leg.listing_id][2] for leg in self._legs if leg.team == team) for team in (0, 1)]
        held_team = next((team for team in (0, 1) if held[team] > 0), None)
        if held_team is not None and self._entry_q is None:
            self._entry_q = quote.q

        if held_team is not None and quote.q != self._entry_q:
            sale = self._best_sale(held_team, fairs[held_team], states)
            if sale is not None:
                leg, s = sale
                return self._order(now_ns, leg, SELL, quote.q, mid, fairs[leg.team], s.best_bid, s.edge_at_best,
                                   s.limit_price, s.shares, s.proceeds, held[leg.team])

        entry = self._best_entry(fairs, (held_team,) if held_team is not None else (0, 1))
        if entry is None:
            return []
        leg, e = entry
        committed = sum(held_cost)
        if self.max_stake_usd - committed < self.min_order_usd:
            return []
        kelly_bankroll = self.kelly_fraction * self.bankroll_usd
        take = kelly_take(e, kelly_bankroll, held[leg.team], held_cost[leg.team],
                          self.max_stake_usd - (committed - held_cost[leg.team]), self._taker_rate(leg),
                          leg.lot / SIZE_SCALE)
        if not self._sendable(leg, take.shares, take.cost):
            return []
        self._entry_q = quote.q
        leg.last_limit = take.limit_price
        return self._order(now_ns, leg, BUY, quote.q, mid, e.fair, e.best_ask, e.edge_at_best, take.limit_price,
                           take.shares, take.cost, held[leg.team], kelly_bankroll)

    def _mid_team_a(self) -> float | None:
        """Team A's mid: Polymarket International's when it has one, else the tightest two-sided book elsewhere."""
        best = None
        for venue, (a, b) in self._venues:
            m = _mid(a)
            if m is None and (mb := _mid(b)) is not None:
                m = (1.0 - mb[0], mb[1])
            if m is None:
                continue
            if venue == MID_VENUE:
                return m[0]
            if best is None or m[1] < best[1]:
                best = m
        return None if best is None else best[0]

    def _best_entry(self, fairs: tuple[float, float], teams: tuple[int, ...]) -> tuple[_Leg, Entry] | None:
        """The listing, on any venue, of an allowed team whose best ask clears the edge by the most."""
        entries = []
        for leg in self._legs:
            if leg.team not in teams:
                continue
            e = evaluate_side(fairs[leg.team], leg.asks, min_edge=self.min_edge, fee_rate=self._taker_rate(leg),
                              tick=leg.tick)
            if e is not None:
                entries.append((leg, e))
        return max(entries, key=lambda le: le[1].edge_at_best) if entries else None

    def _best_sale(self, team: int, fair: float, states: dict) -> tuple[_Leg, Sale] | None:
        """The listing of the held team whose bids beat holding by the most, among those worth an order."""
        sales = []
        for leg in self._legs:
            filled = states[leg.listing_id][0]
            if leg.team != team or filled <= 0:
                continue
            s = sell_down(fair, leg.bids, filled, min_edge=self.min_edge, fee_rate=self._taker_rate(leg),
                          lot=leg.lot / SIZE_SCALE)
            if s is not None and self._sendable(leg, s.shares, s.proceeds):
                sales.append((leg, s))
        return max(sales, key=lambda ls: ls[1].edge_at_best) if sales else None

    def _sendable(self, leg: _Leg, shares: float, usd: float) -> bool:
        size = round(shares * SIZE_SCALE)
        return size > 0 and size >= leg.min_size and usd >= self.min_order_usd

    def _order(self, now_ns: int, leg: _Leg, direction: int, q: float, mid: float, fair: float, best: float,
               edge: float, limit: float, shares: float, usd: float, held_shares: float,
               kelly_bankroll: float = math.nan) -> list[Intent]:
        leg.in_flight = True
        self._record_order(now_ns, leg, direction, q, mid, fair, best, edge, limit, shares, usd, held_shares,
                           kelly_bankroll)
        return [Intent(leg.exchange_id, leg.security_id, take_side=Side.BID if direction == BUY else Side.ASK,
                       take_size=round(shares * SIZE_SCALE), take_order_type=OrderType.LIMIT,
                       take_limit_price=round(limit * PRICE_SCALE))]

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        leg = self._leg_of.get(self.streams.listing_id_for(report.exchange_id, report.security_id))
        if leg is not None and report.order_status in TERMINAL:
            leg.in_flight = False
        return []

    # ---- metrics ---------------------------------------------------------------------------

    def _record_quote(self, now_ns: int, quote: Quote) -> None:
        if self._model_buf is None or quote.at_ns == self._recorded_quote_at:
            return
        self._recorded_quote_at = quote.at_ns
        buf = self._model_buf
        buf.begin()
        buf.set_long(self._q_ts, now_ns)
        buf.set_long(self._q_at, quote.at_ns)
        buf.set_double(self._q_q, quote.q)
        buf.set_long(self._q_lineups, hash(quote.lineups) & ((1 << 63) - 1))
        buf.set_long(self._q_known, int(quote.lineup_known))
        buf.set_long(self._q_cutoff, self._model_cutoff_ns)
        buf.commit()

    def _record_order(self, now_ns: int, leg: _Leg, direction: int, q: float, mid: float, fair: float, best: float,
                      edge: float, limit: float, shares: float, usd: float, held_shares: float,
                      kelly_bankroll: float) -> None:
        if self._metrics_buf is None:
            return
        m = _mid(leg)
        buf = self._metrics_buf
        buf.begin()
        buf.set_long(self._m_ts, now_ns)
        buf.set_long(self._m_exchange, leg.exchange_id)
        buf.set_long(self._m_listing, leg.listing_id)
        buf.set_long(self._m_team, leg.team)
        buf.set_long(self._m_direction, direction)
        buf.set_double(self._m_q, q)
        buf.set_double(self._m_mid, mid)
        buf.set_double(self._m_fair, fair)
        buf.set_double(self._m_best, best)
        # NaN on a one-sided book: the case the column exists to tell apart.
        buf.set_double(self._m_spread, m[1] if m else math.nan)
        buf.set_double(self._m_edge, edge)
        buf.set_double(self._m_limit, limit)
        buf.set_double(self._m_shares, shares)
        buf.set_double(self._m_usd, usd)
        buf.set_double(self._m_kelly_bankroll, kelly_bankroll)
        buf.set_double(self._m_held_shares, held_shares)
        buf.commit()
