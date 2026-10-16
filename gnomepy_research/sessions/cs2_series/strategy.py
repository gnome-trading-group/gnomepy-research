"""
CS2 series winner, pre-match, on one venue for one HLTV match.

gnome-launcher starts one session per match per venue, so each session sits near its venue
and a venue's trouble stays in its own sessions. The session prices the match itself: the
pre-veto model reads each cs2_upcoming_features row as the pipeline writes it (match_feed.py),
and the series DP turns map prices into a series price.

Inside the entry window - from window_hours before HLTV's latest kickoff until kickoff, while
HLTV still lists the match as upcoming - it buys whichever team's listing is underpriced by at
least min_edge after the venue's taker fee. Orders are IOC limits capped at the highest price
that still clears the threshold, so it only ever takes liquidity it has already priced.
Positions are held to settlement: once the match starts (HLTV no longer lists it as upcoming, or
its kickoff has passed) the session ends itself, keeping what it holds.

It does not trade until both lineups are announced: the model leans on player form, which is
blank until then. Nor does it trade on stale match state: if the newest cs2_upcoming row is
older than max_state_age_minutes, the pipeline has stalled, and a match that has since started
or moved would look upcoming. Once it holds a position it stops adding if a lineup changes, because a
stand-in the model has barely seen is news it cannot price.

Settings come from the six-month walk-forward backtest (spec.yaml). State that must survive a
restart - which side it holds, money committed, average entry - is read from the engine's
positions rather than kept here, so a restarted session resumes instead of buying again.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from gnomepy import ExecutionReport, Intent, OrderStatus, OrderType, Scales, Side, Strategy
from gnomepy.data import ArtifactQuery, ArtifactStore, resolve_artifact_path
from gnomepy.java.schemas import Schema
from gnomepy.registry import RegistryClient

from gnomepy_research.arb import registry_fee_rates
from gnomepy_research.models.cs2.win_probability.pre_map_model import CS2PreMapModel
from gnomepy_research.models.cs2.win_probability.series_pricing import price
from gnomepy_research.sessions.cs2_series.decision import Level, choose_entry, cost_per_share, shares_to_buy
from gnomepy_research.sessions.cs2_series.match_feed import FEATURES, MATCHES, MatchFeed, Quote

logger = logging.getLogger(__name__)

PRICE_SCALE = Scales.PRICE
SIZE_SCALE = Scales.SIZE
HOUR_NS = 3_600_000_000_000
TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
UPCOMING = "upcoming"
ARTIFACT = "artifact://"
DEFAULT_MODEL = "artifact://xgboost_model/cs2_prematch_preveto"


@dataclass
class _Leg:
    listing_id: int
    exchange_id: int
    security_id: int
    tick: float
    lot: int
    min_size: int
    asks: list[Level] = field(default_factory=list)
    in_flight: bool = False
    last_limit: float = 0.0


def load_model(model_path: str) -> tuple[CS2PreMapModel, str]:
    """The pre-veto model and the exact version it resolved to (an unpinned URI or an alias is resolved first)."""
    if model_path.startswith(ARTIFACT):
        store = ArtifactStore()
        ref = store.find(ArtifactQuery.parse(model_path))
        return CS2PreMapModel(store.resolve(ref)), str(ref)
    return CS2PreMapModel(resolve_artifact_path(model_path)), model_path


class CS2Series(Strategy):
    """
    Args:
        match_id: the HLTV match.
        listing_team_a, listing_team_b: the venue's registry listings for each HLTV team
            winning the series.
        model_path: the pre-veto model, an artifact URI or path; unpinned means the latest.
        min_edge: the least edge after the fee worth buying.
        window_hours: how long before kickoff the entry window opens.
        stake_usd: most this session may commit, fees included.
        max_slippage: how far above the best ask a single order may reach.
        adverse_move: stop adding once the price has fallen this far below our average entry -
            a big pre-kickoff move is usually news the model cannot see.
        min_order_usd: smallest order worth sending.
        max_state_age_minutes: the oldest match state it trades on; the pipeline writes one every 15 minutes.
        reload_minutes: how often to re-read the match's rows (0 = once, for replays).
        near_kickoff_reload_minutes, near_kickoff_hours: re-read this often inside this long before kickoff,
            so HLTV marking the match live stops trading within a minute of the pipeline writing it.
        features_source, matches_source: DatasetStore tables, or parquet extracts of them.
    """

    def __init__(
        self,
        match_id: int,
        listing_team_a: int,
        listing_team_b: int,
        model_path: str = DEFAULT_MODEL,
        min_edge: float = 0.05,
        window_hours: float = 12.0,
        stake_usd: float = 50.0,
        max_slippage: float = 0.02,
        adverse_move: float = 0.08,
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
        self.min_edge = min_edge
        self.window_ns = int(window_hours * HOUR_NS)
        self.stake_usd = stake_usd
        self.max_slippage = max_slippage
        self.adverse_move = adverse_move
        self.min_order_usd = min_order_usd
        self.book_depth = book_depth
        self._processing_time_ns = processing_time_ns
        self.max_state_age_ns = int(max_state_age_minutes * 60 * 1e9)
        self._metrics_buf = None
        self._entry_lineups = None

        registry = RegistryClient()
        self._legs = (self._resolve(registry, listing_team_a), self._resolve(registry, listing_team_b))
        self._side_of = {(leg.exchange_id, leg.security_id): side for side, leg in enumerate(self._legs)}

        model, resolved = load_model(model_path)
        logger.info("match %d: pricing with %s", self.match_id, resolved)
        self._feed = MatchFeed(self.match_id, lambda rows: price(model, rows)["series"].tolist(),
                               features_source, matches_source, reload_minutes * 60,
                               near_kickoff_reload_minutes * 60, near_kickoff_hours * 3600)

    @staticmethod
    def _resolve(registry, listing_id: int) -> _Leg:
        listings = registry.get_listing(listing_id=int(listing_id))
        if not listings:
            raise ValueError(f"listing {listing_id} not found in registry")
        specs = registry.get_listing_spec(listing_id=int(listing_id))
        spec = specs[0] if specs else None
        tick = int(spec.tick_size) / PRICE_SCALE if spec and spec.tick_size else 0.01
        lot = int(spec.lot_size) if spec and spec.lot_size else 1
        min_size = int(spec.min_size) if spec and spec.min_size else 0
        return _Leg(listing_id=int(listing_id), exchange_id=listings[0].exchange_id,
                    security_id=listings[0].security_id, tick=tick, lot=lot, min_size=min_size)

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def register_metrics(self) -> None:
        buf = self.metrics.create_buffer("cs2_series_orders")
        self._m_ts = buf.add_long_column("timestamp")
        self._m_side = buf.add_long_column("side")
        self._m_fair = buf.add_double_column("fair")
        self._m_ask = buf.add_double_column("best_ask")
        self._m_edge = buf.add_double_column("edge_at_best")
        self._m_limit = buf.add_double_column("limit_price")
        self._m_shares = buf.add_double_column("shares")
        buf.freeze()
        self._metrics_buf = buf

    def _taker_rate(self, leg: _Leg) -> float:
        """The leg's taker rate in the registry; the fee per share is rate * p * (1 - p)."""
        return self._fee_rates((leg.exchange_id, leg.security_id))[0]

    # ---- state read from the engine's positions --------------------------------------

    def _leg_state(self, leg: _Leg) -> tuple[float, float, float]:
        """(filled shares, committed USD incl. fees and pending, average entry price)."""
        pos = self.positions.get_position(leg.exchange_id, leg.security_id)
        filled = pos.net_quantity / SIZE_SCALE if pos else 0.0
        avg = pos.avg_entry_price / PRICE_SCALE if pos and pos.net_quantity else 0.0
        fees = float(pos.total_fees) if pos else 0.0
        pending = max(0.0, self.positions.get_effective_quantity(leg.exchange_id, leg.security_id) / SIZE_SCALE - filled)
        return filled, filled * avg + fees + pending * cost_per_share(leg.last_limit, self._taker_rate(leg)), avg

    # ---- engine callbacks ----------------------------------------------------------------

    def on_market_data(self, data: Schema) -> list[Intent]:
        side = self._side_of.get((data.exchange_id, data.security_id))
        if side is None:
            return []
        self._legs[side].asks = [Level(data.ask_price(i) / PRICE_SCALE, data.ask_size(i) / SIZE_SCALE)
                                 for i in range(self.book_depth) if data.ask_size(i) > 0]
        return self._evaluate(data.event_timestamp)

    def _evaluate(self, now_ns: int) -> list[Intent]:
        state, quote = self._feed.state(now_ns), self._feed.quote(now_ns)
        if state is not None and (state.status != UPCOMING or now_ns >= state.kickoff_ns):
            # Judged on the last state even when it's stale: a stalled pipeline can't make a passed kickoff upcoming.
            self.request_exit(f"match {self.match_id} is {state.status}" if state.status != UPCOMING
                              else f"match {self.match_id} kicked off")
            return []
        if state is None or quote is None:
            return []
        if now_ns - state.at_ns > self.max_state_age_ns:
            return []
        if not (state.kickoff_ns - self.window_ns <= now_ns < state.kickoff_ns):
            return []
        if not quote.lineup_known or any(leg.in_flight for leg in self._legs):
            return []

        states = [self._leg_state(leg) for leg in self._legs]
        held = [side for side, (filled, committed, _) in enumerate(states) if filled > 0 or committed > 0]
        if held and not self._lineups_unchanged(quote):
            return []
        budget = self.stake_usd - sum(s[1] for s in states)
        if budget < self.min_order_usd:
            return []

        fee_rates = (self._taker_rate(self._legs[0]), self._taker_rate(self._legs[1]))
        entry = choose_entry(quote.p_team_a, self._legs[0].asks, self._legs[1].asks,
                             min_edge=self.min_edge, fee_rates=fee_rates, max_slippage=self.max_slippage,
                             tick=self._legs[0].tick, allowed_sides=tuple(held) if held else (0, 1))
        if entry is None:
            return []
        avg_entry = states[entry.side][2]
        if avg_entry > 0 and entry.best_ask <= avg_entry - self.adverse_move:
            return []

        leg = self._legs[entry.side]
        fee_rate = fee_rates[entry.side]
        shares = shares_to_buy(entry, budget, fee_rate)
        size = int(shares * SIZE_SCALE) // leg.lot * leg.lot
        if size <= 0 or size < leg.min_size or shares * cost_per_share(entry.limit_price, fee_rate) < self.min_order_usd:
            return []

        if not held:
            self._entry_lineups = quote.lineups
        leg.in_flight = True
        leg.last_limit = entry.limit_price
        self._record(now_ns, entry, size / SIZE_SCALE)
        return [Intent(leg.exchange_id, leg.security_id, take_side=Side.BID, take_size=size,
                       take_order_type=OrderType.LIMIT, take_limit_price=round(entry.limit_price * PRICE_SCALE))]

    def _lineups_unchanged(self, quote: Quote) -> bool:
        """Whether the lineups still match the ones the position was entered on."""
        if self._entry_lineups is None:
            # Restarted with a position: what the earlier session entered on isn't stored, so
            # the lineups at restart become the reference.
            self._entry_lineups = quote.lineups
        return quote.lineups == self._entry_lineups

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        side = self._side_of.get((report.exchange_id, report.security_id))
        if side is not None and report.order_status in TERMINAL:
            self._legs[side].in_flight = False
        return []

    def _record(self, now_ns: int, entry, shares: float) -> None:
        if self._metrics_buf is None:
            return
        buf = self._metrics_buf
        row = buf.append_row()
        buf.set_long(row, self._m_ts, now_ns)
        buf.set_long(row, self._m_side, entry.side)
        buf.set_double(row, self._m_fair, entry.fair)
        buf.set_double(row, self._m_ask, entry.best_ask)
        buf.set_double(row, self._m_edge, entry.edge_at_best)
        buf.set_double(row, self._m_limit, entry.limit_price)
        buf.set_double(row, self._m_shares, shares)
