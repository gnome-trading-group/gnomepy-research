"""
Pre-match CS2 moneyline strategy for Polymarket: series winner and map 1 winner.

The model runs outside; this class only executes against a predictions table.
For each configured market it watches both outcome tokens' ask books and, inside
that market's entry window, buys whichever side the model says is underpriced by
at least the threshold after the taker fee. Orders are IOC limits capped at the
highest price that still clears the threshold, so it only ever takes liquidity
it has already priced. Positions are held to settlement.

Settings come from the six-month backtest: series from 12h before kickoff with a
5c threshold, map 1 in the final hour with 10c (its 5-10c bucket was break-even),
taker-only, flat stakes with one cap per match across both markets because they
are bets on the same teams.

State that must survive a restart - which side a market holds, money committed,
average entry - is read from the engine's positions rather than kept here, so a
restarted paper or live session resumes instead of buying again.

Prices and kickoff come from the latest prediction written at or before the
current event time, and the source is re-read every reload_minutes, so a live
session picks up new predictions (and a rescheduled kickoff) as the pipeline
writes them, and a backtest never sees a prediction before it existed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from gnomepy import ExecutionReport, Intent, OrderStatus, OrderType, Scales, Side, Strategy
from gnomepy.java.schemas import Schema
from gnomepy.registry import RegistryClient

from gnomepy_research.sessions.cs2_prematch import predictions as predictions_table
from gnomepy_research.sessions.cs2_prematch.decision import Level, choose_entry, cost_per_share, shares_to_buy

logger = logging.getLogger(__name__)

PRICE_SCALE = Scales.PRICE
SIZE_SCALE = Scales.SIZE
HOUR_NS = 3_600_000_000_000
TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}

DEFAULT_MIN_EDGE = {"series": 0.05, "game1": 0.10}
DEFAULT_WINDOWS = {"series": (12.0, 0.0), "game1": (1.0, 0.0)}


@dataclass
class _Leg:
    listing_id: int
    exchange_id: int
    security_id: int
    tick: float
    lot: int
    asks: list[Level] = field(default_factory=list)
    in_flight: bool = False
    last_limit: float = 0.0


@dataclass
class _Market:
    match_id: int
    market: str
    legs: tuple[_Leg, _Leg]


class CS2PreMatch(Strategy):
    """
    Args:
        markets: one dict per market traded -
            {match_id, market: "series" | "game1", listing_team_a, listing_team_b}
            where listing_team_* are the registry listings of each team's outcome token.
            Kickoff is not configured here; it comes with each prediction.
        predictions_path: parquet path or DatasetStore name; see predictions.py.
        reload_minutes: how often to re-read predictions_path (0 = load once, for
            backtests on a fixed table).
        min_edge: per-market threshold after fee, e.g. {"series": 0.05, "game1": 0.10}.
        windows: per-market (open, close) in hours before kickoff.
        stake_usd: most a single market may commit, fees included.
        max_match_usd: most one match may commit across all its markets.
        max_slippage: how far above the best ask a single order may reach.
        fee_rate: Polymarket parametric taker fee rate.
        adverse_move: stop adding to a market once its price has fallen this far
            below our average entry - a big pre-kickoff move is usually news the
            model cannot see.
        skip_unranked_series: skip series where a team was unranked (tentative
            backtest finding, off by default until the forward test confirms it).
        min_order_usd: smallest order worth sending.
    """

    def __init__(
        self,
        markets: list[dict],
        predictions_path: str,
        min_edge: dict | None = None,
        windows: dict | None = None,
        stake_usd: float = 50.0,
        max_match_usd: float = 100.0,
        max_slippage: float = 0.02,
        fee_rate: float = 0.07,
        adverse_move: float = 0.08,
        skip_unranked_series: bool = False,
        min_order_usd: float = 1.0,
        book_depth: int = 10,
        processing_time_ns: int = 0,
        reload_minutes: float = 5.0,
    ):
        self.min_edge = {**DEFAULT_MIN_EDGE, **(min_edge or {})}
        self.windows = {**DEFAULT_WINDOWS, **{k: tuple(v) for k, v in (windows or {}).items()}}
        self.stake_usd = stake_usd
        self.max_match_usd = max_match_usd
        self.max_slippage = max_slippage
        self.fee_rate = fee_rate
        self.adverse_move = adverse_move
        self.skip_unranked_series = skip_unranked_series
        self.min_order_usd = min_order_usd
        self.book_depth = book_depth
        self._processing_time_ns = processing_time_ns
        self._metrics_buf = None

        self._predictions_path = predictions_path
        self._reload_ns = int(reload_minutes * 60 * 1e9)
        self._loaded_at_ns: int | None = None
        self._book = predictions_table.PredictionBook(predictions_table.load(predictions_path))
        registry = RegistryClient()
        self._markets: list[_Market] = []
        self._by_key: dict[tuple[int, int], tuple[_Market, int]] = {}
        for spec in markets:
            if spec["market"] not in self.min_edge:
                raise ValueError(f"unsupported market {spec['market']!r}")
            legs = tuple(self._resolve(registry, spec[k]) for k in ("listing_team_a", "listing_team_b"))
            if (int(spec["match_id"]), spec["market"]) not in self._book:
                logger.warning("no prediction yet for match %s %s - it trades once one appears",
                               spec["match_id"], spec["market"])
            m = _Market(match_id=int(spec["match_id"]), market=spec["market"], legs=legs)
            self._markets.append(m)
            for side, leg in enumerate(legs):
                self._by_key[(leg.exchange_id, leg.security_id)] = (m, side)

    @staticmethod
    def _resolve(registry, listing_id: int) -> _Leg:
        listings = registry.get_listing(listing_id=int(listing_id))
        if not listings:
            raise ValueError(f"listing {listing_id} not found in registry")
        specs = registry.get_listing_spec(listing_id=int(listing_id))
        tick = int(specs[0].tick_size) / PRICE_SCALE if specs and specs[0].tick_size else 0.01
        lot = int(specs[0].lot_size) if specs and specs[0].lot_size else 1
        return _Leg(listing_id=int(listing_id), exchange_id=listings[0].exchange_id,
                    security_id=listings[0].security_id, tick=tick, lot=lot)

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def register_metrics(self) -> None:
        buf = self.metrics.create_buffer("cs2_prematch_orders")
        self._m_ts = buf.add_long_column("timestamp")
        self._m_match = buf.add_long_column("match_id")
        self._m_market = buf.add_long_column("market_is_map1")
        self._m_side = buf.add_long_column("side")
        self._m_fair = buf.add_double_column("fair")
        self._m_ask = buf.add_double_column("best_ask")
        self._m_edge = buf.add_double_column("edge_at_best")
        self._m_limit = buf.add_double_column("limit_price")
        self._m_shares = buf.add_double_column("shares")
        buf.freeze()
        self._metrics_buf = buf

    # ---- state read from the engine's positions --------------------------------------

    def _leg_state(self, leg: _Leg) -> tuple[float, float, float]:
        """(filled shares, committed USD incl. fees and pending, average entry price)."""
        pos = self.positions.get_position(leg.exchange_id, leg.security_id)
        filled = pos.net_quantity / SIZE_SCALE if pos else 0.0
        avg = pos.avg_entry_price / PRICE_SCALE if pos and pos.net_quantity else 0.0
        fees = float(pos.total_fees) if pos else 0.0
        pending = max(0.0, self.positions.get_effective_quantity(leg.exchange_id, leg.security_id) / SIZE_SCALE - filled)
        return filled, filled * avg + fees + pending * cost_per_share(leg.last_limit, self.fee_rate), avg

    def _committed(self, m: _Market) -> float:
        return sum(self._leg_state(leg)[1] for leg in m.legs)

    def _match_committed(self, match_id: int) -> float:
        return sum(self._committed(m) for m in self._markets if m.match_id == match_id)

    # ---- engine callbacks ----------------------------------------------------------------

    def on_market_data(self, data: Schema) -> list[Intent]:
        hit = self._by_key.get((data.exchange_id, data.security_id))
        if hit is None:
            return []
        m, side = hit
        m.legs[side].asks = [Level(data.ask_price(i) / PRICE_SCALE, data.ask_size(i) / SIZE_SCALE)
                             for i in range(self.book_depth) if data.ask_size(i) > 0]
        self._maybe_reload(data.event_timestamp)
        return self._evaluate(m, data.event_timestamp)

    def _maybe_reload(self, now_ns: int) -> None:
        if self._loaded_at_ns is None:
            self._loaded_at_ns = now_ns
            return
        if self._reload_ns <= 0 or now_ns - self._loaded_at_ns < self._reload_ns:
            return
        self._loaded_at_ns = now_ns
        try:
            self._book = predictions_table.PredictionBook(predictions_table.load(self._predictions_path))
        except Exception:
            # a failed refresh keeps trading on the predictions already held rather than stopping
            logger.exception("reloading predictions from %s failed", self._predictions_path)

    def _evaluate(self, m: _Market, now_ns: int) -> list[Intent]:
        pred = self._book.as_of(m.match_id, m.market, now_ns)
        if pred is None:
            return []
        if self.skip_unranked_series and m.market == "series" and pred["rank_known_both"] != 1:
            return []
        kickoff_ns = pred["kickoff_ns"]
        open_h, close_h = self.windows[m.market]
        if not (kickoff_ns - open_h * HOUR_NS <= now_ns < kickoff_ns - close_h * HOUR_NS):
            return []
        if any(leg.in_flight for leg in m.legs):
            return []

        states = [self._leg_state(leg) for leg in m.legs]
        held = [side for side, (filled, committed, _) in enumerate(states) if filled > 0 or committed > 0]
        allowed = tuple(held) if held else (0, 1)
        budget = min(self.stake_usd - sum(s[1] for s in states),
                     self.max_match_usd - self._match_committed(m.match_id))
        if budget < self.min_order_usd:
            return []

        entry = choose_entry(pred["p_team_a"], m.legs[0].asks, m.legs[1].asks,
                             min_edge=self.min_edge[m.market], fee_rate=self.fee_rate,
                             max_slippage=self.max_slippage, tick=m.legs[0].tick, allowed_sides=allowed)
        if entry is None:
            return []
        avg_entry = states[entry.side][2]
        if avg_entry > 0 and entry.best_ask <= avg_entry - self.adverse_move:
            return []

        leg = m.legs[entry.side]
        shares = shares_to_buy(entry, budget, self.fee_rate)
        size = int(shares * SIZE_SCALE) // leg.lot * leg.lot
        if size <= 0 or shares * cost_per_share(entry.limit_price, self.fee_rate) < self.min_order_usd:
            return []

        leg.in_flight = True
        leg.last_limit = entry.limit_price
        self._record(now_ns, m, entry, size / SIZE_SCALE)
        return [Intent(leg.exchange_id, leg.security_id, take_side=Side.BID, take_size=size,
                       take_order_type=OrderType.LIMIT, take_limit_price=round(entry.limit_price * PRICE_SCALE))]

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        hit = self._by_key.get((report.exchange_id, report.security_id))
        if hit is not None and report.order_status in TERMINAL:
            m, side = hit
            m.legs[side].in_flight = False
        return []

    def _record(self, now_ns: int, m: _Market, entry, shares: float) -> None:
        if self._metrics_buf is None:
            return
        buf = self._metrics_buf
        row = buf.append_row()
        buf.set_long(row, self._m_ts, now_ns)
        buf.set_long(row, self._m_match, m.match_id)
        buf.set_long(row, self._m_market, int(m.market == "game1"))
        buf.set_long(row, self._m_side, entry.side)
        buf.set_double(row, self._m_fair, entry.fair)
        buf.set_double(row, self._m_ask, entry.best_ask)
        buf.set_double(row, self._m_edge, entry.edge_at_best)
        buf.set_double(row, self._m_limit, entry.limit_price)
        buf.set_double(row, self._m_shares, shares)
