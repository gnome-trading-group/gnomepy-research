"""
College football moneylines, pregame, on one venue for one slate of games.

gnome-launcher starts one session per variant per venue (a direct_launch trigger built by launch.py), holding both
team listings of every game on the slate. The session prices each game from the newest B0 prediction in
cfb_pregame_predictions (prediction_feed.py) and from its own two books: the fair price moves from the market's
twin-combined mid toward the model by anchor_weight in log-odds (pricing.py), so the anchored variant (0.25) trades only
large disagreements and the raw variant (1.0) is the model-price control the 2026 desk replays lost with.

Until a game's entry cutoff it buys whichever team's listing, priced inside [min_price, max_price), is underpriced by at
least min_edge after the venue's taker fee, sized by the Kelly criterion on kelly_fraction of bankroll_usd, never past
max_game_stake_usd for the game or max_total_stake_usd for the slate; an order walks the asks only as deep as every
share still clears the threshold, as an IOC limit at the deepest level it costed (decision.py, as in cs2_series). It
holds one side per game, stops adding once the price has fallen adverse_move below its average entry, and holds to
settlement. Once every game is past its cutoff and no order is in flight it ends itself, keeping what it holds.

It stays out of a game on a stale prediction, an uncertain QB, a bottom-35 team in weeks 5-9, an FBS newcomer, a wide
or one-sided book, or a listing with no fee schedule. Each prediction is recorded once against the market (the
cfb_pregame_marks stream) whether or not it trades, so the model's live information weight can be measured from the
session alone. State that must survive a restart - sides held, money committed, average entry - is read from the
engine's positions.

This is paper research: B0 trails the 2026 close by about three times its backtest gap.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import pandas as pd

from gnomepy import ExecutionReport, Intent, OrderStatus, OrderType, Scales, Side, Strategy
from gnomepy.java.schemas import Schema
from gnomepy.registry import RegistryClient

from gnomepy_research.fees import registry_fee_rates
from gnomepy_research.sessions.cfb_pregame.decision import Level, choose_entry, cost_per_share, kelly_take
from gnomepy_research.sessions.cfb_pregame.prediction_feed import PREDICTIONS, PredictionFeed
from gnomepy_research.sessions.cfb_pregame.pricing import anchored_fair, skip_reason, twin_mid

logger = logging.getLogger(__name__)

PRICE_SCALE = Scales.PRICE
SIZE_SCALE = Scales.SIZE
MINUTE_NS = 60_000_000_000
TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
HOME, AWAY = 0, 1


@dataclass
class _Leg:
    listing_id: int
    exchange_id: int
    security_id: int
    tick: float
    lot: int
    min_size: int
    asks: list[Level] = field(default_factory=list)
    best_bid: float = math.nan
    in_flight: bool = False
    last_limit: float = 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else math.nan


@dataclass
class _Game:
    game_id: int
    legs: tuple[_Leg, _Leg]
    kickoff_ns: int
    marked_at: set[int] = field(default_factory=set)


class CFBPregame(Strategy):
    """
    Args:
        games: the slate, [{"game_id": CFBD game id, "listing_home": ..., "listing_away": ..., "kickoff": ISO}]; the
            kickoff is the one known at launch, replaced by the predictions' newest once they load.
        bankroll_usd: the money the session sizes its bets against.
        anchor_weight: how far, in log-odds, the fair price moves from the market's mid toward the model (1 = model).
        min_edge: the least edge after the fee worth buying.
        kelly_fraction: the share of the Kelly size to hold.
        max_game_stake_usd, max_total_stake_usd: most the session commits, fees included, per game and on the slate.
        min_price, max_price: the band of best asks it buys in, [min_price, max_price).
        max_spread: widest ask - bid on either leg it trades against.
        entry_cutoff_minutes: how long before kickoff a game's entry closes.
        max_prediction_age_minutes: the oldest prediction it trades on.
        min_qb_p: least p_start for either team's assumed starter (the model prices one QB).
        skip_bottom, skip_weeks: stay out of games with a team in the bottom skip_bottom of FBS during these weeks.
        skip_newcomers: stay out of games with a team in its first three FBS seasons.
        adverse_move: stop adding once the price has fallen this far below the average entry.
        min_order_usd: smallest order worth sending.
        reload_minutes: how often to re-read the predictions (0 = once, for replays).
        predictions_source: the DatasetStore table, or a parquet extract of it.
    """

    def __init__(
        self,
        games: list[dict],
        bankroll_usd: float,
        anchor_weight: float = 0.25,
        min_edge: float = 0.01,
        kelly_fraction: float = 0.25,
        max_game_stake_usd: float = 100.0,
        max_total_stake_usd: float = 2000.0,
        min_price: float = 0.15,
        max_price: float = 0.85,
        max_spread: float = 0.06,
        entry_cutoff_minutes: float = 30.0,
        max_prediction_age_minutes: float = 150.0,
        min_qb_p: float = 0.6,
        skip_bottom: int = 35,
        skip_weeks: tuple[int, int] = (5, 9),
        skip_newcomers: bool = True,
        adverse_move: float = 0.08,
        min_order_usd: float = 1.0,
        book_depth: int = 10,
        reload_minutes: float = 5.0,
        predictions_source: str = PREDICTIONS,
        processing_time_ns: int = 0,
    ):
        if bankroll_usd <= 0:
            raise ValueError(f"bankroll_usd must be positive, got {bankroll_usd}")
        if not 0 <= anchor_weight <= 1:
            raise ValueError(f"anchor_weight must be in [0, 1], got {anchor_weight}")
        if not 0 <= min_price < max_price <= 1:
            raise ValueError(f"need 0 <= min_price < max_price <= 1, got [{min_price}, {max_price})")
        self.bankroll_usd = bankroll_usd
        self.anchor_weight = anchor_weight
        self.min_edge = min_edge
        self.kelly_fraction = kelly_fraction
        self.max_game_stake_usd = max_game_stake_usd
        self.max_total_stake_usd = max_total_stake_usd
        self.min_price, self.max_price = min_price, max_price
        self.max_spread = max_spread
        self.cutoff_ns = int(entry_cutoff_minutes * MINUTE_NS)
        self._rules = dict(max_age_ns=int(max_prediction_age_minutes * MINUTE_NS), min_qb_p=min_qb_p,
                           skip_bottom=int(skip_bottom), skip_weeks=(int(skip_weeks[0]), int(skip_weeks[1])),
                           skip_newcomers=bool(skip_newcomers))
        self.adverse_move = adverse_move
        self.min_order_usd = min_order_usd
        self.book_depth = book_depth
        self._processing_time_ns = processing_time_ns
        self._fee_rates = registry_fee_rates(self)
        self._metrics_ready = False
        self._no_fees: set[int] = set()

        registry = RegistryClient()
        self._games: dict[int, _Game] = {}
        for g in games:
            legs = (self._resolve(registry, g["listing_home"]), self._resolve(registry, g["listing_away"]))
            if None in legs:
                logger.warning("game %s: a listing isn't in the registry; not trading it", g["game_id"])
                continue
            self._games[int(g["game_id"])] = _Game(int(g["game_id"]), legs, pd.Timestamp(g["kickoff"]).value)
        if not self._games:
            raise ValueError("no game on the slate has both listings in the registry")
        self._owner = {(leg.exchange_id, leg.security_id): (game.game_id, side)
                       for game in self._games.values() for side, leg in enumerate(game.legs)}
        self._feed = PredictionFeed(list(self._games), predictions_source, reload_minutes * 60)
        logger.info("trading %d games, anchor weight %.2f", len(self._games), anchor_weight)

    @staticmethod
    def _resolve(registry, listing_id: int) -> _Leg | None:
        listings = registry.get_listing(listing_id=int(listing_id))
        if not listings:
            return None
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
        orders = self.metrics.create_stream("cfb_pregame_orders")
        self._o = {name: (orders.add_long_column(name) if name in ("timestamp", "game_id", "side")
                          else orders.add_double_column(name))
                   for name in ("timestamp", "game_id", "side", "p_model", "market_mid", "fair", "best_ask", "spread",
                                "edge_at_best", "limit_price", "shares", "cost", "held_shares")}
        orders.freeze()
        marks = self.metrics.create_stream("cfb_pregame_marks")
        self._k = {name: (marks.add_long_column(name) if name in ("timestamp", "game_id", "skipped")
                          else marks.add_double_column(name))
                   for name in ("timestamp", "game_id", "p_model", "market_mid", "fair", "spread_home", "spread_away",
                                "skipped")}
        marks.freeze()
        self._orders, self._marks = orders, marks
        self._metrics_ready = True

    def _taker_rate(self, leg: _Leg) -> float:
        return self._fee_rates((leg.exchange_id, leg.security_id))[0]

    def _has_fees(self, game: _Game) -> bool:
        if game.game_id in self._no_fees:
            return False
        try:
            for leg in game.legs:
                self._taker_rate(leg)
        except ValueError:
            self._no_fees.add(game.game_id)
            logger.warning("game %d: the registry has no fee schedule for its listings; not trading it", game.game_id)
            return False
        return True

    # ---- state read from the engine's positions --------------------------------------

    def _leg_state(self, leg: _Leg) -> tuple[float, float, float]:
        """(filled shares, committed USD incl. fees and pending, average entry price)."""
        pos = self.positions.get_position(leg.exchange_id, leg.security_id)
        filled = pos.net_quantity / SIZE_SCALE if pos else 0.0
        avg = pos.avg_entry_price / PRICE_SCALE if pos and pos.net_quantity else 0.0
        fees = pos.total_fees / PRICE_SCALE if pos else 0.0
        pending = max(0.0, self.positions.get_effective_quantity(leg.exchange_id, leg.security_id) / SIZE_SCALE - filled)
        pending_cost = pending * cost_per_share(leg.last_limit, self._taker_rate(leg)) if pending > 0 else 0.0
        return filled, filled * avg + fees + pending_cost, avg

    def _committed_total(self) -> float:
        return sum(self._leg_state(leg)[1] for game in self._games.values() if game.game_id not in self._no_fees
                   for leg in game.legs)

    # ---- engine callbacks ----------------------------------------------------------------

    def on_market_data(self, data: Schema) -> list[Intent]:
        owner = self._owner.get((data.exchange_id, data.security_id))
        if owner is None:
            return []
        game_id, side = owner
        leg = self._games[game_id].legs[side]
        leg.asks = [Level(data.ask_price(i) / PRICE_SCALE, data.ask_size(i) / SIZE_SCALE)
                    for i in range(self.book_depth) if data.ask_size(i) > 0]
        leg.best_bid = data.bid_price(0) / PRICE_SCALE if data.bid_size(0) > 0 else math.nan
        return self._evaluate(self._games[game_id], data.event_timestamp)

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        owner = self._owner.get((report.exchange_id, report.security_id))
        if owner is not None and report.order_status in TERMINAL:
            self._games[owner[0]].legs[owner[1]].in_flight = False
        return []

    def _cutoffs(self) -> dict[int, int]:
        known = self._feed.kickoffs()
        return {g: known.get(g, game.kickoff_ns) - self.cutoff_ns for g, game in self._games.items()}

    def _evaluate(self, game: _Game, now_ns: int) -> list[Intent]:
        cutoffs = self._cutoffs()
        if now_ns >= max(cutoffs.values()):
            if not self.is_exiting and not any(leg.in_flight for g in self._games.values() for leg in g.legs):
                self.request_exit("slate reached its last entry cutoff")
            return []
        if now_ns >= cutoffs[game.game_id] or any(leg.in_flight for leg in game.legs):
            return []
        pred = self._feed.latest(game.game_id, now_ns)
        if pred is None or not self._has_fees(game):
            return []
        home, away = game.legs
        m = twin_mid(home.best_bid, home.best_ask, away.best_bid, away.best_ask)
        if math.isnan(m):
            return []
        spreads = (home.best_ask - home.best_bid, away.best_ask - away.best_bid)
        fair_home = anchored_fair(pred.p_home, m, self.anchor_weight)
        reason = skip_reason(pred, now_ns, **self._rules) or (
            "wide spread" if max(spreads) > self.max_spread + 1e-12 else None)
        if pred.at_ns not in game.marked_at:
            game.marked_at.add(pred.at_ns)
            self._mark(now_ns, game.game_id, pred.p_home, m, fair_home, spreads, reason is not None)
        if reason is not None:
            return []

        states = [self._leg_state(leg) for leg in game.legs]
        held = [side for side, (filled, committed, _) in enumerate(states) if filled > 0 or committed > 0]
        committed_game = sum(s[1] for s in states)
        room = min(self.max_game_stake_usd - committed_game, self.max_total_stake_usd - self._committed_total())
        if room < self.min_order_usd:
            return []

        fee_rates = (self._taker_rate(home), self._taker_rate(away))
        entry = choose_entry(fair_home, home.asks, away.asks, min_edge=self.min_edge, fee_rates=fee_rates,
                             tick=home.tick, allowed_sides=tuple(held) if held else (HOME, AWAY),
                             min_price=self.min_price, max_price=self.max_price)
        if entry is None:
            return []
        held_shares, held_cost, avg_entry = states[entry.side]
        if avg_entry > 0 and entry.best_ask <= avg_entry - self.adverse_move:
            return []

        leg = game.legs[entry.side]
        take = kelly_take(entry, self.kelly_fraction * self.bankroll_usd, held_shares, held_cost, held_cost + room,
                          fee_rates[entry.side], leg.lot / SIZE_SCALE)
        size = round(take.shares * SIZE_SCALE)
        if size <= 0 or size < leg.min_size or take.cost < self.min_order_usd:
            return []

        leg.in_flight = True
        leg.last_limit = take.limit_price
        self._order(now_ns, game.game_id, entry, take, pred.p_home, m, spreads[entry.side], held_shares)
        return [Intent(leg.exchange_id, leg.security_id, take_side=Side.BID, take_size=size,
                       take_order_type=OrderType.LIMIT, take_limit_price=round(take.limit_price * PRICE_SCALE))]

    # ---- metrics --------------------------------------------------------------------------

    def _mark(self, now_ns: int, game_id: int, p_model: float, m: float, fair: float, spreads: tuple[float, float],
              skipped: bool) -> None:
        if not self._metrics_ready:
            return
        buf, c = self._marks, self._k
        buf.begin()
        buf.set_long(c["timestamp"], now_ns)
        buf.set_long(c["game_id"], game_id)
        buf.set_double(c["p_model"], p_model)
        buf.set_double(c["market_mid"], m)
        buf.set_double(c["fair"], fair)
        buf.set_double(c["spread_home"], spreads[0])
        buf.set_double(c["spread_away"], spreads[1])
        buf.set_long(c["skipped"], int(skipped))
        buf.commit()

    def _order(self, now_ns: int, game_id: int, entry, take, p_model: float, m: float, spread: float,
               held_shares: float) -> None:
        if not self._metrics_ready:
            return
        buf, c = self._orders, self._o
        buf.begin()
        buf.set_long(c["timestamp"], now_ns)
        buf.set_long(c["game_id"], game_id)
        buf.set_long(c["side"], entry.side)
        buf.set_double(c["p_model"], p_model)
        buf.set_double(c["market_mid"], m)
        buf.set_double(c["fair"], entry.fair)
        buf.set_double(c["best_ask"], entry.best_ask)
        buf.set_double(c["spread"], spread)
        buf.set_double(c["edge_at_best"], entry.edge_at_best)
        buf.set_double(c["limit_price"], take.limit_price)
        buf.set_double(c["shares"], take.shares)
        buf.set_double(c["cost"], take.cost)
        buf.set_double(c["held_shares"], held_shares)
        buf.commit()
