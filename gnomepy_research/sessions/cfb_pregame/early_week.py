"""
College football moneylines on Kalshi from early in the game week: rest a bid on the side the model prefers, then
sell it back before kickoff (idea A2b, ~/cfb-live-data/early-week-study/SPEC.md). Paper research.

The 2025-26 study found that the model's gap to Kalshi's price 72 hours before kickoff predicts which way the price
moves by kickoff (t 6.5 in 2025, 2.2 in 2026), but a taker round trip loses about 4c, and the simulated resting version
lost 0.4c a game once a fill needed the price to trade through the order. This session runs the resting version
forward.

For each game, at decide_hours before kickoff (or the first look at a usable book within max_late_hours of it), the
model's P(home), frozen in the launch trigger, is compared with the market's mid: the mean of each leg's mid in home
terms, a leg counting only when it is two-sided and at most max_spread wide. When they differ by min_gap or more, a
post-only bid for `contracts` rests on the model's side at that listing's best bid, moved to the best bid every
entry_repeg_hours. From entry_close_hours before kickoff an unfilled bid is cancelled and whatever is held is offered
at the best ask, moved every exit_repeg_hours. Whatever is still held taker_exit_minutes before kickoff is sold into
the best bid; what that sale leaves is held to settlement. With hold_to_settlement (an exploratory variant on the same
entries) nothing is offered: from entry_close_hours out an unfilled bid or remainder is cancelled and what was bought
is kept to settlement, which skips the exit's costs and takes on the game's result. The decision is made once per game
and recorded
(cfb_early_week_decisions), as is every order sent (cfb_early_week_orders).

A game the trigger marks off_protocol (launched after its decision window had closed, on the operator's call) is
decided at its first usable look however late, until its entry closes; its decision row says so and how many hours
before kickoff it was made, so it can be reported apart from the pre-registered games.

After a restart a held position marks its game's side and the game carries on from where the clock is. A game with
nothing held whose decision time passed more than max_late_hours ago is left alone. Kickoffs are the trigger's.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import pandas as pd

from gnomepy import ExecutionReport, Intent, OrderStatus, OrderType, Scales, Side, Strategy
from gnomepy.java.schemas import Schema
from gnomepy.registry import RegistryClient

logger = logging.getLogger(__name__)

PRICE_SCALE = Scales.PRICE
SIZE_SCALE = Scales.SIZE
MINUTE_NS = 60_000_000_000
HOUR_NS = 60 * MINUTE_NS
TERMINAL = {OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED}
HOME, AWAY, NO_SIDE = 0, 1, -1
# Order actions in cfb_early_week_orders.
BID, ASK, SELL, CANCEL = 1, 2, 3, 4
# A game's phases.
WAIT, ENTRY, EXIT, DONE = range(4)


def early_mid(home_bid: float, home_ask: float, away_bid: float, away_ask: float, max_spread: float) -> float:
    """P(home): the mean of the usable legs' mids in home terms, a leg being usable when two-sided and at most
    max_spread wide; NaN when neither is."""
    mids = []
    if home_ask - home_bid <= max_spread + 1e-12:
        mids.append((home_bid + home_ask) / 2)
    if away_ask - away_bid <= max_spread + 1e-12:
        mids.append(1 - (away_bid + away_ask) / 2)
    return sum(mids) / len(mids) if mids else math.nan


@dataclass
class _Leg:
    listing_id: int
    exchange_id: int
    security_id: int
    lot: float
    min_size: int
    best_bid: float = math.nan
    best_ask: float = math.nan
    seen: bool = False
    in_flight: bool = False


@dataclass
class _Game:
    game_id: int
    legs: tuple[_Leg, _Leg]
    kickoff_ns: int
    p_home: float
    off_protocol: bool = False
    phase: int = WAIT
    side: int = NO_SIDE
    next_peg_ns: int = 0


class CFBEarlyWeek(Strategy):
    """
    Args:
        games: the slate, [{"game_id", "listing_home", "listing_away", "kickoff": ISO, "p_home": the model's P(home),
            frozen at launch, "off_protocol": optional, decide however late}].
        model: the model behind p_home, for the record.
        contracts: contracts bought per game.
        min_gap: the least |model - market mid| at the decision time worth trading.
        decide_hours: how long before kickoff the decision is made.
        max_late_hours: how long after the decision time a first usable book still counts.
        entry_close_hours: how long before kickoff an unfilled bid is cancelled and exits start.
        entry_repeg_hours, exit_repeg_hours: how often a resting bid or ask moves to the current best price.
        taker_exit_minutes: how long before kickoff whatever is held is sold into the bid.
        max_spread: widest leg that counts toward the decision's mid.
        hold_to_settlement: keep what was bought to settlement instead of selling it before kickoff.
    """

    def __init__(
        self,
        games: list[dict],
        model: str = "",
        contracts: float = 1.0,
        min_gap: float = 0.05,
        decide_hours: float = 72.0,
        max_late_hours: float = 6.0,
        entry_close_hours: float = 6.0,
        entry_repeg_hours: float = 6.0,
        exit_repeg_hours: float = 1.0,
        taker_exit_minutes: float = 5.0,
        max_spread: float = 0.06,
        hold_to_settlement: bool = False,
    ):
        if contracts <= 0:
            raise ValueError(f"contracts must be positive, got {contracts}")
        if not 0 < min_gap < 1:
            raise ValueError(f"min_gap must be in (0, 1), got {min_gap}")
        if not decide_hours * 60 > entry_close_hours * 60 > taker_exit_minutes >= 0:
            raise ValueError("need decide_hours > entry_close_hours > taker_exit_minutes >= 0, got "
                             f"{decide_hours} h, {entry_close_hours} h, {taker_exit_minutes} min")
        self.model = model
        self.contracts = contracts
        self.min_gap = min_gap
        self.decide_ns = int(decide_hours * HOUR_NS)
        self.late_ns = int(max_late_hours * HOUR_NS)
        self.close_ns = int(entry_close_hours * HOUR_NS)
        self.entry_repeg_ns = int(entry_repeg_hours * HOUR_NS)
        self.exit_repeg_ns = int(exit_repeg_hours * HOUR_NS)
        self.final_ns = int(taker_exit_minutes * MINUTE_NS)
        self.max_spread = max_spread
        self.hold_to_settlement = bool(hold_to_settlement)
        self._metrics_ready = False

        registry = RegistryClient()
        self._games: dict[int, _Game] = {}
        for g in games:
            legs = (self._resolve(registry, g["listing_home"]), self._resolve(registry, g["listing_away"]))
            if None in legs:
                logger.warning("game %s: a listing isn't in the registry; not trading it", g["game_id"])
                continue
            self._games[int(g["game_id"])] = _Game(int(g["game_id"]), legs, pd.Timestamp(g["kickoff"]).value,
                                                   float(g["p_home"]), bool(g.get("off_protocol", False)))
        if not self._games:
            raise ValueError("no game on the slate has both listings in the registry")
        self._owner = {(leg.exchange_id, leg.security_id): (game.game_id, side)
                       for game in self._games.values() for side, leg in enumerate(game.legs)}
        logger.info("%d games, model %s, %.0f contracts, gap %.2f", len(self._games), model, contracts, min_gap)

    @staticmethod
    def _resolve(registry, listing_id: int) -> _Leg | None:
        listings = registry.get_listing(listing_id=int(listing_id))
        if not listings:
            return None
        specs = registry.get_listing_spec(listing_id=int(listing_id))
        spec = specs[0] if specs else None
        lot = int(spec.lot_size) / SIZE_SCALE if spec and spec.lot_size else 1.0
        min_size = int(spec.min_size) if spec and spec.min_size else 0
        return _Leg(listing_id=int(listing_id), exchange_id=listings[0].exchange_id,
                    security_id=listings[0].security_id, lot=lot, min_size=min_size)

    def register_metrics(self) -> None:
        decisions = self.metrics.create_stream("cfb_early_week_decisions")
        self._d = {name: (decisions.add_long_column(name) if name in ("timestamp", "game_id", "side", "late",
                                                                      "off_protocol")
                          else decisions.add_double_column(name))
                   for name in ("timestamp", "game_id", "side", "late", "off_protocol", "p_model", "market_mid", "gap",
                                "hours_before")}
        decisions.freeze()
        orders = self.metrics.create_stream("cfb_early_week_orders")
        self._o = {name: (orders.add_long_column(name) if name in ("timestamp", "game_id", "side", "action")
                          else orders.add_double_column(name))
                   for name in ("timestamp", "game_id", "side", "action", "price", "shares", "held", "best_bid",
                                "best_ask")}
        orders.freeze()
        self._decisions, self._orders = decisions, orders
        self._metrics_ready = True

    # ---- engine callbacks ----------------------------------------------------------------

    def on_market_data(self, data: Schema) -> list[Intent]:
        owner = self._owner.get((data.exchange_id, data.security_id))
        if owner is None:
            return []
        leg = self._games[owner[0]].legs[owner[1]]
        leg.best_bid = data.bid_price(0) / PRICE_SCALE if data.bid_size(0) > 0 else math.nan
        leg.best_ask = data.ask_price(0) / PRICE_SCALE if data.ask_size(0) > 0 else math.nan
        leg.seen = True
        now = data.event_timestamp
        # Every game is stepped on any update: the slate's books together are the session's clock.
        intents = [i for game in self._games.values() for i in self._step(game, now)]
        if (all(g.phase == DONE for g in self._games.values()) and not self.is_exiting
                and not any(leg.in_flight for g in self._games.values() for leg in g.legs)):
            self.request_exit("every game is done")
        return intents

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        owner = self._owner.get((report.exchange_id, report.security_id))
        if owner is None or report.order_status not in TERMINAL:
            return []
        game = self._games[owner[0]]
        game.legs[owner[1]].in_flight = False
        if report.order_status == OrderStatus.REJECTED and game.phase in (ENTRY, EXIT):
            game.next_peg_ns = 0
        return []

    # ---- one game --------------------------------------------------------------------------

    def _held(self, game: _Game) -> tuple[int, float] | None:
        for side, leg in enumerate(game.legs):
            pos = self.positions.get_position(leg.exchange_id, leg.security_id)
            if pos and pos.net_quantity > 0:
                return side, pos.net_quantity / SIZE_SCALE
        return None

    def _step(self, game: _Game, now: int) -> list[Intent]:
        decide_at = game.kickoff_ns - self.decide_ns
        close_at = game.kickoff_ns - self.close_ns
        final_at = game.kickoff_ns - self.final_ns
        if game.phase == DONE or now < decide_at:
            return []
        held = self._held(game)
        if game.phase == WAIT:
            if held is not None:
                game.side, game.phase, game.next_peg_ns = held[0], ENTRY, now
            elif (not game.off_protocol and now > decide_at + self.late_ns) or now >= close_at:
                game.phase = DONE
                self._decision(now, game, math.nan, late=True)
                return []
            else:
                home, away = game.legs
                if not (home.seen and away.seen):
                    return []
                m = early_mid(home.best_bid, home.best_ask, away.best_bid, away.best_ask, self.max_spread)
                if math.isnan(m):
                    return []
                gap = game.p_home - m
                game.side = HOME if gap >= self.min_gap - 1e-12 else AWAY if gap <= -self.min_gap + 1e-12 else NO_SIDE
                self._decision(now, game, m, late=False)
                if game.side == NO_SIDE:
                    game.phase = DONE
                    return []
                game.phase, game.next_peg_ns = ENTRY, now
        leg = game.legs[game.side]
        shares = held[1] if held is not None else 0.0
        if game.phase == ENTRY:
            if now < close_at:
                want = self._lots(leg, self.contracts - shares)
                if want <= 0 or now < game.next_peg_ns or math.isnan(leg.best_bid):
                    return []
                game.next_peg_ns = now + self.entry_repeg_ns
                return self._rest(now, game, leg, BID, leg.best_bid, want, shares)
            if shares <= 0:
                game.phase = DONE
                return self._cancel(now, game, leg)
            if self.hold_to_settlement:
                game.phase = DONE
                return self._cancel(now, game, leg) if self._lots(leg, self.contracts - shares) > 0 else []
            game.phase, game.next_peg_ns = EXIT, now
        if game.phase == EXIT:
            if shares <= 0:
                game.phase = DONE
                return []
            if now >= final_at:
                game.phase = DONE
                sell = self._lots(leg, shares)
                if math.isnan(leg.best_bid) or sell <= 0:
                    return self._cancel(now, game, leg)
                leg.in_flight = True
                self._order(now, game, leg, SELL, leg.best_bid, sell, shares)
                return [Intent(leg.exchange_id, leg.security_id, take_side=Side.ASK, take_size=round(sell * SIZE_SCALE),
                               take_order_type=OrderType.LIMIT, take_limit_price=round(leg.best_bid * PRICE_SCALE))]
            if now < game.next_peg_ns or math.isnan(leg.best_ask):
                return []
            game.next_peg_ns = now + self.exit_repeg_ns
            return self._rest(now, game, leg, ASK, leg.best_ask, self._lots(leg, shares), shares)
        return []

    @staticmethod
    def _lots(leg: _Leg, shares: float) -> float:
        return math.floor(shares / leg.lot + 1e-9) * leg.lot if shares > 0 else 0.0

    def _rest(self, now: int, game: _Game, leg: _Leg, action: int, price: float, shares: float,
              held: float) -> list[Intent]:
        size = round(shares * SIZE_SCALE)
        if size <= 0 or size < leg.min_size:
            return []
        self._order(now, game, leg, action, price, shares, held)
        p = round(price * PRICE_SCALE)
        if action == BID:
            return [Intent(leg.exchange_id, leg.security_id, bid_price=p, bid_size=size, post_only=True)]
        return [Intent(leg.exchange_id, leg.security_id, ask_price=p, ask_size=size, post_only=True)]

    def _cancel(self, now: int, game: _Game, leg: _Leg) -> list[Intent]:
        self._order(now, game, leg, CANCEL, math.nan, 0.0, 0.0)
        return [Intent(leg.exchange_id, leg.security_id)]

    # ---- metrics --------------------------------------------------------------------------

    def _decision(self, now: int, game: _Game, m: float, *, late: bool) -> None:
        if not self._metrics_ready:
            return
        buf, c = self._decisions, self._d
        buf.begin()
        buf.set_long(c["timestamp"], now)
        buf.set_long(c["game_id"], game.game_id)
        buf.set_long(c["side"], game.side)
        buf.set_long(c["late"], int(late))
        buf.set_long(c["off_protocol"], int(game.off_protocol))
        buf.set_double(c["p_model"], game.p_home)
        buf.set_double(c["market_mid"], m)
        buf.set_double(c["gap"], game.p_home - m)
        buf.set_double(c["hours_before"], (game.kickoff_ns - now) / HOUR_NS)
        buf.commit()

    def _order(self, now: int, game: _Game, leg: _Leg, action: int, price: float, shares: float, held: float) -> None:
        if not self._metrics_ready:
            return
        buf, c = self._orders, self._o
        buf.begin()
        buf.set_long(c["timestamp"], now)
        buf.set_long(c["game_id"], game.game_id)
        buf.set_long(c["side"], game.side)
        buf.set_long(c["action"], action)
        buf.set_double(c["price"], price)
        buf.set_double(c["shares"], shares)
        buf.set_double(c["held"], held)
        buf.set_double(c["best_bid"], leg.best_bid)
        buf.set_double(c["best_ask"], leg.best_ask)
        buf.commit()
