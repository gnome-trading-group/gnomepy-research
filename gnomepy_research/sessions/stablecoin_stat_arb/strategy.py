from __future__ import annotations

import math

from gnomepy import ExecutionReport, Intent, OrderType, Side, Strategy
from gnomepy.java.schemas import Schema

Listing = tuple[int, int]


class _TimedSpreadEWMA:
    """EWMA that decays by wall-clock time rather than tick count.

    Prevents the effective half-life from collapsing to seconds when tick
    rate is high. halflife_seconds controls how quickly the mean tracks the
    spread; the resulting std reflects genuine medium-term spread volatility.
    """

    def __init__(self, halflife_seconds: float = 300.0, warmup_ticks: int = 50):
        self.halflife_seconds = halflife_seconds
        self.warmup_ticks = warmup_ticks
        self._mean = 0.0
        self._var = 0.0
        self._count = 0
        self._last_ts: int = 0
        self._ln2_over_hl = math.log(2) / halflife_seconds

    def update(self, value: float, timestamp_ns: int) -> None:
        if self._count == 0:
            self._mean = value
            self._var = 0.0
            self._last_ts = timestamp_ns
        else:
            elapsed_s = (timestamp_ns - self._last_ts) / 1_000_000_000.0
            alpha = math.exp(-self._ln2_over_hl * elapsed_s)
            delta = value - self._mean
            self._mean = alpha * self._mean + (1.0 - alpha) * value
            self._var = alpha * self._var + (1.0 - alpha) * delta * delta
            self._last_ts = timestamp_ns
        self._count += 1

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return math.sqrt(self._var) if self._var > 0 else 0.0

    @property
    def is_ready(self) -> bool:
        return self._count >= self.warmup_ticks and self._var > 0


class StablecoinStatArb(Strategy):
    def __init__(
        self,
        size: int = 1000,
        max_position: int = 5,
        halflife_seconds: float = 300.0,
        warmup_ticks: int = 50,
        entry_z: float = 2.5,
        close_z: float = 0.0,
        max_hold_ns: int = 600_000_000_000,
        max_imbalance_ticks: int = 20,
        max_staleness_ns: int = 60_000_000_000,
        min_spread_bps: float = 0.5,
        processing_time_ns: int = 0,
    ):
        self.size = size
        self.max_position = max_position
        self.halflife_seconds = halflife_seconds
        self.warmup_ticks = warmup_ticks
        self.entry_z = entry_z
        self.close_z = close_z
        self.max_hold_ns = max_hold_ns
        self.max_imbalance_ticks = max_imbalance_ticks
        self.max_staleness_ns = max_staleness_ns
        self.min_spread_bps = min_spread_bps
        self._processing_time_ns = processing_time_ns

        self._listings: list[Listing] = []
        self._best_bid: dict[Listing, int] = {}
        self._best_ask: dict[Listing, int] = {}
        self._last_update_ts: dict[Listing, int] = {}
        self._ewma: dict[tuple[Listing, Listing], _TimedSpreadEWMA] = {}

        self._open_pair: tuple[Listing, Listing] | None = None
        self._open_direction: int = 0
        self._entry_ts: int = 0
        self._closing: bool = False
        self._last_close_ts: int = 0
        self._imbalance_ticks = 0

        # Locked at entry — prevents close being triggered by EWMA drifting up
        # to meet a persistent spread rather than the spread actually reverting.
        self._locked_mean: float = 0.0
        self._locked_std: float = 0.0

        self._buf = None
        self._col_ts = None
        self._col_z = None
        self._col_spread = None
        self._col_pos_long = None
        self._col_pos_short = None
        self._col_state = None

    def register_metrics(self) -> None:
        buf = self.metrics.create_buffer("diagnostics")
        self._col_ts = buf.addLongColumn("timestamp")
        self._col_z = buf.addDoubleColumn("z_score")
        self._col_spread = buf.addDoubleColumn("spread_bps")
        self._col_pos_long = buf.addLongColumn("pos_long")
        self._col_pos_short = buf.addLongColumn("pos_short")
        self._col_state = buf.addLongColumn("state")
        buf.freeze()
        self._buf = buf

    def _log(self, ts: int, z: float, spread: float, pos_long: int, pos_short: int, state: int) -> None:
        if self._buf is None:
            return
        row = self._buf.appendRow()
        self._buf.setLong(row, self._col_ts, ts)
        self._buf.setDouble(row, self._col_z, z)
        self._buf.setDouble(row, self._col_spread, spread)
        self._buf.setLong(row, self._col_pos_long, pos_long)
        self._buf.setLong(row, self._col_pos_short, pos_short)
        self._buf.setLong(row, self._col_state, state)

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def _net_qty(self, listing: Listing) -> int:
        pos = self.oms.get_position(*listing)
        return pos.net_quantity if pos is not None else 0

    def on_market_data(self, data: Schema) -> list[Intent]:
        bid = data.bid_price(0)
        ask = data.ask_price(0)
        if bid <= 0 or ask <= 0:
            return []

        listing: Listing = (data.exchange_id, data.security_id)
        timestamp = data.event_timestamp

        if listing not in self._best_bid:
            self._listings.append(listing)
            for other in self._listings[:-1]:
                key = (listing, other) if listing > other else (other, listing)
                if key not in self._ewma:
                    self._ewma[key] = _TimedSpreadEWMA(
                        halflife_seconds=self.halflife_seconds,
                        warmup_ticks=self.warmup_ticks,
                    )

        self._best_bid[listing] = bid
        self._best_ask[listing] = ask
        self._last_update_ts[listing] = timestamp

        if len(self._listings) < 2:
            return []

        stale = {
            lst for lst in self._listings
            if timestamp - self._last_update_ts.get(lst, 0) > self.max_staleness_ns
        }

        self._update_ewma(listing, stale, timestamp)

        if self._open_pair is not None:
            return self._manage_open_pair(timestamp, stale)

        return self._scan_entry(timestamp, stale)

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        return []

    def _spread_bps(self, lst_a: Listing, lst_b: Listing) -> float:
        mid_a = (self._best_bid[lst_a] + self._best_ask[lst_a]) / 2
        mid_b = (self._best_bid[lst_b] + self._best_ask[lst_b]) / 2
        avg_mid = (mid_a + mid_b) / 2
        if avg_mid <= 0:
            return 0.0
        return (mid_b - mid_a) / avg_mid * 10_000

    def _ewma_key(self, lst_a: Listing, lst_b: Listing) -> tuple[Listing, Listing]:
        return (lst_a, lst_b) if lst_a > lst_b else (lst_b, lst_a)

    def _signed_spread(self, lst_a: Listing, lst_b: Listing) -> float:
        spread = self._spread_bps(lst_a, lst_b)
        if lst_a > lst_b:
            spread = -spread
        return spread

    def _update_ewma(self, source: Listing, stale: set[Listing], timestamp: int) -> None:
        live = [lst for lst in self._listings if lst not in stale]
        for i, lst_a in enumerate(live):
            for lst_b in live[i + 1:]:
                if source not in (lst_a, lst_b):
                    continue
                key = self._ewma_key(lst_a, lst_b)
                if key in self._ewma:
                    spread = self._signed_spread(lst_a, lst_b)
                    self._ewma[key].update(spread, timestamp)

    def _manage_open_pair(self, timestamp: int, stale: set[Listing]) -> list[Intent]:
        long_lst, short_lst = self._open_pair
        pos_long = self._net_qty(long_lst)
        pos_short = self._net_qty(short_lst)

        spread = self._signed_spread(long_lst, short_lst)
        locked_z = (
            (spread - self._locked_mean) / self._locked_std
            if self._locked_std > 0 else 0.0
        )
        self._log(timestamp, locked_z, spread, pos_long, pos_short, 1)

        if pos_long == 0 and pos_short == 0:
            if self._closing:
                self._open_pair = None
                self._closing = False
                return []
            if timestamp - self._entry_ts > 5_000_000_000:
                self._open_pair = None
            return []

        if self._closing:
            if timestamp - self._last_close_ts >= 30_000_000_000:
                self._open_pair = None
                self._closing = False
                return []
            eff_long = self.oms.get_effective_quantity(*long_lst) or 0
            eff_short = self.oms.get_effective_quantity(*short_lst) or 0
            long_inflight = pos_long != 0 and abs(eff_long) < abs(pos_long) * 0.5
            short_inflight = pos_short != 0 and abs(eff_short) < abs(pos_short) * 0.5
            if not long_inflight and not short_inflight:
                if timestamp - self._last_close_ts >= 5_000_000_000:
                    self._last_close_ts = timestamp
                    return self._close(long_lst, short_lst, pos_long, pos_short)
            return []

        if timestamp - self._entry_ts >= self.max_hold_ns:
            self._closing = True
            self._last_close_ts = timestamp
            self._imbalance_ticks = 0
            return self._close(long_lst, short_lst, pos_long, pos_short)

        if (pos_long != 0) != (pos_short != 0):
            self._imbalance_ticks += 1
        else:
            self._imbalance_ticks = 0

        if self._imbalance_ticks >= self.max_imbalance_ticks:
            self._imbalance_ticks = 0
            self._closing = True
            self._last_close_ts = timestamp
            return self._unwind(long_lst, short_lst, pos_long, pos_short)

        if pos_long != 0 and pos_short != 0:
            if long_lst not in stale and short_lst not in stale:
                directed_z = locked_z * self._open_direction
                if directed_z < self.close_z:
                    self._closing = True
                    self._last_close_ts = timestamp
                    return self._close(long_lst, short_lst, pos_long, pos_short)

        return []

    def _scan_entry(self, timestamp: int, stale: set[Listing]) -> list[Intent]:
        best_z = 0.0
        best_long: Listing | None = None
        best_short: Listing | None = None
        best_direction = 0
        best_locked_mean = 0.0
        best_locked_std = 0.0

        live = [lst for lst in self._listings if lst not in stale]
        for i, lst_a in enumerate(live):
            for lst_b in live[i + 1:]:
                key = self._ewma_key(lst_a, lst_b)
                ewma = self._ewma.get(key)
                if not ewma or not ewma.is_ready or ewma.std <= 0:
                    continue

                spread = self._signed_spread(lst_a, lst_b)
                z = (spread - ewma.mean) / ewma.std

                self._log(timestamp, z, spread, 0, 0, 0)

                eff_a = self.oms.get_effective_quantity(*lst_a) or 0
                eff_b = self.oms.get_effective_quantity(*lst_b) or 0
                net_a = self._net_qty(lst_a)
                net_b = self._net_qty(lst_b)
                if (
                    max(abs(net_a), abs(eff_a)) >= self.max_position
                    or max(abs(net_b), abs(eff_b)) >= self.max_position
                ):
                    continue

                larger = lst_a if lst_a > lst_b else lst_b
                smaller = lst_b if lst_a > lst_b else lst_a

                if z > self.entry_z and z > best_z:
                    if abs(spread) >= self.min_spread_bps:
                        best_z = z
                        best_long = larger
                        best_short = smaller
                        best_direction = 1
                        best_locked_mean = ewma.mean
                        best_locked_std = ewma.std
                elif -z > self.entry_z and -z > best_z:
                    if abs(spread) >= self.min_spread_bps:
                        best_z = -z
                        best_long = smaller
                        best_short = larger
                        best_direction = -1
                        best_locked_mean = ewma.mean
                        best_locked_std = ewma.std

        if best_z > self.entry_z and best_long is not None:
            self._open_pair = (best_long, best_short)
            self._open_direction = best_direction
            self._locked_mean = best_locked_mean
            self._locked_std = best_locked_std
            self._imbalance_ticks = 0
            self._closing = False
            self._entry_ts = timestamp
            return [
                self._take(*best_long, Side.ASK),
                self._take(*best_short, Side.BID),
            ]

        return []

    def _take(self, exchange_id: int, security_id: int, side: Side) -> Intent:
        return Intent(
            exchange_id=exchange_id,
            security_id=security_id,
            take_side=side,
            take_size=self.size,
            take_order_type=OrderType.MARKET,
        )

    def _take_qty(self, exchange_id: int, security_id: int, side: Side, qty: int) -> Intent:
        return Intent(
            exchange_id=exchange_id,
            security_id=security_id,
            take_side=side,
            take_size=qty,
            take_order_type=OrderType.MARKET,
        )

    def _close(
        self, long_lst: Listing, short_lst: Listing, pos_long: int, pos_short: int
    ) -> list[Intent]:
        intents = []
        if pos_long < 0:
            intents.append(self._take_qty(*long_lst, Side.BID, abs(pos_long)))
        elif pos_long > 0:
            intents.append(self._take_qty(*long_lst, Side.ASK, abs(pos_long)))
        if pos_short < 0:
            intents.append(self._take_qty(*short_lst, Side.BID, abs(pos_short)))
        elif pos_short > 0:
            intents.append(self._take_qty(*short_lst, Side.ASK, abs(pos_short)))
        return intents

    def _unwind(
        self, long_lst: Listing, short_lst: Listing, pos_long: int, pos_short: int
    ) -> list[Intent]:
        intents = []
        if pos_long != 0:
            side = Side.BID if pos_long < 0 else Side.ASK
            intents.append(self._take_qty(*long_lst, side, abs(pos_long)))
        if pos_short != 0:
            side = Side.BID if pos_short < 0 else Side.ASK
            intents.append(self._take_qty(*short_lst, side, abs(pos_short)))
        return intents
