from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from gnomepy import ExecutionReport, Intent, OrderType, Side, Strategy
from gnomepy.java.schemas import Schema
from gnomepy.registry import RegistryClient, RelationshipGraph

PRICE_SCALE = 1_000_000_000

Listing = tuple[int, int]  # (exchange_id, security_id)


class State(IntEnum):
    FLAT = 0
    ENTERING = 1
    PARTIAL = 2
    UNWINDING = 3


@dataclass
class ImpliesPair:
    index: int
    a_yes_sid: int
    b_yes_sid: int
    a_no_sid: int
    confidence: float


def _build_pairs(event_ids: list[int], min_confidence: float) -> tuple[list[ImpliesPair], list[int]]:
    client = RegistryClient()

    # Fetch all securities across all events
    all_sids: set[int] = set()
    for event_id in event_ids:
        for ec in client.get_event_contracts(event_id=event_id):
            all_sids.add(ec.security_id)

    # Fetch all relationships touching any of these securities, dedup by relationship_id
    seen_rel_ids: set[int] = set()
    all_rels = []
    for sid in all_sids:
        for rel in client.get_contract_relationships(security_id=sid):
            if rel.relationship_id not in seen_rel_ids:
                seen_rel_ids.add(rel.relationship_id)
                # Only keep relationships where both sides are in our event set
                if rel.security_id_a in all_sids and rel.security_id_b in all_sids:
                    all_rels.append(rel)

    graph = RelationshipGraph(all_rels)

    pairs: list[ImpliesPair] = []
    for a_sid, b_sid, confidence in graph.get_implies_pairs(min_confidence=min_confidence):
        complements = graph.get_complement(a_sid)
        if not complements:
            continue
        pairs.append(ImpliesPair(
            index=len(pairs),
            a_yes_sid=a_sid,
            b_yes_sid=b_sid,
            a_no_sid=complements[0],
            confidence=confidence,
        ))

    # Fetch all listing_ids for all securities so the runner can subscribe
    listing_ids: list[int] = []
    for sid in all_sids:
        for listing in client.get_listing(security_id=sid):
            listing_ids.append(listing.listing_id)

    return pairs, listing_ids


class ImplicationBoundaryArb(Strategy):
    def __init__(
        self,
        event_ids: list[int],
        size: int = 1_000_000,
        fee_rate: float = 0.07,
        min_edge_bps: float = 10.0,
        min_implies_confidence: float = 0.95,
        stale_threshold_ns: int = 2_000_000_000,
        imbalance_timeout_ns: int = 30_000_000_000,
        cooldown_ns: int = 500_000_000,
        max_open_pairs: int = 1,
        processing_time_ns: int = 0,
    ):
        self.size = size
        self.fee_rate = fee_rate
        self.min_edge_bps = min_edge_bps
        self.stale_threshold_ns = stale_threshold_ns
        self.imbalance_timeout_ns = imbalance_timeout_ns
        self.cooldown_ns = cooldown_ns
        self.max_open_pairs = max_open_pairs
        self._processing_time_ns = processing_time_ns

        self._pairs, self.listing_ids = _build_pairs(event_ids, min_implies_confidence)

        # sid -> list of Listing (populated as market data arrives)
        self._sid_to_listings: dict[int, list[Listing]] = {}

        self._best_ask: dict[Listing, int] = {}
        self._best_ask_size: dict[Listing, int] = {}
        self._last_update_ts: dict[Listing, int] = {}

        self._state: dict[int, State] = {p.index: State.FLAT for p in self._pairs}
        self._entry_ts: dict[int, int] = {p.index: 0 for p in self._pairs}
        self._last_attempt_ts: dict[int, int] = {p.index: 0 for p in self._pairs}
        self._target_qty: dict[int, int] = {p.index: 0 for p in self._pairs}
        self._active_b_yes: dict[int, Listing] = {}
        self._active_a_no: dict[int, Listing] = {}

        # Track which security_ids we care about for fast filtering
        self._watched_sids: set[int] = set()
        for p in self._pairs:
            self._watched_sids.update([p.a_yes_sid, p.b_yes_sid, p.a_no_sid])

        self._buf = None
        self._col_ts = None
        self._col_pair = None
        self._col_edge_bps = None
        self._col_cost = None
        self._col_state = None
        self._col_ask_b_yes = None
        self._col_ask_a_no = None

    def register_metrics(self) -> None:
        buf = self.metrics.create_buffer("diagnostics")
        self._col_ts = buf.addLongColumn("timestamp")
        self._col_pair = buf.addLongColumn("pair_index")
        self._col_edge_bps = buf.addDoubleColumn("edge_bps")
        self._col_cost = buf.addDoubleColumn("cost")
        self._col_state = buf.addLongColumn("state")
        self._col_ask_b_yes = buf.addLongColumn("best_ask_b_yes")
        self._col_ask_a_no = buf.addLongColumn("best_ask_a_no")
        buf.freeze()
        self._buf = buf

    def _log(self, ts: int, pair_idx: int, edge_bps: float, cost: float, state: State, ask_b: int, ask_a: int) -> None:
        if self._buf is None:
            return
        row = self._buf.appendRow()
        self._buf.setLong(row, self._col_ts, ts)
        self._buf.setLong(row, self._col_pair, pair_idx)
        self._buf.setDouble(row, self._col_edge_bps, edge_bps)
        self._buf.setDouble(row, self._col_cost, cost)
        self._buf.setLong(row, self._col_state, int(state))
        self._buf.setLong(row, self._col_ask_b_yes, ask_b)
        self._buf.setLong(row, self._col_ask_a_no, ask_a)

    def simulate_processing_time(self) -> int:
        return self._processing_time_ns

    def _net_qty(self, listing: Listing) -> int:
        pos = self.positions.get_position(*listing)
        return pos.net_quantity if pos is not None else 0

    def _best_ask_for_sid(self, sid: int, timestamp: int) -> tuple[int, int, Listing | None]:
        best_price = 0
        best_size = 0
        best_listing: Listing | None = None
        for lst in self._sid_to_listings.get(sid, []):
            ask = self._best_ask.get(lst, 0)
            if ask <= 0:
                continue
            if timestamp - self._last_update_ts.get(lst, 0) > self.stale_threshold_ns:
                continue
            if best_price == 0 or ask < best_price:
                best_price = ask
                best_size = self._best_ask_size.get(lst, 0)
                best_listing = lst
        return best_price, best_size, best_listing

    def _compute_edge(self, pair: ImpliesPair, timestamp: int) -> tuple[float, float, Listing | None, Listing | None]:
        ask_b, size_b, lst_b = self._best_ask_for_sid(pair.b_yes_sid, timestamp)
        ask_a, size_a, lst_a = self._best_ask_for_sid(pair.a_no_sid, timestamp)

        if ask_b <= 0 or ask_a <= 0 or lst_b is None or lst_a is None:
            return 0.0, float("-inf"), None, None

        p_b = ask_b / PRICE_SCALE
        p_a = ask_a / PRICE_SCALE
        cost = p_b + p_a
        fee = self.fee_rate * p_b * (1.0 - p_b) + self.fee_rate * p_a * (1.0 - p_a)
        edge_bps = (1.0 - cost - fee) * 10_000

        return cost, edge_bps, lst_b, lst_a

    def _open_pair_count(self) -> int:
        return sum(1 for s in self._state.values() if s != State.FLAT)

    def on_market_data(self, data: Schema) -> list[Intent]:
        sid = data.security_id
        if sid not in self._watched_sids:
            return []

        listing: Listing = (data.exchange_id, sid)
        timestamp = data.event_timestamp

        if sid not in self._sid_to_listings:
            self._sid_to_listings[sid] = []
        if listing not in self._sid_to_listings[sid]:
            self._sid_to_listings[sid].append(listing)

        ask = data.ask_price(0)
        if ask > 0:
            self._best_ask[listing] = ask
            self._best_ask_size[listing] = data.ask_size(0)
        self._last_update_ts[listing] = timestamp

        intents = []
        for pair in self._pairs:
            state = self._state[pair.index]
            if state == State.FLAT:
                if self._open_pair_count() < self.max_open_pairs:
                    intents.extend(self._handle_flat(pair, timestamp))
            elif state == State.ENTERING:
                intents.extend(self._handle_entering(pair, timestamp))
            elif state == State.PARTIAL:
                intents.extend(self._handle_partial(pair, timestamp))
            elif state == State.UNWINDING:
                intents.extend(self._handle_unwinding(pair))
        return intents

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        return []

    def _handle_flat(self, pair: ImpliesPair, timestamp: int) -> list[Intent]:
        if timestamp - self._last_attempt_ts[pair.index] < self.cooldown_ns:
            return []

        cost, edge_bps, lst_b, lst_a = self._compute_edge(pair, timestamp)
        self._log(timestamp, pair.index, edge_bps, cost, State.FLAT,
                  self._best_ask.get(lst_b, 0) if lst_b else 0,
                  self._best_ask.get(lst_a, 0) if lst_a else 0)

        if edge_bps < self.min_edge_bps or lst_b is None or lst_a is None:
            return []

        size_b = self._best_ask_size.get(lst_b, 0)
        size_a = self._best_ask_size.get(lst_a, 0)
        if size_b <= 0 or size_a <= 0:
            return []

        trade_size = min(self.size, size_b, size_a)

        self._state[pair.index] = State.ENTERING
        self._entry_ts[pair.index] = timestamp
        self._last_attempt_ts[pair.index] = timestamp
        self._target_qty[pair.index] = trade_size
        self._active_b_yes[pair.index] = lst_b
        self._active_a_no[pair.index] = lst_a

        return [
            Intent(
                exchange_id=lst_b[0],
                security_id=lst_b[1],
                take_side=Side.BID,
                take_size=self.positions.compliant_size(lst_b[0], lst_b[1], trade_size, self._best_ask[lst_b]),
                take_order_type=OrderType.MARKET,
            ),
            Intent(
                exchange_id=lst_a[0],
                security_id=lst_a[1],
                take_side=Side.BID,
                take_size=self.positions.compliant_size(lst_a[0], lst_a[1], trade_size, self._best_ask[lst_a]),
                take_order_type=OrderType.MARKET,
            ),
        ]

    def _handle_entering(self, pair: ImpliesPair, timestamp: int) -> list[Intent]:
        if timestamp - self._entry_ts[pair.index] < 1_000_000_000:
            return []

        lst_b = self._active_b_yes.get(pair.index)
        lst_a = self._active_a_no.get(pair.index)
        if lst_b is None or lst_a is None:
            self._state[pair.index] = State.FLAT
            return []

        target = self._target_qty[pair.index]
        filled_b = self._net_qty(lst_b) >= target
        filled_a = self._net_qty(lst_a) >= target

        if filled_b and filled_a:
            self._state[pair.index] = State.FLAT
            self._active_b_yes.pop(pair.index, None)
            self._active_a_no.pop(pair.index, None)
            return []

        if filled_b or filled_a:
            self._state[pair.index] = State.PARTIAL
            return []

        self._state[pair.index] = State.FLAT
        return []

    def _handle_partial(self, pair: ImpliesPair, timestamp: int) -> list[Intent]:
        lst_b = self._active_b_yes.get(pair.index)
        lst_a = self._active_a_no.get(pair.index)
        if lst_b is None or lst_a is None:
            self._state[pair.index] = State.FLAT
            return []

        target = self._target_qty[pair.index]
        if self._net_qty(lst_b) >= target and self._net_qty(lst_a) >= target:
            self._state[pair.index] = State.FLAT
            self._active_b_yes.pop(pair.index, None)
            self._active_a_no.pop(pair.index, None)
            return []

        if timestamp - self._entry_ts[pair.index] > self.imbalance_timeout_ns:
            self._state[pair.index] = State.UNWINDING
            return self._unwind_intents(pair)

        return []

    def _handle_unwinding(self, pair: ImpliesPair) -> list[Intent]:
        lst_b = self._active_b_yes.get(pair.index)
        lst_a = self._active_a_no.get(pair.index)
        has_open = (lst_b and self._net_qty(lst_b) > 0) or (lst_a and self._net_qty(lst_a) > 0)
        if not has_open:
            self._state[pair.index] = State.FLAT
            self._active_b_yes.pop(pair.index, None)
            self._active_a_no.pop(pair.index, None)
            return []
        return self._unwind_intents(pair)

    def _unwind_intents(self, pair: ImpliesPair) -> list[Intent]:
        intents = []
        for lst in (self._active_b_yes.get(pair.index), self._active_a_no.get(pair.index)):
            if lst is None:
                continue
            qty = self._net_qty(lst)
            if qty > 0:
                intents.append(Intent(
                    exchange_id=lst[0],
                    security_id=lst[1],
                    take_side=Side.ASK,
                    take_size=qty,
                    take_order_type=OrderType.MARKET,
                ))
        return intents
