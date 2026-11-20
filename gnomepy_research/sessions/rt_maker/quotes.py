"""
Pure quoting rules for the RT maker: no engine, no registry, no JVM.

A side is quoted at the best price other makers show there: the first level, walking away from the touch, holding at
least min_join_size of someone else's size. A live book includes our own resting order, so our size is taken off the
level it rests at first; otherwise a level we were left alone on would keep us there after the others had gone.

Sizes are capped by the strike's position limit, so a side stops once the position reaches it, and by the money the
film may commit. Prices are plain probabilities in [0, 1] and sizes contracts; the strategy converts at the edge.

With the review signal leaning the quotes (lean()), the latest review decides what is quoted. A review the signal
gives no side, or one whose side flips from the last side leaned, pulls both sides for pull_ns; a review with a side
quotes only that side (the YES bid for YES, the YES ask for NO) for skew_ns, or until the next review when skew_ns is
None, then both. With pull_every_review, every review pulls both sides for pull_ns first, whatever its side, and the
side is quoted alone only after that. Reviews seen at the same moment count as one. This is the rule ~/gnome-research-work/rt/outlets/
maker_skew.py scored, so a change here should be scored there first.

Where a side rests in its level's queue decides how often it is filled, and fills are where the maker loses; Kalshi pays
for resting size near the touch either way (under one reading of its rule, only size near the front of the queue). The
feed never shows single orders, so a side's place is estimated (QueueSpot): everyone else's size at the level when it
joined, less the trades at its price and the share of the level's other shrinks assumed to be cancels ahead of it.
Kalshi's feed shows a trade both as the book change it makes and as a trade print at the same moment, in either order;
they count once. A side whose estimate says it has moved near the front can be pulled and sent again, which puts it at
the back (requeue). A side can also rest behind the price it would join (behind), and the side that adds to a position
can step further back as the position grows (inventory skew); each tick back halves what Kalshi's rule scores of it.
Scoring these needs the queue, which a trades-only replay doesn't model: ~/gnome-research-work/rt/lip/backtest_queue.py
runs them in gnome-backtest under each of the simulator's queue models.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from gnomepy.price_grid import PriceGrid

from gnomepy_research.models.rt.review_feed import Review
from gnomepy_research.models.rt.signal import NO, YES, Signal, signal

PRICE_SCALE = 1_000_000_000


@dataclass(frozen=True)
class Level:
    price: float
    size: float


@dataclass(frozen=True)
class Quote:
    bid_price: float = 0.0
    bid_size: float = 0.0
    ask_price: float = 0.0
    ask_size: float = 0.0

    @property
    def empty(self) -> bool:
        return self.bid_size <= 0 and self.ask_size <= 0


NONE = Quote()
BOTH, PULL = "both", "pull"
# Where a side rests in its level's queue: as it joins, sent again from the back when near the front, or a tick behind.
JOIN, REQUEUE, BEHIND = "join", "requeue", "behind"
QUEUE_POLICIES = (JOIN, REQUEUE, BEHIND)


@dataclass(frozen=True)
class ReviewState:
    """What the latest review said: its side, whether that flipped the last side leaned, and when it was seen."""
    side: int
    flip: bool
    seen_ns: int
    signal: Signal


def review_state(reviews: list[Review], baseline: float, threshold: float, min_reviews: int) -> ReviewState | None:
    """The state after the film's reviews (in the order seen), or None before any."""
    if not reviews:
        return None
    last_side = side = 0
    flip = False
    sig = None
    i = 0
    while i < len(reviews):
        j = i
        while j < len(reviews) and reviews[j].seen_ns == reviews[i].seen_ns:
            j += 1
        sig = signal(reviews[:j], baseline, threshold, min_reviews)
        side = sig.side
        flip = side != 0 and last_side != 0 and side != last_side
        if side != 0:
            last_side = side
        i = j
    return ReviewState(side, flip, reviews[-1].seen_ns, sig)


def lean(state: ReviewState | None, now_ns: int, pull_ns: int, skew_ns: int | None,
         pull_every_review: bool = False) -> str | int:
    """What to quote now: BOTH, PULL, YES or NO."""
    if state is None:
        return BOTH
    if pull_every_review and now_ns < state.seen_ns + pull_ns:
        return PULL
    if (state.side == 0 or state.flip) and now_ns < state.seen_ns + pull_ns:
        return PULL
    if state.side != 0 and (skew_ns is None or now_ns < state.seen_ns + skew_ns):
        return state.side
    return BOTH


def one_side(q: Quote, side: int) -> Quote:
    """The side of the quote that leans with the signal: the YES bid for YES, the YES ask for NO."""
    if side == YES:
        return Quote(q.bid_price, q.bid_size, 0.0, 0.0)
    if side == NO:
        return Quote(0.0, 0.0, q.ask_price, q.ask_size)
    return q


def join_price(levels: list[Level], ours: tuple[float, float] | None, min_join_size: float) -> float | None:
    """The best price on one side (levels best first) with at least min_join_size of others' size; ours is our resting
    (price, size) on that side when the book shows it."""
    for level in levels:
        others = level.size
        if ours is not None and math.isclose(level.price, ours[0]):
            others -= ours[1]
        if others >= min_join_size:
            return level.price
    return None


def lots(contracts: float, lot: float, min_size: float) -> float:
    """Contracts rounded down to the lot, or 0 under the venue's minimum."""
    n = math.floor(contracts / lot + 1e-9) * lot if lot > 0 else contracts
    return n if n >= max(min_size, lot) else 0.0


def quote(bids: list[Level], asks: list[Level], position: float, resting: Quote | None, quote_size: float,
          max_position: float, min_price: float, max_price: float, min_join_size: float, lot: float,
          min_size: float) -> Quote:
    """Both sides of one strike. position is YES contracts held (negative: sold); resting is what we have in the book
    when the book shows it (None otherwise)."""
    bid = join_price(bids, (resting.bid_price, resting.bid_size) if resting and resting.bid_size else None,
                     min_join_size)
    ask = join_price(asks, (resting.ask_price, resting.ask_size) if resting and resting.ask_size else None,
                     min_join_size)
    if bid is not None and ask is not None and bid >= ask:
        return NONE
    bid_size = ask_size = 0.0
    if bid is not None and min_price <= bid <= max_price:
        bid_size = lots(min(quote_size, max_position - position), lot, min_size)
    if ask is not None and min_price <= ask <= max_price:
        ask_size = lots(min(quote_size, max_position + position), lot, min_size)
    return Quote(bid if bid_size else 0.0, bid_size, ask if ask_size else 0.0, ask_size)


def adds(q: Quote, position: float) -> float:
    """What filling the quote's sides could add to the money the strike commits: a bid commits its price per contract
    beyond what it would buy back of a short, an ask one minus its price beyond what it would sell of a long."""
    bid = max(0.0, q.bid_size - max(0.0, -position)) * q.bid_price
    ask = max(0.0, q.ask_size - max(0.0, position)) * (1 - q.ask_price)
    return bid + ask


def reducing(q: Quote, position: float, lot: float, min_size: float) -> Quote:
    """Only the side that trades the position back towards flat, at most the position."""
    bid = lots(min(q.bid_size, max(0.0, -position)), lot, min_size)
    ask = lots(min(q.ask_size, max(0.0, position)), lot, min_size)
    return Quote(q.bid_price if bid else 0.0, bid, q.ask_price if ask else 0.0, ask)


def held_cost(position: float, avg_price: float) -> float:
    """Money a held position commits: a long cost its price, a short (YES sold) one minus it."""
    return position * avg_price if position >= 0 else -position * (1 - avg_price)


@dataclass
class QueueSpot:
    """Where one of our resting sides stands in its level's queue, as the book shows it: everyone else's contracts
    ahead of it and the level's size apart from ours at the last record; and for the moment of that record (ts_ns),
    what was ahead before it and the trades at its price and the level's shrink seen at it, so the moment's records
    give the same place in whatever order they come and a trade that shows as both counts once."""
    price: float
    size: float
    ahead: float
    level: float
    ts_ns: int = 0
    before: float = 0.0
    traded: float = 0.0
    shrunk: float = 0.0


def others_at(levels: list[Level], price: float, ours: float) -> float:
    """Everyone else's size at a price on one side (levels best first); ours is our size there when the book shows
    it."""
    for level in levels:
        if math.isclose(level.price, price):
            return max(0.0, level.size - ours)
    return 0.0


def joined(price: float, size: float, others: float) -> QueueSpot:
    """A side just sent to a price: behind everyone already there."""
    return QueueSpot(price, size, others, others)


def advance(spot: QueueSpot, ts_ns: int, others: float, traded: float, cancel_ahead: float) -> None:
    """Moves a side's place by one record: traded is the size that traded at its price, others the level's size apart
    from ours after it. Trades come off the size ahead; of the level's other shrinks, cancel_ahead (0 to 1) is taken to
    be ahead of us. A shrink and a trade at the same moment are the same contracts as far as they overlap."""
    if ts_ns != spot.ts_ns:
        spot.ts_ns, spot.before, spot.traded, spot.shrunk = ts_ns, spot.ahead, 0.0, 0.0
    spot.traded += traded
    spot.shrunk += max(0.0, spot.level - others)
    cancels = max(0.0, spot.shrunk - spot.traded)
    spot.level = others
    spot.ahead = min(max(spot.before - spot.traded - cancel_ahead * cancels, 0.0), others)


def near_front(spot: QueueSpot, min_ahead: float, ahead_fraction: float) -> bool:
    """Less than min_ahead contracts, or less than ahead_fraction of everyone else's size at the level, ahead of it."""
    return spot.ahead < max(min_ahead, ahead_fraction * spot.level)


def skew_ticks(position: float, skew_unit: float, max_ticks: int) -> int:
    """Ticks the side that adds to the position steps back: one per skew_unit contracts held, at most max_ticks."""
    if skew_unit <= 0 or max_ticks <= 0:
        return 0
    return min(max_ticks, int(abs(position) // skew_unit))


def moved_steps(grid: PriceGrid, then: float, now: float, steps: int) -> bool:
    """Whether a price (a mid, so possibly between grid points) moved at least steps grid steps, counted from the
    lower price's grid point."""
    low, high = sorted((round(then * PRICE_SCALE), round(now * PRICE_SCALE)))
    base = grid.floor(low)
    return high - low >= grid.up(base, steps) - base


def placed(q: Quote, position: float, grid: PriceGrid, behind: int, skew: int, min_price: float,
           max_price: float) -> Quote:
    """The quote with each side stepped back from the price it joins: behind grid steps each, and the side that adds to
    the position (the bid when long, the ask when short) skew steps more, each step the grid's where it lands. A side
    stepped outside the price range is left alone."""
    bid_back = behind + (skew if position > 0 else 0)
    ask_back = behind + (skew if position < 0 else 0)
    bid, bid_size = q.bid_price, q.bid_size
    ask, ask_size = q.ask_price, q.ask_size
    if bid_size and bid_back:
        bid = grid.down(grid.floor(round(bid * PRICE_SCALE)), bid_back) / PRICE_SCALE
        if not min_price <= bid <= max_price:
            bid, bid_size = 0.0, 0.0
    if ask_size and ask_back:
        ask = grid.up(grid.ceil(round(ask * PRICE_SCALE)), ask_back) / PRICE_SCALE
        if not min_price <= ask <= max_price:
            ask, ask_size = 0.0, 0.0
    return Quote(bid, bid_size, ask, ask_size)
