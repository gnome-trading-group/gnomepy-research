# Building Strategies

This tutorial covers the signal library, strategy patterns, and code conventions for writing `strategy.py`.

---

## Strategy Basics

Every strategy subclasses `gnomepy.Strategy` and implements two methods:

```python
from gnomepy import ExecutionReport, Intent, Side, Strategy
from gnomepy.java.schemas import Schema

class MyStrategy(Strategy):
    def __init__(self, param_a: float = 1.0, param_b: int = 10):
        self.param_a = param_a
        self.param_b = param_b

    def on_market_data(self, data: Schema) -> list[Intent]:
        # Called on every market data tick.
        # Return a list of desired orders (Intents).
        return []

    def on_execution_report(self, report: ExecutionReport) -> list[Intent]:
        # Called when an order is filled, cancelled, or rejected.
        # Return [] unless you need reactive order logic.
        return []
```

**`Schema`** is the market data object (SBE-encoded MBP-10). Key accessors:
```python
data.bid_price(level)    # bid price at level 0-9 (scaled int)
data.ask_price(level)    # ask price at level 0-9 (scaled int)
data.bid_size(level)     # bid size at level 0-9
data.ask_size(level)     # ask size at level 0-9
data.exchange_id         # integer exchange identifier
data.security_id         # integer security identifier
data.event_timestamp     # nanoseconds since epoch
```

**Price scaling:** all prices in the engine are scaled integers. Divide by `Scales.PRICE` (≈ 1e9) to get a human-readable float. Never pass floats to `Intent` — the engine expects scaled integers. Example:
```python
# Wrong: passing a float
# Intent(..., bid_price=102.50)

# Right: prices from Schema are already scaled — pass them through directly
bid = data.bid_price(0)  # e.g. 102500000000 (scaled)
mid = (data.bid_price(0) + data.ask_price(0)) // 2
```

---

## Signal Library Overview

All signals live in `gnomepy_research/signals/` and are imported from `gnomepy_research.signals`.

The tables below are generated from the package's own `__all__` and docstrings — regenerate with
`poetry run python scripts/gen_signal_catalog.py` after adding or renaming a signal.
`tests/test_docs_sync.py` fails if they drift.

<!-- BEGIN GENERATED SIGNAL CATALOG -->

### Fair Value — 5 signals

`Signal[int]` — scaled integer prices, same units as bid/ask.

| Signal | Parameters | What it measures |
|--------|-----------|------------------|
| `MidFairValue` | — | Fair value = simple mid price. |
| `MicropriceFairValue` | — | L1 microprice: volume-weighted mid using top-of-book. |
| `WeightedMicropriceFairValue` | `num_levels=5, decay=0.5` | Multi-level weighted microprice using MBP book depth. |
| `ImbalanceAdjustedMid` | `num_levels=1, warmup=10` | Fair value: mid shifted by depth imbalance (Stoikov 2018). |
| `TradeAdjustedFairValue` | `flow_weight, flow_horizon_ns=1000000000, warmup=10` | Fair value: microprice adjusted by recent signed trade flow. |

### Volatility — 9 signals

`Signal[float]` — market noise and spread environment.

| Signal | Parameters | What it measures |
|--------|-----------|------------------|
| `SpreadVolatility` | `warmup_ticks=50, scale=1.0` | Volatility estimate derived from the bid-ask spread. |
| `RealizedVolatility` | `horizon=100, warmup=None` | Rolling standard deviation of mid-price log returns, in basis points. |
| `HighLowVolatility` | `horizon=100, warmup=None` | Parkinson (1980) volatility estimator using rolling high-low range. |
| `MicroVolatility` | `horizon=100, warmup=None` | Rolling standard deviation of microprice changes, in basis points. |
| `ReturnKurtosis` | `horizon=200, warmup=None` | Rolling excess kurtosis of mid-price log returns. |
| `SpreadVolRegime` | `alpha=0.999, warmup=100` | Current spread relative to its EWMA — a spread regime indicator. |
| `VolOfVol` | `inner_horizon=50, outer_horizon=100` | Rolling standard deviation of RealizedVolatility. |
| `BidAskBounce` | `horizon=50, warmup=50` | Fraction of mid-price changes that immediately reverse direction. |
| `VolatilityAsymmetry` | `horizon=100, warmup=None` | Ratio of upside volatility to downside volatility. |

### Flow — 22 signals

`Signal[float]` — trade activity and order-flow pressure.

| Signal | Parameters | What it measures |
|--------|-----------|------------------|
| `TradeImbalance` | `horizon_ns=1000000000, warmup_trades=20` | Trade-flow imbalance signal over a rolling horizon. |
| `Aggression` | `warmup_trades=20` | How far past mid the aggressor reached, in bps. |
| `Impact` | `warmup_trades=20` | Net book displacement from a trade, in bps. |
| `Reversion` | `warmup_trades=20` | Where the book settled relative to the trade price, in bps. |
| `LevelStaleness` | `side, level=0, warmup_ticks=1` | Staleness of a single book level in nanoseconds. |
| `LevelLiquidityDelta` | `side, level=0, horizon_ns=1000000000, warmup_events=20` | Net liquidity change at a specific book level over a rolling horizon. |
| `TradeArrivalTime` | `warmup_trades=2` | Time since the last trade event in nanoseconds. |
| `SignedVolume` | `horizon_ns=1000000000, warmup_trades=20` | Cumulative signed trade volume over a rolling window. |
| `TradeIntensity` | `horizon_ns=1000000000, warmup_trades=10` | Trade arrival rate within a rolling window, in trades per second. |
| `SweepDetector` | `window_ns=10000000, min_levels=2, warmup_trades=20` | Detects multi-level sweeps: consecutive same-side trades spanning multiple prices. |
| `TradeVWAP` | `horizon_ns=1000000000, warmup_trades=20` | Deviation of rolling trade VWAP from current mid, in basis points. |
| `CancelImbalance` | `horizon_ns=5000000000, warmup_events=20` | Net cancel volume imbalance over a rolling window. |
| `AddImbalance` | `horizon_ns=5000000000, warmup_events=20` | Net new-order volume imbalance over a rolling window. |
| `TradeClusterRate` | `horizon_ns=10000000000, cluster_ns=1000000, warmup_trades=50` | Fraction of trades arriving within cluster_ns of the previous trade. |
| `NetLiquidityDelta` | `horizon_ns=5000000000, warmup_events=20` | Net aggregate liquidity change (add - cancel) across all book levels. |
| `TradeSizeSkew` | `horizon=100, min_trades=None` | Rolling skewness of trade size distribution. |
| `MidMomentum` | `horizon=100, warmup=None` | Signed cumulative return normalized by realized vol — a t-statistic of trend. |
| `PriceAnchor` | `alpha=0.999, warmup=100` | Distance of current mid from its EWMA, in basis points. |
| `LevelMagnetism` | `size_threshold_mult=3.0, num_levels=10, warmup=10` | Distance from mid to the nearest abnormally thick resting price level. |
| `SpoofDetector` | `detection_window_ns=1000000000, warmup_events=20` | L1 cancel-to-add volume ratio over a short rolling window. |
| `TradeSizeEntropy` | `horizon_ns=60000000000, num_buckets=10, warmup_trades=50` | Rolling Shannon entropy of the trade size distribution. |
| `PriceImpactDecay` | `observation_ns=1000000000, alpha=0.95, warmup_trades=20` | Fraction of trade price impact remaining after observation_ns. |

### Book — 12 signals

`Signal[float]` — order book shape and imbalance.

| Signal | Parameters | What it measures |
|--------|-----------|------------------|
| `DepthImbalance` | `num_levels=5, warmup=10` | Volume-weighted depth imbalance across top N book levels. |
| `BookPressure` | `num_levels=5, decay=0.5, warmup=10` | Exponentially-weighted depth imbalance, prioritizing near levels. |
| `CountImbalance` | `num_levels=5, warmup=10` | Order count imbalance across top N book levels. |
| `TopHeaviness` | `num_levels=5, warmup=10` | Fraction of total book depth concentrated at L1. |
| `SpreadBps` | `warmup=10` | Bid-ask spread in basis points. |
| `BookSlope` | `num_levels=5, warmup=10` | OLS regression slope of cumulative depth vs price distance from mid. |
| `LevelConcentration` | `num_levels=5, warmup=10` | Herfindahl index of depth distribution across book levels. |
| `DepthRatio` | `num_levels=5, warmup=10` | Ratio of L1 depth to deeper levels, averaged across both sides. |
| `QueueImbalanceDelta` | `num_levels=5, diff_ticks=10, warmup=20` | Rate of change of depth imbalance over a fixed tick window. |
| `BookEntropy` | `num_levels=5, warmup=10` | Shannon entropy of the depth distribution across book levels. |
| `GapRisk` | `num_levels=5, warmup=10` | Weighted price gaps between consecutive book levels. |
| `SyntheticDepth` | `bps_radius=10.0, warmup=10` | Cumulative depth available within a fixed bps radius of mid. |

### Market State — 5 signals

`Signal[float]` — regime and meta-market signals.

| Signal | Parameters | What it measures |
|--------|-----------|------------------|
| `LiquidityScore` | `num_levels=5, alpha=0.99, warmup=100` | Current book depth normalized by its EWMA. |
| `ActivityRegime` | `fast_horizon_ns=1000000000, slow_horizon_ns=60000000000, warmup_trades=50` | Ratio of fast trade intensity to slow trade intensity. |
| `TickDirection` | `horizon=50, warmup=50` | Normalized rolling count of up-ticks minus down-ticks. |
| `ExchangeLatency` | `alpha=0.99, warmup=100` | EWMA of feed latency (timestamp_recv - timestamp_event), in nanoseconds. |
| `SequenceGap` | `horizon=100, warmup=100` | Rolling rate of detected sequence number gaps per tick. |

### Operations — 18 operators

Every operator wraps a signal and **preserves its type**, so an operator applied to a fair-value signal is still int-valued. Note the spelling `Zscore`, not `ZScore`.

| Operator | Parameters | What it does |
|----------|-----------|--------------|
| `Negate` | `_cls=<class 'abc.NegateOperation'>` | — |
| `Abs` | `_cls=<class 'abc.AbsOperation'>` | — |
| `Log` | `_cls=<class 'abc.LogOperation'>` | — |
| `EWMA` | `alpha=0.95, warmup=1` | Apply EWMA smoothing to any signal, preserving its type. |
| `Kalman` | `Q=0.0001, R=0.01, adaptive_alpha=None, r_floor=1e-10, warmup=1` | Apply Kalman filtering to any signal, preserving its type. |
| `Std` | `horizon=100` | Apply rolling std to any signal, preserving its type. |
| `PctChange` | `warmup=2` | Apply percent change to any signal, preserving its type. |
| `Skew` | `horizon=100` | Apply rolling skewness to any signal, preserving its type. |
| `Autocorrelation` | `horizon=100` | Apply rolling lag-1 autocorrelation to any signal, preserving its type. |
| `Weight` | `s, weights` | Weighted combination of signals. All signals must be the same type. |
| `Lag` | `n=1` | Apply n-tick lag to any signal, preserving its type. |
| `Diff` | `n=1` | Apply first difference to any signal, preserving its type. |
| `Zscore` | `horizon=100` | Apply rolling z-score normalization to any signal, preserving its type. |
| `RollingSum` | `horizon=100` | Apply rolling sum to any signal, preserving its type. |
| `RollingMax` | `horizon=100` | Apply rolling max to any signal, preserving its type. |
| `RollingMin` | `horizon=100` | Apply rolling min to any signal, preserving its type. |
| `Rank` | `horizon=100` | Apply rolling percentile rank to any signal, preserving its type. |
| `Clip` | `lo, hi` | Clamp any signal's output to [lo, hi], preserving its type. |

Each operator also has a lower-level `<Name>Operation` form (`ZscoreOperation`, `LagOperation`, ...) that you attach with `apply(op, signal)` when you want to build the operation once and reuse it.

Plus `PerAsset` to filter a signal to one listing, the `Weighted*` combiners (`WeightedFairValue`, `WeightedVolatility`, `WeightedFlow`, `WeightedBook`, `WeightedMarketState`) for weighted blends of same-type signals, and `is_trade_event(data)` to test whether a tick was a trade.

<!-- END GENERATED SIGNAL CATALOG -->

---

## Using Signals

All signals share the same interface. Constructor parameters vary per signal — check the catalog
above rather than assuming a `window=` argument; most use `horizon`, `horizon_ns`, `num_levels`, or
`warmup_trades`.

```python
from gnomepy_research.signals import EWMA, MicropriceFairValue, TradeImbalance

class MyStrategy(Strategy):
    def __init__(self):
        self._fair_value = MicropriceFairValue()
        self._flow = EWMA(TradeImbalance(horizon_ns=1_000_000_000), alpha=0.95, warmup=30)

    def on_market_data(self, data: Schema) -> list[Intent]:
        self._fair_value.update(data.event_timestamp, data)
        self._flow.update(data.event_timestamp, data)

        if not self._fair_value.is_ready() or not self._flow.is_ready():
            return []

        fv = self._fair_value.value()   # scaled int (same units as prices)
        flow = self._flow.value()       # float
        ...
```

Key rules:
- Call `.update(timestamp, data)` on every tick
- Check `.is_ready()` before using `.value()` — signals return undefined values during warmup
- `.reset()` clears internal state — call it if you need to restart a signal mid-session (rare)
- **FairValueSignal** returns `int` (scaled price units). All other signals return `float`.

---

## Composing Signals

Signals implement `+`, `-`, `*`, `/` and unary `-`, so they compose directly:

```python
from gnomepy_research.signals import (
    DepthImbalance, EWMA, MicropriceFairValue, MidFairValue, TradeImbalance, WeightedFlow, Zscore,
)

# Arithmetic composition — runs both signals, combines their values
spread_signal = MicropriceFairValue() - MidFairValue()   # signed difference

# Scale by a constant
skewed_fv = MicropriceFairValue() + 0.5 * DepthImbalance(num_levels=3)

# Operators wrap any signal and preserve its type
smooth_flow = EWMA(TradeImbalance(horizon_ns=1_000_000_000), alpha=0.99, warmup=50)
normalized  = Zscore(TradeImbalance(), horizon=200)

# Weighted combination of multiple signals (all must be the same type)
composite_flow = WeightedFlow(
    [TradeImbalance(horizon_ns=500_000_000), TradeImbalance(horizon_ns=5_000_000_000)],
    weights=[0.7, 0.3],
)
```

**Type preservation:** an operator applied to a `FairValueSignal` stays int-typed. Mixing a fair
value with a float signal produces a float composite — divide by `Scales.PRICE` before comparing it
to spread or flow signals.

**`PerAsset`** is a function, not a wrapper class, and it takes a signal **instance** plus the
listing to filter to. It forwards only matching ticks to the wrapped signal, which is how you build
cross-asset signals — give each listing its own instance and combine them:

```python
from gnomepy_research.signals import MicropriceFairValue, PerAsset

btc_micro = PerAsset(MicropriceFairValue(), security_id=1)
eth_micro = PerAsset(MicropriceFairValue(), security_id=2)
spread = btc_micro - eth_micro

# In on_market_data — feed every tick to each; PerAsset discards the ones it does not own
spread.update(data.event_timestamp, data)
if spread.is_ready():
    value = spread.value()
```

Pass `exchange_id=` as well when the same `security_id` trades on more than one venue.

---

## Intent-Based OMS

The OMS is intent-based: you declare the **desired state** of your orders, not individual order operations. The OMS diffs your desired state against what's currently live and sends only the necessary new/cancel/amend messages.

```python
from gnomepy import Intent, Side, OrderType

# Passive quote on both sides
return [
    Intent(
        exchange_id=data.exchange_id,
        security_id=data.security_id,
        bid_price=bid_price,          # scaled int
        bid_size=100_000,
        ask_price=ask_price,          # scaled int
        ask_size=100_000,
    )
]

# Aggressive (taker) order — BID buys, ASK sells
return [
    Intent(
        exchange_id=data.exchange_id,
        security_id=data.security_id,
        take_side=Side.BID,                 # BID = buy
        take_size=100_000,
        take_order_type=OrderType.MARKET,
    )
]

# Cancel all orders on a listing (return empty intent for it)
return []
```

**Side is named for the side you are creating, not the side you are hitting.** `Side.BID` buys and
`Side.ASK` sells, for both quotes and takes. To close a long position you take `Side.ASK`. Getting
this backwards is the single easiest way to build a strategy that loses money on a real signal — see
`gnomepy_research/strategies/momentum.py:97-121` for the canonical usage.

Setting a size to `0` cancels that side; `Intent(exchange_id, security_id)` with no price or size
fields cancels everything on the listing.

**Return `[]` from `on_execution_report`** unless you need to react to a fill. Common reactive use case: after a fill on one leg of an arb, immediately submit the hedge on the other leg. Most strategies don't need this.

**Position access:**
```python
pos = self.positions.get_position(exchange_id, security_id)
net_qty = pos.net_quantity if pos is not None else 0

# Effective quantity (net + pending orders) — use to avoid overshooting position limits
eff_qty = self.positions.get_effective_quantity(exchange_id, security_id) or 0
```

---

## Common Patterns

### Market Maker

Quote around fair value with a spread proportional to volatility:

```python
from gnomepy import Intent, Strategy
from gnomepy.java.schemas import Schema
from gnomepy_research.signals import MicropriceFairValue, EWMA, SpreadVolatility

class SimpleMarketMaker(Strategy):
    def __init__(
        self,
        size: int = 100_000,
        gamma: float = 1.0,
        max_position: int = 10,
    ):
        self.size = size
        self.gamma = gamma
        self.max_position = max_position
        self._fv = MicropriceFairValue()
        self._vol = EWMA(SpreadVolatility(warmup_ticks=50), alpha=0.95, warmup=20)

    def on_market_data(self, data: Schema) -> list[Intent]:
        self._fv.update(data.event_timestamp, data)
        self._vol.update(data.event_timestamp, data)

        if not self._fv.is_ready() or not self._vol.is_ready():
            return []

        fv = self._fv.value()
        half_spread = int(self._vol.value() * self.gamma)
        half_spread = max(half_spread, 1)

        pos = self.positions.get_position(data.exchange_id, data.security_id)
        net = pos.net_quantity if pos else 0
        if abs(net) >= self.max_position:
            return []

        return [Intent(
            exchange_id=data.exchange_id,
            security_id=data.security_id,
            bid_price=fv - half_spread,
            bid_size=self.size,
            ask_price=fv + half_spread,
            ask_size=self.size,
        )]

    def on_execution_report(self, report):
        return []
```

### Momentum / Mean-Reversion

Enter aggressively when a flow signal exceeds a threshold:

```python
from gnomepy import Intent, OrderType, Side, Strategy
from gnomepy.java.schemas import Schema
from gnomepy_research.signals import EWMA, TradeImbalance

class MomentumStrategy(Strategy):
    def __init__(self, threshold: float = 0.3, size: int = 100_000, horizon_ns: int = 1_000_000_000):
        self.threshold = threshold
        self.size = size
        self._signal = EWMA(TradeImbalance(horizon_ns=horizon_ns), alpha=0.95)

    def on_market_data(self, data: Schema) -> list[Intent]:
        self._signal.update(data.event_timestamp, data)
        if not self._signal.is_ready():
            return []

        flow = self._signal.value()
        if flow > self.threshold:
            return [Intent(
                exchange_id=data.exchange_id,
                security_id=data.security_id,
                take_side=Side.BID,
                take_size=self.size,
                take_order_type=OrderType.MARKET,
            )]
        elif flow < -self.threshold:
            return [Intent(
                exchange_id=data.exchange_id,
                security_id=data.security_id,
                take_side=Side.ASK,
                take_size=self.size,
                take_order_type=OrderType.MARKET,
            )]
        return []

    def on_execution_report(self, report):
        return []
```

### Cross-Exchange Arbitrage

Track spread between listings, enter when z-score exceeds threshold:

```python
# See gnomepy_research/sessions/n_exchange_arb/strategy.py for a full implementation.
# Key pattern:
#   - Maintain per-listing bid/ask state in dicts keyed by (exchange_id, security_id)
#   - Compute EWMA of spread across each pair
#   - When z-score exceeds entry_z, take aggressive orders on both legs simultaneously
#   - When z-score reverts below close_z, close both legs
#   - Track imbalance ticks and max hold time to force-close stuck positions
```

---

## Custom Metrics

Log per-tick diagnostics for the parquet analyzer in Step 5:

```python
def register_metrics(self) -> None:
    buf = self.metrics.create_buffer("diagnostics")
    self._col_ts    = buf.addLongColumn("timestamp")
    self._col_z     = buf.addDoubleColumn("z_score")
    self._col_state = buf.addLongColumn("state")
    buf.freeze()
    self._buf = buf

def _log(self, ts: int, z: float, state: int) -> None:
    if self._buf is None:
        return
    row = self._buf.appendRow()
    self._buf.setLong(row, self._col_ts, ts)
    self._buf.setDouble(row, self._col_z, z)
    self._buf.setLong(row, self._col_state, state)
```

Custom metric buffers appear in `report.custom_metrics()` and can be analyzed in `notebooks/02_backtest_deep_dive.ipynb` under "Signal Attribution".

---

## Code Conventions

These apply to every `strategy.py`:

1. **All imports at the top** — never inside functions, methods, or conditionals
2. **No comments unless the WHY is non-obvious** — the signal names and variable names should be self-documenting
3. **Prices are scaled integers** — never pass floats to `Intent`; divide by `Scales.PRICE` for display only
4. **`on_execution_report` returns `[]`** by default — only implement reactive logic if you specifically need it
5. **`__init__` must accept all tunable parameters as kwargs** — the backtest config's `strategy.args` maps directly to constructor kwargs; parameters that aren't in `__init__` cannot be swept
6. **Strategy lives in one file** — `strategy.py`. Helper classes can be in the same file or extracted to a new module in `gnomepy_research/` and imported at the top
7. **Position limits must be enforced** — always check net quantity against `spec.constraints.avoid` rules before emitting intents

---

## Debugging Tips

**JVM or JPype errors:** Usually caused by passing float prices to `Intent`, or calling `.value()` on an unready signal and passing the result to a price field. Add `is_ready()` guards.

**Zero fills:** Check that bid/ask prices are valid (`bid > 0 and ask > 0`) before computing fair value. An invalid spread (bid >= ask) causes the queue model to reject the order.

**Position never closes:** If using aggressive close orders, verify the side is correct — to close a long position, take `Side.ASK` (sell). `Side.BID` buys and `Side.ASK` sells.

**Strategy not found by backtest runner:** Ensure `__init__.py` exists in `gnomepy_research/sessions/<name>/` and the `class_name` in the config exactly matches `gnomepy_research.sessions.<name>.strategy:ClassName`.

**Sweeps don't improve over local results:** The local iteration used fixed parameters; the sweep may have explored the wrong range. Read the winning sweep job's parameters from `results/iter_NNN/<job_id>/summary.json` and check whether they match your expectation.

---

## Next Steps

- Read **`01_getting_started.md`** for session creation and first iteration
- Read **`02_research_workflow.md`** for evaluation, notes, and validation
- Open `notebooks/01_signal_exploration.ipynb` to explore signal behavior on real data before building a strategy
- Browse `gnomepy_research/signals/` for the full signal catalog — each file has a docstring explaining the formula and parameters
