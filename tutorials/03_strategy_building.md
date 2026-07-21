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
# Intent(..., quote_bid_price=102.50)

# Right: prices from Schema are already scaled — pass them through directly
bid = data.bid_price(0)  # e.g. 102500000000 (scaled)
mid = (data.bid_price(0) + data.ask_price(0)) // 2
```

---

## Signal Library Overview

All signals live in `gnomepy_research/signals/` and are imported from `gnomepy_research.signals`.

### Fair Value — `Signal[int]`

Estimate where the true price is. Values are scaled integers (same units as bid/ask prices).

| Signal | What it measures |
|--------|-----------------|
| `MidFairValue` | Simple arithmetic mid `(bid + ask) / 2` |
| `MicropriceFairValue` | Volume-weighted mid: `(bid × ask_size + ask × bid_size) / (bid_size + ask_size)` — shifts toward the side with more pressure |
| `WeightedMicropriceFairValue(num_levels=5, decay=0.5)` | Multi-level microprice with exponential decay across book levels |
| `ImbalanceAdjustedMid` | Mid adjusted by depth imbalance signal |
| `TradeAdjustedFairValue` | Mid adjusted by recent trade flow |

### Volatility — `Signal[float]`

Measure market noise and spread environment. Values are floats.

| Signal | What it measures |
|--------|-----------------|
| `SpreadVolatility(window=20)` | EWMA variance of bid-ask spread |
| `RealizedVolatility(window=50)` | Rolling realized variance of mid returns |
| `HighLowVolatility(window=20)` | Range-based volatility estimator |
| `MicroVolatility(window=20)` | Tick-by-tick microprice variance |
| `BidAskBounce(window=20)` | Fraction of ticks where mid bounces between bid and ask |
| `VolOfVol(window=100)` | Volatility of volatility |
| `SpreadVolRegime(window=50)` | Discrete regime: 0=calm, 1=elevated, 2=stressed |

### Flow — `Signal[float]`

Measure trade activity and order flow pressure.

| Signal | What it measures |
|--------|-----------------|
| `TradeImbalance(window=20)` | `(buy_vol - sell_vol) / (buy_vol + sell_vol)` in [-1, 1] |
| `SignedVolume(window=20)` | Cumulative signed trade volume |
| `Aggression(window=20)` | Fraction of trades that were aggressive (market orders) |
| `Impact(window=20)` | Average mid-price impact per trade |
| `TradeIntensity(window=20)` | Trades per unit time |
| `SweepDetector(threshold=3)` | 1.0 when a sweep through multiple price levels occurred |
| `MidMomentum(window=20)` | Short-term trend in mid price returns |
| `CancelImbalance(window=20)` | `(bid_cancels - ask_cancels) / total` |
| `SpoofDetector(window=20)` | Large order add + cancel within N ticks |
| `PriceImpactDecay(window=20)` | How quickly price impact reverts |

### Book — `Signal[float]`

Measure order book shape and imbalance.

| Signal | What it measures |
|--------|-----------------|
| `DepthImbalance(levels=5)` | `(bid_depth - ask_depth) / (bid_depth + ask_depth)` across N levels |
| `BookPressure(levels=5)` | Price-weighted depth imbalance |
| `CountImbalance(levels=5)` | Imbalance by order count rather than size |
| `SpreadBps` | Current spread in basis points |
| `BookSlope(levels=5)` | How steeply the book thins out from L1 |
| `TopHeaviness(levels=5)` | Fraction of depth concentrated at L1 |
| `BookEntropy(levels=5)` | Shanon entropy of depth distribution |
| `GapRisk(levels=5)` | Largest price gap between adjacent levels |
| `SyntheticDepth(levels=5)` | Hypothetical market impact of a given order size |

### Market State — `Signal[float]`

High-level regime and meta-market signals.

| Signal | What it measures |
|--------|-----------------|
| `LiquidityScore(window=20)` | Composite liquidity score (depth × tightness) |
| `ActivityRegime(window=50)` | Discrete activity level: 0=quiet, 1=normal, 2=active |
| `TickDirection(window=10)` | Recent tick direction: +1 up, -1 down, 0 mixed |
| `ExchangeLatency(window=20)` | Estimated exchange processing latency from timestamp gaps |
| `SequenceGap(window=20)` | Missing sequence numbers (connectivity issues) |

---

## Using Signals

All signals share the same interface:

```python
from gnomepy_research.signals import MicropriceFairValue, TradeImbalance, EWMA

class MyStrategy(Strategy):
    def __init__(self):
        self._fair_value = MicropriceFairValue()
        self._flow = EWMA(TradeImbalance(window=30), alpha=0.95, warmup=30)

    def on_market_data(self, data: Schema) -> list[Intent]:
        self._fair_value.update(data.event_timestamp, data)
        self._flow.update(data.event_timestamp, data)

        if not self._fair_value.is_ready() or not self._flow.is_ready():
            return []

        fv = self._fair_value.value()   # scaled int (same units as prices)
        flow = self._flow.value()       # float in [-1, 1]
        ...
```

Key rules:
- Call `.update(timestamp, data)` on every tick
- Check `.is_ready()` before using `.value()` — signals return undefined values during warmup
- `.reset()` clears internal state — call it if you need to restart a signal mid-session (rare)
- **FairValueSignal** returns `int` (scaled price units). All other signals return `float`.

---

## Composing Signals

Signals support arithmetic operators. The result is always a `float`-valued composite signal:

```python
from gnomepy_research.signals import MicropriceFairValue, MidFairValue, DepthImbalance, EWMA, Weight

# Arithmetic composition — runs both signals, combines their values
spread_signal = MicropriceFairValue() - MidFairValue()   # signed difference

# Scale by a constant
skewed_fv = MicropriceFairValue() + 0.5 * DepthImbalance(levels=3)

# EWMA smoothing — wraps any signal
smooth_flow = EWMA(TradeImbalance(window=20), alpha=0.99, warmup=50)

# Weighted combination of multiple signals (all must be the same type)
from gnomepy_research.signals import WeightedFlow
composite_flow = WeightedFlow(
    [TradeImbalance(window=10), TradeImbalance(window=50)],
    weights=[0.7, 0.3],
)
```

**Type preservation:** `FairValueSignal op FairValueSignal` → `WeightedFairValue` (still int-typed). Mixing fair value with a float signal (e.g. `MicropriceFairValue() + DepthImbalance()`) produces a float composite — divide by `Scales.PRICE` before comparing to spread or flow signals.

**`PerAsset`** — wraps any single-asset signal to maintain per-listing state in multi-exchange strategies:
```python
from gnomepy_research.signals import PerAsset, MicropriceFairValue

self._fv = PerAsset(MicropriceFairValue)   # creates one instance per listing seen

# In on_market_data:
listing_key = (data.exchange_id, data.security_id)
self._fv.update(listing_key, data.event_timestamp, data)
fv = self._fv.value(listing_key)
```

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
        quote_bid_price=bid_price,          # scaled int
        quote_bid_size=100_000,
        quote_ask_price=ask_price,          # scaled int
        quote_ask_size=100_000,
    )
]

# Aggressive (taker) order
return [
    Intent(
        exchange_id=data.exchange_id,
        security_id=data.security_id,
        take_side=Side.ASK,                 # ASK = buy at ask price
        take_size=100_000,
        take_order_type=OrderType.MARKET,
    )
]

# Cancel all orders on a listing (return empty intent for it)
return []
```

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
        self._vol = EWMA(SpreadVolatility(window=20), alpha=0.95, warmup=20)

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
            quote_bid_price=fv - half_spread,
            quote_bid_size=self.size,
            quote_ask_price=fv + half_spread,
            quote_ask_size=self.size,
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
    def __init__(self, threshold: float = 0.3, size: int = 100_000, window: int = 30):
        self.threshold = threshold
        self.size = size
        self._signal = EWMA(TradeImbalance(window=window), alpha=0.95)

    def on_market_data(self, data: Schema) -> list[Intent]:
        self._signal.update(data.event_timestamp, data)
        if not self._signal.is_ready():
            return []

        flow = self._signal.value()
        if flow > self.threshold:
            return [Intent(
                exchange_id=data.exchange_id,
                security_id=data.security_id,
                take_side=Side.ASK,
                take_size=self.size,
                take_order_type=OrderType.MARKET,
            )]
        elif flow < -self.threshold:
            return [Intent(
                exchange_id=data.exchange_id,
                security_id=data.security_id,
                take_side=Side.BID,
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

**Position never closes:** If using aggressive close orders, verify the side is correct — to close a long position, take at the **ask** (sell), not bid. `Side.ASK` = sell at ask in the engine's convention.

**Strategy not found by backtest runner:** Ensure `__init__.py` exists in `gnomepy_research/sessions/<name>/` and the `class_name` in the config exactly matches `gnomepy_research.sessions.<name>.strategy:ClassName`.

**Sweeps don't improve over local results:** The local iteration used fixed parameters; the sweep may have explored the wrong range. Read the winning sweep job's parameters from `results/iter_NNN/<job_id>/summary.json` and check whether they match your expectation.

---

## Next Steps

- Read **`01_getting_started.md`** for session creation and first iteration
- Read **`02_research_workflow.md`** for evaluation, notes, and validation
- Open `notebooks/01_signal_exploration.ipynb` to explore signal behavior on real data before building a strategy
- Browse `gnomepy_research/signals/` for the full signal catalog — each file has a docstring explaining the formula and parameters
