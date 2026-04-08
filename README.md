# gnomepy-research

Signals library for the gnome trading system.

## Install

```bash
poetry install
```

Requires Python 3.13 and `gnomepy`.

## Usage

```python
from gnomepy_research.signals import (
    MicropriceFairValue,
    SpreadVolatility,
    TradeImbalance,
    EWMA,
    PerAsset,
)

# A signal ingests JavaMBP10Schema ticks and exposes value()
fv = EWMA(MicropriceFairValue(), alpha=0.95)

for ts, tick in stream:
    fv.update(ts, tick)
    if fv.is_ready():
        print(fv.value())

# Compose with arithmetic and per-asset filtering
btc = PerAsset(MicropriceFairValue(), security_id=1)
eth = PerAsset(MicropriceFairValue(), security_id=2)
spread = btc - eth
```

## Layout

- `signals/base.py` — `Signal[T]` base class with arithmetic operators
- `signals/fair_value/` — `MidFairValue`, `MicropriceFairValue`, ...
- `signals/volatility/` — `SpreadVolatility`, ...
- `signals/flow/` — `TradeImbalance`, `Aggression`, `Impact`, ...
- `signals/operations/` — type-preserving wrappers: `EWMA`, `Kalman`, `Std`, `Log`, `Weight`, `PerAsset`, ...
