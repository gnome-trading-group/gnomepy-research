# gnomepy-research

Research and production models for the gnome trading system: the signal library, research sessions
(strategies, backtests and their results), and the production models that gnome-pipelines trains and
live strategies load.

## Install

```bash
poetry install --all-extras
poetry run pytest
```

Requires Python 3.13 and `gnomepy`.

## Layout

- `gnomepy_research/models/` — production models and point-in-time features (CS2, CFB); released by tag
- `gnomepy_research/sessions/<name>/` — one directory per research session: `spec.yaml`, `strategy.py`, `configs/`, `results/`
- `gnomepy_research/signals/` — composable `Signal` library (fair value, volatility, flow, book, market state, operators)
- `gnomepy_research/arb/` — shared machinery for the prediction-market arb sessions
- `gnomepy_research/analysis/` — loading and diagnosing backtest results, significance and walk-forward validation
- `tutorials/` — getting started, research workflow, strategy building, artifacts and datasets, analysis toolkit

`CLAUDE.md` has the full conventions: where code goes, session structure, dependency families and
releases.

## Signals

```python
from gnomepy_research.signals import EWMA, MicropriceFairValue, PerAsset

fv = EWMA(MicropriceFairValue(), alpha=0.95)

for ts, tick in stream:
    fv.update(ts, tick)
    if fv.is_ready():
        print(fv.value())

btc = PerAsset(MicropriceFairValue(), security_id=1)
eth = PerAsset(MicropriceFairValue(), security_id=2)
spread = btc - eth
```
