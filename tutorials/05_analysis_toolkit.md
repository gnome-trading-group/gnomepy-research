# Analysis Toolkit

Everything in `gnomepy_research` that analyses a result rather than producing one. Most of this
existed for months without being referenced anywhere, so the research loop was re-deriving from
parquet what was already implemented and tested.

**Imports are submodule-qualified.** `gnomepy_research/__init__.py` and several package `__init__.py`
files are empty, so `from gnomepy_research.validation import deflated_sharpe_ratio` fails — use
`from gnomepy_research.validation.statistics import deflated_sharpe_ratio`.

---

## Before you write a strategy — `explore`

`gnomepy_research/explore.py` loads market data and evaluates signals without running a backtest.
Use it to check a signal actually separates before building a strategy around it.

```python
from gnomepy_research.explore import (
    compare_results, compute_signals, load_datastore, load_market_data, load_results, plot_signals,
)
from gnomepy_research.signals import DepthImbalance, MicropriceFairValue, TradeImbalance

market = load_market_data(listing_id=222852, start="2026-08-24T00:25:00", end="2026-08-24T03:11:00")

source = load_datastore(listing_id=222852, start="2026-08-24T00:25:00", end="2026-08-24T03:11:00")
signals = compute_signals(source, {
    "fair_value": MicropriceFairValue(),
    "depth_imb":  DepthImbalance(num_levels=5),
    "flow":       TradeImbalance(horizon_ns=1_000_000_000),
})

plot_signals(market, signals, signals_to_plot=["depth_imb", "flow"]).show()
```

After a run, `load_results(path)` gives a `BacktestReport` from a results directory (local or S3),
and `compare_results(*paths, metric="sharpe")` tabulates several runs side by side.

---

## Diagnosing a result — `reporting.backtest`

These are what `/research` Step 5 calls. Each takes the `BacktestReport` from `load_results`.

```python
from gnomepy_research.explore import load_results
from gnomepy_research.reporting.backtest.adverse_selection import compute_adverse_selection
from gnomepy_research.reporting.backtest.market_making import compute_mm_stats, plot_mm_dashboard
from gnomepy_research.reporting.backtest.rolling_performance import (
    compute_alpha_decay, compute_rolling_sharpe, detect_regimes, pnl_by_regime,
)

report = load_results("gnomepy_research/sessions/<name>/results/iter_007")
```

| Call | Gives you |
|---|---|
| `compute_mm_stats(report)` | `time_quoting`, `time_quoting_both_sides`, `avg_edge_captured_bps`, `avg_net_edge_captured_bps`, `avg_quoted_spread_bps`, `avg_market_spread_bps`, `quoted_vs_market_spread`, `quote_to_fill_ratio`, `buy_sell_fill_ratio`, `bid_improvement_bps`, `ask_improvement_bps`, `mean_abs_position`, `max_abs_position`, `position_turnover`, `total_intents` |
| `compute_adverse_selection(fills, market_df, horizons_ms=None)` | Mean post-fill price move per horizon — negative means you are being picked off |
| `compute_rolling_sharpe(pnl_curve, window="1h", bar="10s")` | Is the edge stable across the session? |
| `compute_alpha_decay(pnl_curve, window="1h", bar="10s")` | Does the edge degrade as the session runs? |
| `detect_regimes(market_df, vol_window="5min")` + `pnl_by_regime(pnl_curve, regimes)` | Does PnL come from one market condition only? |
| `plot_mm_dashboard(report)` | The market-making panel as a figure |

---

## Is the result real? — `validation`

### `validation.statistics`

```python
from gnomepy_research.validation.statistics import (
    bootstrap_sharpe_ci, deflated_sharpe_ratio, expected_max_sharpe,
    minimum_backtest_length, sharpe_standard_error,
)
```

All Sharpe inputs are **per-bar** at 10s bars, not annualized — see
`tutorials/02_research_workflow.md`. The CLI wraps these:

```bash
poetry run research validate significance <session> \
  --fills .../fills.parquet --market .../market.parquet --n-trials 240 --json
```

`--n-trials` is every strategy variant evaluated so far, including each sweep job — not the
iteration number.

### `validation.walk_forward`

Out-of-sample evaluation across date folds. **No refitting happens between folds** — parameters are
fixed from the best iteration — so these are evaluation windows, not train/test splits.

```bash
poetry run research validate walk-forward <session> \
  --config .../configs/iter_019.yaml --start 2026-02-01T00:00:00 --end 2026-03-01T00:00:00 \
  --folds 5 --mode rolling
```

`rolling` gives equal disjoint windows marching forward; `expanding` anchors every window at
`--start` and grows it.

### `validation.monte_carlo`

Robustness against simulation assumptions. The two generators emit configs with a top-level `sweep:`
section, which is the only part `gnomepy.sweep.expand_sweep` expands.

```python
from gnomepy_research.validation.monte_carlo import (
    bootstrap_pnl_paths, generate_latency_sweep_config, generate_queue_sweep_config, summarize_mc_paths,
)

# Profiles sweep independently, so job count is len(values) ** len(profiles).
# Pass profile_names to vary one venue at a time.
generate_latency_sweep_config(base, "configs/sens_latency.yaml", profile_names=["kalshi"])
generate_queue_sweep_config(base, "configs/sens_queue.yaml", profile_names=["kalshi"])

# Path-luck check that needs no engine re-run:
summarize_mc_paths(bootstrap_pnl_paths(report.fills, report.market_df))
```

---

## Which signal is carrying the strategy? — `analysis.signal_attribution`

```python
from gnomepy_research.analysis.signal_attribution import (
    attribute_pnl_by_signal, generate_ablation_configs, summarize_ablation_results,
)

# Attribute realized PnL to signal values at fill time. Requires the strategy to log those
# signals via register_metrics (see tutorials/03_strategy_building.md).
#
# The buffer name is whatever the strategy passed to self.metrics.create_buffer(...) —
# cross_prediction_arb registers "cpa_signals". Call custom_metrics() with no argument
# to see which buffers a run actually has.
report.custom_metrics()                      # -> {"cpa_signals": DataFrame, ...}
signals = report.custom_metrics("cpa_signals")

attribute_pnl_by_signal(report.fills, signals, signal_columns=["edge", "qty"])

# Turn each signal off in turn and compare.
configs = generate_ablation_configs(base_config_path, {"use_flow_signal": False,
                                                       "use_depth_signal": False}, "configs/ablation")
# ...run each, then:
summarize_ablation_results(["use_flow_signal", "use_depth_signal"],
                           baseline_summary, ablated_summaries, primary_metric="final_pnl")
```

Ablation answers "does this signal earn its place?" far more directly than a parameter sweep.

---

## Comparing strategies — `analysis.portfolio`

```python
from gnomepy_research.analysis.portfolio import (
    combined_sharpe, compute_cross_strategy_correlation, plot_correlation_matrix,
)

corr = compute_cross_strategy_correlation(
    ["sessions/a/results/iter_010", "sessions/b/results/iter_007"],
    session_names=["a", "b"], resample_bar="1min",
)
plot_correlation_matrix(corr).show()
combined_sharpe([curve_a, curve_b], weights=[0.5, 0.5])
```

Low correlation between two profitable strategies is worth more than a marginal improvement in
either one.

---

## Prediction-market arbitrage — `arb`

Shared machinery behind the arb sessions, rather than each session reimplementing pricing and
book-walking.

```python
from gnomepy_research.arb import (
    ArbContext, ArbLeg, ArbPhase, ArbPortfolio, ArbBudgetConstraint, LegConstraint,
    CostModel, CompositeCostModel, FeeCost, FillRiskCost, DepthCoverageCost,
    PriceModel, PriceResult, JoinBestBidModel, AggressivePriceModel, FillProbTargetModel, OptimalEVModel,
    ContractGroup, OutcomeLeg, VenuePairing, discover_group, enumerate_pairings,
    OrderMode, walk_books,
)
```

- **`discover_group(...)` / `enumerate_pairings(...)`** — find the contracts across venues that form a complete outcome set, and the pairings that should sum to $1.
- **`walk_books(...)`** — realistic fill price and size for a given quantity against real depth, instead of assuming top-of-book.
- **Cost models** — `FeeCost` for parametric fees, plus `FillRiskCost` and `DepthCoverageCost`; compose with `CompositeCostModel`.
- **Price models** — `JoinBestBidModel` (rest in the book), `AggressivePriceModel` (cross), `FillProbTargetModel`, `OptimalEVModel`.
- **`ArbPortfolio` / `ArbPhase`** — leg state tracking through the entry/unwind lifecycle.

`gnomepy_research/research_learnings.md` records which of these worked on live sessions — notably
that maker-first (`JoinBestBidModel`) beats taker on venues where latency loses the race, and that
`DepthCoverageCost` had no effect when the raw edge dwarfed the penalty.

---

## Reproducibility — `environment`

```python
from gnomepy_research.environment import capture_environment
capture_environment()   # gnomepy version, JVM/engine metadata, research git commit, platform
```

`research iterations record-from-results` already calls this, so every recorded iteration carries the
commit and engine version it ran against.

---

## Next Steps

- **`02_research_workflow.md`** — the evaluation → iterate → validate lifecycle
- **`03_strategy_building.md`** — the signal catalog and strategy patterns
