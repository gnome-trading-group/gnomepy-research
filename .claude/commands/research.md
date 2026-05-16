# Strategy Research Iteration

Run one research iteration for session: **$ARGUMENTS**

All paths below are relative to the gnomepy-research project root (your current working directory).

---

## Step 1: Orient

Read the session spec:
```
gnomepy_research/sessions/$ARGUMENTS/spec.yaml
```

Read the session log (may not exist yet if this is iteration 1):
```
gnomepy_research/sessions/$ARGUMENTS/session.json
```

If `session.json` doesn't exist, this is iteration 1. Initialize it:
```json
{
  "session_name": "$ARGUMENTS",
  "status": "running",
  "created_at": "<now ISO8601>",
  "iterations": [],
  "best_iteration": null,
  "best_pnl": null,
  "best_sharpe": null
}
```

Determine the next iteration number: `len(iterations) + 1`.

**Branch management:**
- If iteration 1: create and check out a session branch:
  ```bash
  git checkout -b research/$ARGUMENTS
  ```
- If not iteration 1: ensure you're on the session branch:
  ```bash
  git checkout research/$ARGUMENTS
  ```

---

## Step 2: Hypothesize

**Check for user hints:**
Read `gnomepy_research/sessions/$ARGUMENTS/hints.md` if it exists. If it has content (one or more timestamped hints written by `/research-hint`), treat them as user directives that take priority over the session history's `next_action`. After reading, clear the file by writing an empty string to it.

**Interaction mode** (from `spec.meta.interaction_mode`, defaults to `autonomous`):

- **`interactive`**: Use `AskUserQuestion` before forming your hypothesis. Present:
  - A 2-3 sentence summary of the last iteration's results (or "This is iteration 1" if first)
  - Your proposed direction for this iteration
  - Options: "Proceed with this direction" / "I have a different idea" (via Other)
  Incorporate any user input into the hypothesis.

- **`autonomous`**: Skip the question — proceed with hints.md content (if any) or the session history's `next_action`.

**Iteration 1:** Design an initial strategy from scratch based on `spec.description` and `spec.constraints`. Look at the existing strategies in `gnomepy_research/strategies/` for patterns. Choose signals from `gnomepy_research/signals/` that match the description. State your strategy design as your hypothesis.

**Later iterations:** Read the last entry's `analysis` and `next_action`. If hints.md had content, use it instead of `next_action`. Otherwise follow `next_action` unless a better approach is evident from the full history. If the last 3+ iterations show no >5% improvement in `spec.goals.primary_metric`, try something fundamentally different — different signals, different strategy class, or a new structural approach.

Always write your hypothesis explicitly before making any changes. Example: "Hypothesis: Adding a TradeImbalance signal to skew the reservation price will reduce adverse selection by pulling quotes away from the active side."

---

## Step 3: Decide — Local Run or Remote Sweep?

First, check `spec.meta.execution_mode` (defaults to `auto` if absent):

- **`local`** — always use a local run, regardless of what changed.
- **`batch`** — always submit a remote sweep, regardless of what changed.
- **`auto`** — apply the rules below to decide:

**Use a local run** when:
- This is a logic change (new signals, new strategy structure, new intent logic)
- You're validating that the strategy works at all before sweeping parameters
- You're debugging a bad result from a previous iteration

**Use a remote sweep** when:
- The strategy logic is sound but you want to find better parameter values
- You've identified 2+ parameters worth searching (e.g., gamma, delta, min_spread_bps)
- The previous local iteration was profitable and you want to optimize

---

## Step 4A: Local Run

### 4A.1 — Write/modify the strategy

Write the strategy to:
```
gnomepy_research/sessions/$ARGUMENTS/strategy.py
```

Ensure `gnomepy_research/sessions/$ARGUMENTS/__init__.py` exists (create it empty if not).

Rules:
- Subclass `gnomepy.Strategy`
- No comments unless the WHY is non-obvious
- Use signals from `gnomepy_research.signals`
- Follow the pattern in `gnomepy_research/strategies/market_maker.py` (quoting) or `momentum.py` (taking)
- Return `[]` from `on_execution_report` unless reactive logic is needed
- **Not everything needs to live in `strategy.py`.** If you need a signal or reusable component that doesn't exist yet in `gnomepy_research/signals/`, create it there following the existing signal patterns. If you need a reusable non-signal utility, add it to `gnomepy_research/` as a new module. Import it from `strategy.py` as you would any other package code. Stage any new files in the Step 6 commit.

### 4A.2 — Write the backtest config

Write a config YAML to:
```
gnomepy_research/sessions/$ARGUMENTS/configs/iter_NNN.yaml
```
(where NNN is the zero-padded iteration number, e.g. `iter_001.yaml`)

The output directory for this config is `gnomepy_research/sessions/$ARGUMENTS/results/iter_NNN` — use this exact path as the `--output` value in Step 4A.3.

Structure — populate all values from `spec.yaml`:
```yaml
strategy:
  class_name: "gnomepy_research.sessions.$ARGUMENTS.strategy:YourStrategyClassName"
  args:
    # constructor kwargs for this iteration — must match the strategy's __init__ signature

start_date: "<spec.data.date_ranges[0].start>"
end_date: "<spec.data.date_ranges[0].end>"

listings:
  - listing_id: <spec.data.listings[0].listing_id>
    profile: <spec.data.listings[0].profile>
  # repeat for each listing in spec

profiles:
  <profile_name>:
    fee_model:
      type: static
      taker_fee: <spec.profiles.<name>.fee_model.taker_fee>
      maker_fee: <spec.profiles.<name>.fee_model.maker_fee>
    network_latency:
      type: static
      latency_nanos: <spec.profiles.<name>.network_latency.latency_nanos>
    order_processing_latency:
      type: static
      latency_nanos: <spec.profiles.<name>.order_processing_latency.latency_nanos>
    queue_model:
      type: <spec.profiles.<name>.queue_model.type>
  # repeat for each profile in spec
```

### 4A.3 — Run the backtest

**Always pass `--output`. Without it, results land in the project root as UUID-named directories.**

```bash
poetry run gnomepy backtest run \
  --config gnomepy_research/sessions/$ARGUMENTS/configs/iter_NNN.yaml \
  --output gnomepy_research/sessions/$ARGUMENTS/results/iter_NNN
```

After completion, check the output directory for results. Look for `summary.json` first; if present, parse it for metrics. If only `report.html` is present, note the path for the user and extract what metrics you can.

If the run fails (non-zero exit, JVM error, or no output), diagnose the issue. If the root cause is in the strategy code, fix it and rerun. If the root cause appears to be a bug in the backtesting engine itself (unexpected crash, wrong metric values, inconsistent parquet output unrelated to strategy logic), do NOT attempt to work around it — report the bug clearly to the user (engine version, config, error/symptom, steps to reproduce) and stop the iteration.

---

## Step 4B: Remote Sweep (AWS Batch)

### 4B.1 — Generate sweep config

Write a sweep config to:
```
gnomepy_research/sessions/$ARGUMENTS/configs/sweep_NNN.yaml
```

Use the same structure as the local config (Step 4A.2), but use list or range syntax in `strategy.args` for parameters to sweep:

```yaml
strategy:
  args:
    gamma: [0.5, 1.0, 2.0, 4.0]            # list sweep
    delta: {min: 0.5, max: 3.0, step: 0.5} # range sweep
```

**Hard cap: the cartesian product of all sweep parameters must not exceed 100 jobs.** Before writing the config, calculate the total job count (product of all list/range lengths). If it exceeds 100, reduce the parameter grid — coarsen the ranges, drop less important parameters, or split into multiple iterations.

### 4B.2 — Commit and push the session branch

The Batch container checks out the gnomepy-research repo at the given commit, so the strategy must be committed before submitting.

```bash
git add gnomepy_research/sessions/$ARGUMENTS/
git commit -m "research/$ARGUMENTS: iteration N"
git push -u origin research/$ARGUMENTS
COMMIT_SHA=$(git rev-parse HEAD)
```

### 4B.3 — Submit and poll

```bash
poetry run gnomepy backtest submit \
  --config gnomepy_research/sessions/$ARGUMENTS/configs/sweep_NNN.yaml \
  --research-commit $COMMIT_SHA
```

Save the returned `run_id`. Poll status until complete:
```bash
poetry run gnomepy backtest status <run_id>
```

### 4B.4 — Retrieve and evaluate results

Download all sweep job artifacts to the session results directory:

```bash
poetry run gnomepy backtest results <run_id> \
  --output gnomepy_research/sessions/$ARGUMENTS/results/iter_NNN
```

This downloads all output artifacts (parquet files, summary.json, etc.) for every sweep job into subdirectories under `iter_NNN/`. Find the job with the best value for `spec.goals.primary_metric` by reading each job's `summary.json`. Record that job's parameters as the winning sweep configuration.

Once the best job is identified, run the same parquet analysis as for local runs (see Step 5) on that job's artifacts. Present the winning job's parameters and metrics to the user.

---

## Step 5: Evaluate Results

**Start with `summary.json`** — it contains all high-level metrics (PnL, Sharpe, fill count, etc.). Read it first to understand overall performance. For sweeps, find the best job by comparing each job's `summary.json` against `spec.goals.primary_metric`.

**Parquet analysis** — dig into the parquet files to understand *why* the metrics came out as they did before forming your next hypothesis. The results directory contains `fills.parquet`, `orders.parquet`, `intents.parquet`, and `market.parquet`. Load and query them with:

```bash
poetry run python3 - <<'EOF'
import pandas as pd
fills   = pd.read_parquet("gnomepy_research/sessions/$ARGUMENTS/results/iter_NNN/fills.parquet")
orders  = pd.read_parquet("gnomepy_research/sessions/$ARGUMENTS/results/iter_NNN/orders.parquet")
intents = pd.read_parquet("gnomepy_research/sessions/$ARGUMENTS/results/iter_NNN/intents.parquet")
market  = pd.read_parquet("gnomepy_research/sessions/$ARGUMENTS/results/iter_NNN/market.parquet")
# explore as needed
print(fills.dtypes)
print(fills.head())
EOF
```

Investigate fill timing, fill quality, PnL attribution, order lifecycle, and market context.

**Custom metrics** — compute and record any derived metrics that illuminate strategy-specific behavior. Examples:
- For arb strategies: spread at entry, spread at close, per-leg PnL split
- For market makers: time quoting, adverse selection rate, fill-to-cancel ratio
- For momentum: signal strength at entry, holding period, win/loss by market regime

Record these under a `"custom_metrics"` key in the iteration's `results` block in `session.json`. This builds a richer history than summary stats alone and lets you track strategy-specific health across iterations.

Use parquet findings and custom metrics to write a specific `analysis` in session.json and to directly motivate the `next_action`. Vague analysis ("strategy underperformed") is not acceptable — point to specific fills, timestamps, or order patterns.

**Threshold check** (from `spec.goals.thresholds`):
- Parse each threshold expression (e.g., `">0"`, `"<0.5"`)
- Evaluate against the corresponding metric from `summary()`
- Record which thresholds passed and failed

**Target check** (from `spec.goals.targets`):
- Compare each metric against its target
- Compare against `best_pnl` / `best_sharpe` from previous iterations
- Note the direction of change (improving, flat, degrading)

**Plateau check:**
- If the `primary_metric` has not improved by >5% relative for 3 consecutive iterations, note a plateau

**Validation** (when all thresholds are met on the primary date range):
- If the spec has additional `date_ranges`, run or submit the strategy on those ranges
- If performance degrades significantly, note potential overfitting

**Sensitivity check** (when all thresholds are met AND `spec.meta.sensitivity_tests` is present AND this is the first iteration this strategy revision has passed thresholds — do not repeat on every iteration):

- **Latency sensitivity** (`sensitivity_tests.latency: true`):
  Re-run the strategy with network latency doubled (e.g. 5ms → 10ms) and halved (e.g. 5ms → 2.5ms) from the spec profile values. Compare `primary_metric` in each.
  - If 2x latency causes any threshold to fail → flag as "latency-sensitive" in analysis. The edge depends on speed; note this as a live-trading risk.
  - If 0.5x latency materially improves the primary metric → note as an opportunity (unrealized alpha at lower latency).

- **Queue model sensitivity** (`sensitivity_tests.queue_model: true`):
  Re-run with `queue_model: risk_averse` if the spec profile is not already risk_averse. (Skip otherwise.)
  - If thresholds fail under risk_averse → flag as "queue-sensitive" in analysis. Fills may be unrealistically optimistic in the baseline config.

Record sensitivity results under a `"sensitivity"` key in the iteration's `results` block:
```json
"sensitivity": {
  "latency_2x":        { "primary_metric": <value>, "thresholds_met": true },
  "latency_0.5x":      { "primary_metric": <value>, "thresholds_met": true },
  "queue_risk_averse": { "primary_metric": <value>, "thresholds_met": true }
}
```
Omit keys for tests that were skipped. These results are informational — they do not block progress — but must be mentioned in `analysis` and the end-of-iteration user summary.

---

## Step 6: Record

Append to `session.json`. After writing session.json, ALWAYS commit the iteration — regardless of whether it was a local run or a sweep:

```bash
git add \
  gnomepy_research/sessions/$ARGUMENTS/strategy.py \
  gnomepy_research/sessions/$ARGUMENTS/__init__.py \
  gnomepy_research/sessions/$ARGUMENTS/spec.yaml \
  gnomepy_research/sessions/$ARGUMENTS/configs/ \
  gnomepy_research/sessions/$ARGUMENTS/session.json
git commit -m "research/$ARGUMENTS: iter NNN"
```

Do NOT stage `results/` — parquet and HTML artifacts are gitignored, but this also keeps the lightweight JSON files out of history. The config YAML + strategy.py is sufficient to reproduce any iteration.

Update `best_iteration`, `best_pnl`, `best_sharpe` if this was the best run. Write `analysis` explaining what happened. Write `next_action` describing what to change next.

```json
{
  "iteration": <N>,
  "timestamp": "<ISO8601>",
  "type": "local" | "sweep",
  "hypothesis": "<stated hypothesis>",
  "changes": ["<what changed from last iteration>"],
  "config": "<configs/iter_NNN.yaml or configs/sweep_NNN.yaml>",
  "results": {
    "date_range": "<start> to <end>",
    "run_id": "<AWS Batch run_id if sweep, else null>",
    "best_params": { "<param>": "<value>" },
    "summary": { "<full summary() dict>" },
    "mm_stats": { "<mm_stats() dict or null>" },
    "thresholds_met": true,
    "threshold_failures": ["<metric> <op> <value> FAILED (<actual>)"],
    "targets_met": { "<metric>": true }
  },
  "analysis": "<What happened and why — be specific about signals, parameters, market behavior>",
  "next_action": "<Exactly what to change next and why>"
}
```

---

## Step 7: Continue or Stop

**Stop and set status to `"completed"`** if:
- All targets in `spec.goals.targets` are met on all date ranges

**Stop and set status to `"stalled"`** if:
- `max_iterations` reached
- Primary metric plateaued (no >5% improvement for 3+ consecutive iterations) with no clear path forward

**Otherwise:** Output a brief summary to the user: what happened this iteration, whether it's better or worse, and what you'll try next. The session continues.

---

## Important Notes

- **DO NOT modify `spec.yaml`** — it is the user's document
- **All Python imports at the top** of `strategy.py` — never inside functions
- **Prices are scaled integers** — divide by 1e9 for display, never pass floats to Intent
- **`__init__.py` must exist** in `gnomepy_research/sessions/$ARGUMENTS/` for the strategy to be importable
- If the backtest crashes with a JVM or JPype error, check the strategy for float prices or bad signal initialization
- Keep `session.json` under 50KB. If it grows large, summarize old iteration `analysis` fields to one sentence each
- Reference existing strategies and signals by their actual file paths when asked by the user
- **Always use `poetry run python3`** for any Python commands — never bare `python` or `python3`
- **Engine bugs**: the backtesting engine is new and may have bugs. If you encounter behavior that looks like an engine bug (not a strategy error), report it clearly and stop the iteration — do not work around it
