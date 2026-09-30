# Strategy Research Iteration

Run one research iteration for session: **$ARGUMENTS**

All paths below are relative to the gnomepy-research project root (your current working directory).

---

## Step 1: Orient

Read the session spec:
```
gnomepy_research/sessions/$ARGUMENTS/spec.yaml
```

Fetch session state from the API (creates the session if it doesn't exist yet). Pass `--branch` so an
auto-created session is registered against the same branch `/research-new` would have used; for a
branch session also pass `--tags "branch,parent:<parent>"` so `/research-status` can group it:
```bash
poetry run research sessions get $ARGUMENTS 2>/dev/null \
  || poetry run research sessions create $ARGUMENTS \
       --spec gnomepy_research/sessions/$ARGUMENTS/spec.yaml \
       --branch research/$ARGUMENTS
```

Then fetch the full session JSON to read iteration history:
```bash
poetry run research sessions get $ARGUMENTS
```

**The API returns snake_case fields.** Read `iteration_count`, `best_iteration`, `best_pnl`,
`best_sharpe`, `best_metric`, `best_metric_value`, `session_name`, `updated_at` — not their camelCase
spellings.

Parse the JSON output. The next iteration number is `iteration_count + 1`. The last 2-3 iteration
records (if any) are in `iterations` sorted by iteration number — use them to understand recent
history and inform the hypothesis. For a compact view of history at any time:
```bash
poetry run research iterations list $ARGUMENTS --limit 10
```

**Stop before starting if the budget is spent.** If `iteration_count >= spec.constraints.max_iterations`,
do not run another iteration — set the session status to `stalled` (Step 7) and report that the
iteration budget is exhausted. Raising the budget is the user's call, not the loop's.

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

**Read cross-session learnings:**
Read `gnomepy_research/research_learnings.md` if it exists. Scan for entries that match this session's `strategy_type` and listing IDs. Before forming your hypothesis, note:
- Approaches that worked in similar sessions (start here rather than re-discovering)
- Approaches that definitively failed (skip these to avoid re-running dead ends)
- Market-specific insights for the same listing IDs (e.g., latency constraints, fee structures that make certain approaches unviable)

**Check for user directives:**
If the user has given a directive in this conversation — a signal to try, a parameter to change, a structural rethink — it takes priority over the session history's `next_action`. Say which directive you are acting on in the hypothesis so it is visible in the iteration record.

**Interaction mode** (from `spec.meta.interaction_mode`, defaults to `autonomous`):

- **`interactive`**: Use `AskUserQuestion` before forming your hypothesis. Present:
  - A 2-3 sentence summary of the last iteration's results (or "This is iteration 1" if first)
  - Your proposed direction for this iteration
  - Options: "Proceed with this direction" / "I have a different idea" (via Other)
  Incorporate any user input into the hypothesis.

- **`autonomous`**: Skip the question — proceed with the user's latest directive (if any) or the session history's `next_action`.

**Iteration 1:** Design an initial strategy from scratch based on `spec.description` and `spec.constraints`. Look at the existing strategies in `gnomepy_research/strategies/` for patterns. Choose signals from `gnomepy_research/signals/` that match the description. State your strategy design as your hypothesis.

**Later iterations:** Read the last entry's `analysis` and `next_action`. If the user has given a directive, use it instead of `next_action`. Otherwise follow `next_action` unless a better approach is evident from the full history. If the last 3+ iterations show no meaningful improvement in `spec.goals.primary_metric` (see the plateau rule in Step 5), try something fundamentally different — different signals, different strategy class, or a new structural approach.

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
- **All code changes stay under the session directory.** Never modify files outside `gnomepy_research/sessions/$ARGUMENTS/` during research. If you need a signal or component that doesn't exist yet or needs modification, copy it into the session directory and import from there (e.g., `from gnomepy_research.sessions.$ARGUMENTS.signals import MySignal`). Shared code in `gnomepy_research/signals/`, `gnomepy_research/strategies/`, etc. is stable — promote session-local code to shared locations only after the session completes and the approach is proven.

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

When the strategy operates across multiple distinct events (e.g., different prediction market events with separate listing IDs and time windows), use a `scenarios` config instead of the flat format above. Each scenario provides its own `start_date`, `end_date`, `listings`, and optional `strategy_args` overrides (merged on top of the shared `strategy.args`):

```yaml
strategy:
  class_name: "gnomepy_research.sessions.$ARGUMENTS.strategy:YourStrategyClassName"
  args:
    # shared constructor kwargs (sweep params go in sweep: section, not here)

scenarios:
  <event_name_1>:
    start_date: "<start>"
    end_date: "<end>"
    listings:
      - listing_id: <id>
        profile: <profile_name>
    strategy_args:       # optional — merged into strategy.args for this scenario only
      event_ids: [...]
  <event_name_2>:
    start_date: "<start>"
    end_date: "<end>"
    listings: [...]
    strategy_args:
      event_ids: [...]

profiles:
  # same profile definitions as flat format
```

Each scenario produces a separate backtest job with its own report and summary.json. The top-level `start_date`, `end_date`, and `listings` keys are omitted when using `scenarios`.

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

Use the same structure as the local config (Step 4A.2), but add a top-level `sweep:` section for parameters to sweep. Lists in `strategy.args` are always passed as-is — only values declared in `sweep:` are expanded:

```yaml
strategy:
  args:
    gamma: 0.5      # fixed default (overridden per job by sweep)
    delta: 0.5
    outcomes:       # list — always fixed, never swept
      - {pm: 222852, k: 97203}

sweep:
  strategy:
    gamma: [0.5, 1.0, 2.0, 4.0]             # list sweep
    delta: {min: 0.5, max: 3.0, step: 0.5}  # range sweep
  profiles:
    default:
      network_latency:
        latency_nanos: [5000000, 10000000]   # profile sweep
```

**Hard cap: the cartesian product of scenarios × sweep parameters must not exceed 100 jobs.** Before writing the config, calculate the total job count (`len(scenarios) × product(sweep_lengths)`). If it exceeds 100, reduce the parameter grid or the number of scenarios.

Scenarios compose with sweeps — a config can have both:
```yaml
strategy:
  args:
    min_pure_arb_bps: 5   # fixed default

sweep:
  strategy:
    min_pure_arb_bps: [5, 10, 15]   # sweep — 3 values

scenarios:
  baseball: { ... }
  football: { ... }
# → 2 scenarios × 3 values = 6 jobs total
```

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

This downloads all output artifacts (parquet files, summary.json, etc.) for every sweep job into subdirectories under `iter_NNN/`. Find the job with the best value for `spec.goals.primary_metric` by reading each job's `summary.json` — highest when `spec.goals.direction` is `maximize`, lowest when it is `minimize`. Record that job's parameters as the winning sweep configuration.

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

Record these as additional keys in the `metrics` dict when calling `research iterations record` in Step 6. This builds a richer history than summary stats alone and lets you track strategy-specific health across iterations.

**Mandatory diagnostic checks** — MUST run every iteration. Results inform the next hypothesis
directly.

**Use the library — do not hand-roll these from parquet.** Most of the table below is already
implemented and tested:

```bash
poetry run python3 - <<'EOF'
import json
from gnomepy_research.explore import load_results
from gnomepy_research.reporting.backtest.adverse_selection import compute_adverse_selection
from gnomepy_research.reporting.backtest.market_making import compute_mm_stats
from gnomepy_research.reporting.backtest.rolling_performance import (
    compute_alpha_decay, compute_rolling_sharpe, detect_regimes, pnl_by_regime,
)

report = load_results("gnomepy_research/sessions/$ARGUMENTS/results/iter_NNN")

# Market makers: time_quoting_both_sides, avg_net_edge_captured_bps, buy_sell_fill_ratio,
# quoted_vs_market_spread, quote_to_fill_ratio, mean_abs_position, max_abs_position
print(json.dumps(compute_mm_stats(report), indent=2, default=str))

# All strategy types
adverse = compute_adverse_selection(report.fills_df(), report.market_records_df())
print(adverse)
print(compute_rolling_sharpe(report.pnl_curve).describe())
print(compute_alpha_decay(report.pnl_curve).describe())
print(pnl_by_regime(report.pnl_curve, detect_regimes(report.market_records_df())))
EOF
```

Report each number below explicitly in the `analysis` field, with the action you are taking from it.

*Arb strategies (`strategy_type: arb`):*

| Check | Where it comes from | Symptom → Action |
|-------|--------------------|-----------------|
| Fill rate | `fill_count / total_intents` (`compute_mm_stats`) | <20% → loosen entry threshold |
| Leg imbalance | entries where only 1 leg filled / total entries (from `fills.parquet`) | >30% → add imbalance timeout, try passive orders |
| Adverse selection | `compute_adverse_selection(...)` | Consistently negative → entry is stale or too slow |
| Hold time | mean time from entry fill to full unwind (from `fills.parquet`) | >60s for pure arb → closing logic too conservative |
| Fee drag | `total_fees / abs(gross_pnl)` (`summary.json`) | >50% → edge doesn't cover costs; widen threshold or switch venue |
| PnL concentration | % of PnL from top 3 trades (from `fills.parquet`) | >80% → outlier-dependent, not a consistent edge |

*Market-maker strategies (`strategy_type: mm`):* every row is a key of `compute_mm_stats(report)`.

| Check | Key | Symptom → Action |
|-------|-----|-----------------|
| Quote presence | `time_quoting_both_sides` | <0.70 → too cautious; widen reentry |
| Edge per fill | `avg_net_edge_captured_bps` | <0 → adverse selection dominates; add flow signal |
| Inventory | `mean_abs_position`, `max_abs_position` | Growing across iterations → risk management broken |
| Fill symmetry | `buy_sell_fill_ratio` | >2.0 or <0.5 → quotes skewed; check fair-value signal |
| Spread vs market | `quoted_vs_market_spread` | >2.0 → too wide to fill; <1.0 → crossing the book |
| Quote-to-fill | `quote_to_fill_ratio` | >100 → too many phantom quotes |

*All strategy types:*

| Check | How | Action |
|-------|-----|--------|
| Zero fills | `fill_count == 0` | Fix entry logic before any parameter tuning |
| Flat PnL | `abs(final_pnl) < $0.01` | Not trading or every trade offsets → check position/close logic |
| Alpha decay | `compute_alpha_decay(report.pnl_curve)` | Declining → edge degrades within the session |
| Regime dependence | `pnl_by_regime(...)` | PnL from one regime only → not robust |

Include each diagnostic result explicitly in the `analysis` field of the iteration record.

Use parquet findings, diagnostics, and custom metrics to write a specific `analysis` (included in the
`--description` field) and to directly motivate the next action. Vague analysis ("strategy
underperformed") is not acceptable — point to specific fills, timestamps, or order patterns.

**Threshold check** (from `spec.goals.thresholds`):
- Parse each threshold expression (e.g., `">0"`, `"<0.5"`)
- Evaluate against the corresponding metric from `summary()`
- Record which thresholds passed and failed

**Target check** (from `spec.goals.targets`):
- Compare each metric against its target
- Compare against `best_pnl` / `best_sharpe` from previous iterations
- Note the direction of change (improving, flat, degrading)

**Plateau check:**

A purely relative test breaks down for `final_pnl` and `sharpe`, which are routinely negative or near
zero — 5% of −0.01 is meaningless and 5% of 0 is undefined. Use both tests, in the spec's
`direction`:

- Let `B` be the best accepted value so far and `X` this iteration's `primary_metric`.
- The iteration is an improvement if it moves in the spec's direction by **more than 5% of `|B|`**
  *and* by more than an absolute epsilon — `0.01` for `final_pnl` (one cent), `0.0005` for per-bar
  `sharpe`/`sortino` (about 1.0 annualized), otherwise 1% of `|B|`.
- If `B` is zero or unset, any move in the spec's direction beyond the epsilon counts.
- Note a plateau when 3 consecutive iterations fail that test.

**Significance check** (when all thresholds are met). Use `--json` so the values can be read
directly rather than scraped from prose:
```bash
poetry run research validate significance $ARGUMENTS \
  --fills gnomepy_research/sessions/$ARGUMENTS/results/iter_NNN/fills.parquet \
  --market gnomepy_research/sessions/$ARGUMENTS/results/iter_NNN/market.parquet \
  --n-trials N --json
```

**`--n-trials` is the total number of strategy variants evaluated in this session so far**, not the
iteration number: cumulative iterations to date, plus every additional job from each sweep (a sweep
of 24 jobs counts as 24, not 1). DSR is highly sensitive to this — understating it inflates the
result.

Pass `sharpe_ci_95`, `deflated_sharpe` and `dsr_significant` from the JSON into `--extra-metrics` in
Step 6. If `dsr_significant` is false or the CI includes zero, flag it in the analysis. **Do not
block iteration progress** — significance is informational, and on a session-length window it will
essentially never pass (see `tutorials/02_research_workflow.md`).

**Validation** (when all thresholds are met on the primary date range):
- If the spec has additional `date_ranges`, run or submit the strategy on those ranges
- If the iteration uses multiple scenarios, compare metrics across scenarios — consistent performance across events is a positive signal; large variance suggests the edge is event-specific
- If performance degrades significantly, note potential overfitting

**Sensitivity check** (when all thresholds are met AND `spec.meta.sensitivity_tests` is present AND
this is the first iteration this strategy revision has passed thresholds — do not repeat on every
iteration).

**Generate the sweep configs, don't hand-edit profiles:**

```bash
poetry run python3 - <<'EOF'
from gnomepy_research.validation.monte_carlo import (
    generate_latency_sweep_config, generate_queue_sweep_config,
)

base = "gnomepy_research/sessions/$ARGUMENTS/configs/iter_NNN.yaml"

# sensitivity_tests.latency: sweeps network latency around the spec value
generate_latency_sweep_config(base, "gnomepy_research/sessions/$ARGUMENTS/configs/sens_latency.yaml")

# sensitivity_tests.queue_model: sweeps queue-model pessimism
generate_queue_sweep_config(base, "gnomepy_research/sessions/$ARGUMENTS/configs/sens_queue.yaml")
EOF
```

Run each generated config the same way as any other local run, with its own `--output` directory, then compare `primary_metric` across the sweep points.

- **Latency sensitivity** (`sensitivity_tests.latency: true`):
  - If higher latency causes any threshold to fail → flag as "latency-sensitive" in analysis. The edge depends on speed; note this as a live-trading risk.
  - If lower latency materially improves the primary metric → note as an opportunity (unrealized alpha at lower latency).

- **Queue model sensitivity** (`sensitivity_tests.queue_model: true`):
  - If thresholds fail under the more pessimistic queue assumptions → flag as "queue-sensitive" in analysis. Fills may be unrealistically optimistic in the baseline config.

Optionally, bootstrap the PnL path to see how much of the result is path luck:
```python
from gnomepy_research.validation.monte_carlo import bootstrap_pnl_paths, summarize_mc_paths
summarize_mc_paths(bootstrap_pnl_paths(report.fills_df(), report.market_records_df()))
```

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

## Step 5.5: Accept or Reject

Compare this iteration's `primary_metric` value against the best accepted baseline.

**Find the baseline.** The last accepted iteration number is `best_iteration` in the session JSON
from Step 1. The baseline *value* depends on which metric the spec optimizes:

- `final_pnl` → `best_pnl`
- `sharpe` → `best_sharpe`
- anything else (`sortino`, `avg_net_edge_captured_bps`, ...) → `best_metric_value`, valid only when
  `best_metric` equals `spec.goals.primary_metric`

If none of those is populated, fall back to the `primary_metric` value of iteration `best_iteration`
in the `iterations` array. If `best_iteration` is null or this is iteration 1, there is no baseline.

**Respect `spec.goals.direction`.** It is `maximize` or `minimize`. "Improved" means strictly greater
for `maximize` and strictly less for `minimize` — never assume higher is better.

**ACCEPT** if:
- This is iteration 1 (always accept — establishes the baseline), OR
- The current `primary_metric` strictly improved on the baseline in the spec's direction

**REJECT** if:
- The current `primary_metric` is equal to or worse than the baseline in the spec's direction

**On ACCEPT:**
1. Snapshot all session `.py` files (except `__init__.py`) into `best/`:
   ```bash
   poetry run python3 - <<'EOF'
   import shutil
   from pathlib import Path

   session = Path("gnomepy_research/sessions/$ARGUMENTS")
   best = session / "best"
   best.mkdir(parents=True, exist_ok=True)
   for f in session.glob("*.py"):
       if f.name != "__init__.py":
           shutil.copy2(f, best / f.name)
           print("snapshot", f.name)
   EOF
   ```
2. In Step 6: set `"accepted": true` in `--extra-metadata`
3. Use commit message: `research/$ARGUMENTS: iter NNN (accepted)`
4. After the Step 6 commit, record the new best (substitute actual values from `summary.json`).
   Always send `--best-iteration`; send `--best-pnl` / `--best-sharpe` when available, and
   `--best-metric` / `--best-metric-value` whenever the primary metric is neither PnL nor Sharpe:
   ```bash
   poetry run research sessions update $ARGUMENTS \
     --best-iteration N \
     --best-pnl <final_pnl> \
     --best-sharpe <sharpe> \
     --best-metric <primary_metric> --best-metric-value <value>
   ```

**On REJECT:**
1. Note `M` = the last accepted iteration number (from `best_iteration`)
2. **Restore from `best/` BEFORE the Step 6 commit**, so the commit records the reverted state and
   the working tree is left clean. Restore the files that existed at accept time and remove any file
   the rejected iteration added:
   ```bash
   poetry run python3 - <<'EOF'
   import shutil
   from pathlib import Path

   session = Path("gnomepy_research/sessions/$ARGUMENTS")
   best = session / "best"
   for f in session.glob("*.py"):
       if f.name == "__init__.py":
           continue
       snapshot = best / f.name
       if snapshot.exists():
           shutil.copy2(snapshot, f)
           print("restored", f.name)
       else:
           f.unlink()
           print("removed", f.name, "(added by the rejected iteration)")
   EOF
   ```
   The rejected strategy still exists in the iteration record and in `results/iter_NNN` — only the
   working copy is reverted.
3. In Step 6: set `"accepted": false, "returned_to": M` in `--extra-metadata`
4. Use commit message: `research/$ARGUMENTS: iter NNN (rejected, returned to iter M)`
5. Do **not** call `sessions update` — the recorded best is unchanged
6. The next hypothesis starts from the `best/` snapshot

---

## Step 6: Record

Record the iteration. Metrics and environment are read automatically from the results directory — only provide the human-authored fields and any extras from Step 5:

```bash
poetry run research iterations record-from-results $ARGUMENTS \
  --iteration N \
  --type local \
  --title "<one-line hypothesis summary>" \
  --description "## Hypothesis
<stated hypothesis>

## Changes
- <what changed from last iteration>

## Analysis
<what happened and why — specific fills, signals, market behavior>

## Next
<exactly what to change next and why>" \
  --results-dir gnomepy_research/sessions/$ARGUMENTS/results/iter_NNN \
  --extra-metrics '{"sharpe_ci_95": [<lo>, <hi>], "deflated_sharpe": <dsr>}' \
  --extra-metadata '{"config_name": "gnomepy_research/sessions/$ARGUMENTS/configs/iter_NNN.yaml", "run_id": null, "best_params": {}, "thresholds_met": true, "threshold_failures": [], "changes": ["<change 1>"], "accepted": true}'
```

Include `"accepted": true` or `"accepted": false` (from Step 5.5). On reject, also include `"returned_to": M`. Omit `--extra-metrics` if the significance check was skipped (thresholds not met).

ALWAYS commit the iteration after recording — regardless of whether it was a local run or a sweep:

```bash
git add \
  gnomepy_research/sessions/$ARGUMENTS/*.py \
  gnomepy_research/sessions/$ARGUMENTS/best/ \
  gnomepy_research/sessions/$ARGUMENTS/__init__.py \
  gnomepy_research/sessions/$ARGUMENTS/spec.yaml \
  gnomepy_research/sessions/$ARGUMENTS/configs/
git commit -m "research/$ARGUMENTS: iter NNN (accepted)"
# or on reject: git commit -m "research/$ARGUMENTS: iter NNN (rejected, returned to iter M)"
```

Do NOT stage `results/` — session state is in the API and backtest output does not belong in git.

On REJECT the `best/` restore has already run (Step 5.5), so this commit records the reverted state
and leaves the working tree clean.

After committing:
- **On ACCEPT:** run the `sessions update` command from Step 5.5
- **On REJECT:** nothing further — the recorded best is unchanged

---

## Step 7: Continue or Stop

**Stop and set status to `"completed"`** if:
- All targets in `spec.goals.targets` are met on all date ranges

**Stop and set status to `"stalled"`** if:
- `max_iterations` reached
- Primary metric plateaued (no >5% improvement for 3+ consecutive iterations) with no clear path forward

When stopping, update the session status:
```bash
poetry run research sessions update $ARGUMENTS --status completed
# or: --status stalled
```

**Write cross-session learnings** (only when stopping as `completed` or `stalled`, OR when a new best is accepted and all thresholds are met for the first time):

**One entry per session — UPDATE the existing entry, do not append a second one.** Search
`gnomepy_research/research_learnings.md` for `### [<date>] $ARGUMENTS` first. If it exists, rewrite
that entry in place with the current verdict, folding in anything still true from the old text and
noting what it supersedes. Append only when the session has no entry. Blind appending is what left
three sessions with contradictory verdicts (`informed_pmm` recorded as both completed and stalled).

Use exactly this shape, and place it below the `<!-- Entries updated in place here -->` marker:
```markdown
### [YYYY-MM-DD] $ARGUMENTS — <COMPLETED|STALLED> (best=iter_NNN)
**Type**: <strategy_type> | **Venues**: <profiles> | **Listings**: <ids> | **Window**: <date range>
**Best**: iter_NNN — PnL $X, N fills, per-bar Sharpe Y

**Worked:**
- <what produced the best accepted iteration>

**Failed:**
- <what definitively regressed and was rejected>

**Insights:**
- <market constraints, fee structure, latency findings, structural ceilings>
```

Use the `strategy_type` value from `spec.yaml` verbatim so entries stay searchable. Always label
Sharpe as **per-bar** — it is not annualized, and mislabelling it is what made every pre-2026-09-30
session conclude its target was unreachable.

Then push the same learning as a session note (for API/web UI visibility):
```bash
poetry run research notes add $ARGUMENTS "LEARNING: <one-paragraph summary of WORKED/FAILED/INSIGHT>"
```

Then commit the learnings file:
```bash
git add gnomepy_research/research_learnings.md
git commit -m "research/$ARGUMENTS: learnings (iter NNN)"
```

**Otherwise:** Output a brief summary to the user: what happened this iteration, whether it's better or worse, and what you'll try next. The session continues (status remains `"running"`).

---

## Important Notes

- **DO NOT modify `spec.yaml`** — it is the user's document
- **All Python imports at the top** of `strategy.py` — never inside functions
- **Prices are scaled integers** — divide by 1e9 for display, never pass floats to Intent
- **`__init__.py` must exist** in `gnomepy_research/sessions/$ARGUMENTS/` for the strategy to be importable
- If the backtest crashes with a JVM or JPype error, check the strategy for float prices or bad signal initialization
- Reference existing strategies and signals by their actual file paths when asked by the user
- **Always use `poetry run python3`** for any Python commands — never bare `python` or `python3`
- **Engine bugs**: the backtesting engine is new and may have bugs. If you encounter behavior that looks like an engine bug (not a strategy error), report it clearly and stop the iteration — do not work around it
