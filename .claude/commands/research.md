# Strategy Research Iteration

Run one research iteration for session: **$ARGUMENTS**

All paths below are relative to the gnomepy-research project root (your current working directory).

---

## Step 1: Orient

Read the session spec:
```
gnomepy_research/sessions/$ARGUMENTS/spec.yaml
```

Fetch session state from the API (creates the session if it doesn't exist yet):
```bash
poetry run research sessions get $ARGUMENTS 2>/dev/null \
  || poetry run research sessions create $ARGUMENTS \
       --spec gnomepy_research/sessions/$ARGUMENTS/spec.yaml
```

Then fetch the full session JSON to read iteration history:
```bash
poetry run research sessions get $ARGUMENTS
```

Parse the JSON output. The next iteration number is `iterationCount + 1`. The last 2-3 iteration records (if any) are in `iterations` sorted by iteration number — use them to understand recent history and inform the hypothesis.

**Migration:** If `session.json` exists locally but the API session was just created (iterationCount=0 and session.json has iterations), migrate the local data:
```bash
poetry run python3 - <<'EOF'
import json
from gnomepy_research.api import record_iteration

with open('gnomepy_research/sessions/$ARGUMENTS/session.json') as f:
    data = json.load(f)

for it in data.get('iterations', []):
    record_iteration(
        session_name='$ARGUMENTS',
        iteration=it['iteration'],
        type=it.get('type', 'local'),
        title=it.get('hypothesis', '')[:200],
        description='\n'.join(filter(None, [
            f"## Hypothesis\n{it.get('hypothesis', '')}",
            f"## Analysis\n{it.get('analysis', '')}",
            f"## Next\n{it.get('next_action', '')}",
        ])),
        metrics=it.get('results', {}).get('summary', {}),
        metadata={
            'config_name': it.get('config'),
            'run_id': it.get('results', {}).get('run_id'),
            'best_params': it.get('results', {}).get('best_params', {}),
            'thresholds_met': it.get('results', {}).get('thresholds_met'),
            'threshold_failures': it.get('results', {}).get('threshold_failures', []),
            'changes': it.get('changes', []),
        },
        environment={},
        timestamp=it.get('timestamp'),
    )
print(f"Migrated {len(data.get('iterations', []))} iterations")
EOF
```
After migration, the session.json is no longer used — do not write to it in future steps.

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

**Check for user hints:**
Read `gnomepy_research/sessions/$ARGUMENTS/hints.md` if it exists. If it has content (one or more timestamped hints written by `/research-hint`), treat them as user directives that take priority over the session history's `next_action`. **Do NOT clear this file yet** — it will be cleared in Step 7 after the iteration is successfully recorded.

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

Record these as additional keys in the `metrics` dict when calling `research iterations record` in Step 6. This builds a richer history than summary stats alone and lets you track strategy-specific health across iterations.

**Mandatory diagnostic checks** — MUST run every iteration. Results inform the next hypothesis directly. Compute from fills/orders/intents/market parquet files.

*Arb strategies (`strategy_type: arb`):*

| Check | How to Compute | Symptom → Action |
|-------|---------------|-----------------|
| Fill rate | fills / intents total | <20% → loosen entry threshold |
| Leg imbalance | entries where only 1 leg filled / total entries | >30% → add imbalance timeout, try passive orders |
| Adverse selection | mean price move 1s post-fill (from market.parquet mid) | Consistently negative → entry is stale or too slow |
| Hold time | mean time from entry fill to full unwind | >60s for pure arb → closing logic too conservative |
| Fee drag | total_fees / abs(gross_pnl) | >50% → edge doesn't cover costs; widen threshold or switch venue |
| PnL concentration | % of PnL from top 3 trades | >80% → outlier-dependent, not a consistent edge |

*Market-maker strategies (`strategy_type: mm`):*

| Check | How to Compute | Symptom → Action |
|-------|---------------|-----------------|
| Quote presence | time quoting both sides / total time | <70% → too cautious; widen reentry |
| Edge per fill | mean fill price vs mid at fill time | <0 → adverse selection dominates; add flow signal |
| Inventory duration | mean time at non-zero position | Growing across iterations → risk management broken |
| Fill symmetry | buy_fills / sell_fills | >2:1 or <1:2 → quotes skewed; check fair-value signal |
| Spread vs market | quoted_spread / market_spread | >2.0 → spread too wide to fill; <1.0 → crossing the book |
| Cancel rate | (total_orders - fill_count) / total_orders | >99% → too many phantom quotes |

*All strategy types:*

| Check | How | Action |
|-------|-----|--------|
| Zero fills | fill_count == 0 | Fix entry logic before any parameter tuning |
| Flat PnL | abs(final_pnl) < $0.01 | Not trading or every trade offsets → check position/close logic |

Include each diagnostic result explicitly in the `analysis` field of the iteration record.

Use parquet findings, diagnostics, and custom metrics to write a specific `analysis` (included in the `--description` field) and to directly motivate the next action. Vague analysis ("strategy underperformed") is not acceptable — point to specific fills, timestamps, or order patterns.

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

**Significance check** (when all thresholds are met):
```bash
poetry run research validate significance $ARGUMENTS \
  --fills gnomepy_research/sessions/$ARGUMENTS/results/iter_NNN/fills.parquet \
  --market gnomepy_research/sessions/$ARGUMENTS/results/iter_NNN/market.parquet \
  --n-trials N
```
Note the printed `sharpe_ci_95`, `deflated_sharpe`, and `dsr_significant` values — pass them via `--extra-metrics` in Step 6. If DSR < 0.95 or the CI includes zero, flag this in the analysis — the result may not be statistically significant. **Do not block iteration progress** — significance is informational.

**Validation** (when all thresholds are met on the primary date range):
- If the spec has additional `date_ranges`, run or submit the strategy on those ranges
- If the iteration uses multiple scenarios, compare metrics across scenarios — consistent performance across events is a positive signal; large variance suggests the edge is event-specific
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

## Step 5.5: Accept or Reject

Compare this iteration's `primary_metric` value against the best accepted baseline.

The last accepted iteration number is in `bestIteration` from the session JSON fetched in Step 1. If `bestIteration` is null or this is iteration 1, there is no prior baseline.

**ACCEPT** if:
- This is iteration 1 (always accept — establishes the baseline), OR
- The current `primary_metric` strictly improved vs. the best accepted value (`bestPnl` or `bestSharpe`, whichever is the `primary_metric`)

**REJECT** if:
- The current `primary_metric` is equal to or worse than the best accepted value

**On ACCEPT:**
1. Snapshot all session `.py` files (except `__init__.py`) into `best/`:
   ```bash
   mkdir -p gnomepy_research/sessions/$ARGUMENTS/best
   for f in gnomepy_research/sessions/$ARGUMENTS/*.py; do
     [ "$(basename "$f")" != "__init__.py" ] && cp "$f" gnomepy_research/sessions/$ARGUMENTS/best/
   done
   ```
2. In Step 6: set `"accepted": true` in `--extra-metadata`
3. After Step 6 git commit, update the session with the new best values (substitute actual metric values from `summary.json`):
   ```bash
   poetry run research sessions update $ARGUMENTS \
     --best-iteration N \
     --best-pnl <final_pnl> \
     --best-sharpe <sharpe>
   ```
4. Use commit message: `research/$ARGUMENTS: iter NNN (accepted)`

**On REJECT:**
1. Note `M` = the last accepted iteration number (from `bestIteration`)
2. In Step 6: set `"accepted": false, "returned_to": M` in `--extra-metadata`
3. After Step 6 git commit, restore from `best/` — restoring files that existed at accept time and removing any new files added in the rejected iteration:
   ```bash
   for f in gnomepy_research/sessions/$ARGUMENTS/*.py; do
     base=$(basename "$f")
     [ "$base" = "__init__.py" ] && continue
     if [ -f "gnomepy_research/sessions/$ARGUMENTS/best/$base" ]; then
       cp "gnomepy_research/sessions/$ARGUMENTS/best/$base" "$f"
     else
       rm "$f"
     fi
   done
   ```
4. Use commit message: `research/$ARGUMENTS: iter NNN (rejected, returned to iter M)`
5. The next hypothesis starts from the `best/` snapshot — the rejected strategy is discarded

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

Do NOT stage `results/` or `session.json` — session state is now in the API.

After committing, complete the accept/reject actions from Step 5.5:
- **On ACCEPT:** run the `sessions update` command from Step 5.5 (`--best-iteration`, `--best-pnl`, `--best-sharpe`)
- **On REJECT:** run the `best/` restore script from Step 5.5 to reset all session `.py` files to the accepted baseline

---

## Step 7: Continue or Stop

**Clear hints** (always, after successful record + commit in Step 6):
Write an empty string to `gnomepy_research/sessions/$ARGUMENTS/hints.md` if it exists and had content. Only clear after the commit succeeds — if the iteration failed before reaching this point, hints are preserved for the next retry.

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

Append an entry to `gnomepy_research/research_learnings.md`:
```markdown
### [YYYY-MM-DD] $ARGUMENTS (<strategy_type>, <exchange profiles>)
- STATUS: completed | stalled
- WORKED: <what approaches produced the best accepted iteration>
- FAILED: <what approaches definitively regressed and were rejected>
- INSIGHT: <market-specific constraints, fee structure observations, latency findings>
```

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
