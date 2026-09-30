# Getting Started with Research Sessions

This tutorial takes you from zero to your first backtest result.

---

## Prerequisites

**Environment setup:**
```bash
cd gnomepy-research
poetry install
```

**AWS credentials** must be configured — the backtest engine reads market data from S3 (`gnome-market-data-dev`). Run `aws configure` or set `AWS_PROFILE` if you haven't already.

**JVM** starts automatically when you run a backtest. If you see a JPype error on first run, check that `JAVA_HOME` points to a JDK 17+ installation.

**Web UI** — the Research page shows session state, iteration history, and notes. Session state is stored in the API, not in local files.

---

## Starting a New Session

Run:
```
/research-new <session_name>
```

The session name must be a valid Python identifier: lowercase letters, digits, and underscores only (`n_exchange_arb`, `mm_btc_v2`). No hyphens.

The wizard walks you through 5 rounds of questions:

| Round | What it collects |
|-------|-----------------|
| 1 | Strategy type (`market_maker`, `momentum`, `arb`, `custom`), description, primary metric, direction |
| 2 | Exchange profile(s) from `gnomepy_research/profiles/`, listing IDs per profile, loop for more profiles |
| 3 | Dev date range (keep to 30 min–2 hr), max iterations |
| 4 | Goals — thresholds and targets (accept the strategy-type defaults, or customize), signals to explore |
| 5 | Design constraints, execution mode, interaction mode, final confirmation |

Round 2 can create a new exchange profile (fees, latency, queue model) and saves it to
`gnomepy_research/profiles/<name>.yaml` for reuse. Round 4's defaults are in **per-bar** Sharpe units
— see "Reading Results" below.

After confirmation, the wizard:
1. Creates `gnomepy_research/sessions/<name>/` with `configs/` and `results/` subdirectories
2. Writes `spec.yaml` with all your answers
3. Registers the session in the API (visible in web UI immediately)

It does **not** create `strategy.py` or run any backtests — that happens when you start iterating.

---

## Understanding spec.yaml

`spec.yaml` is your research contract. It defines goals and constraints that every iteration must respect.

```yaml
name: n_exchange_arb
description: >
  Cross-exchange perpetual futures arb across 3 listings:
  hyperliquid listings 1 and 28, lighter listing 36.

data:
  listings:
    - listing_id: 1
      profile: hyperliquid
    - listing_id: 36
      profile: lighter
  date_ranges:
    - start: "2026-05-13T18:00:00"    # dev window — keep to 30min–2hr
      end:   "2026-05-13T19:50:00"
    # Additional ranges for OOS validation (not used during iteration):
    # - start: "2026-06-01T00:00:00"
    #   end:   "2026-06-07T00:00:00"

profiles:
  hyperliquid:
    fee_model:
      type: static
      taker_fee: 0.0004      # 4 bps
      maker_fee: 0.00012     # 1.2 bps
    network_latency:
      type: static
      latency_nanos: 20000000      # 20ms
    order_processing_latency:
      type: static
      latency_nanos: 5000000       # 5ms
    queue_model: risk_averse       # conservative fill estimates

goals:
  primary_metric: final_pnl        # what Claude optimizes
  direction: maximize
  thresholds:                      # hard constraints — all must pass
    sharpe: ">0"
    fill_count: ">20"
    final_pnl: ">0"
  targets:                         # aspirational — iteration continues until all met
    sharpe: ">0.002"               # PER-BAR at 10s bars (≈ 3.6 annualized), not annualized
    sortino: ">0.003"
    pct_positive_buckets: ">0.6"

constraints:
  strategy_type: arb
  max_iterations: 20
  avoid:
    - "max_position must stay under 20"

meta:
  execution_mode: auto             # local for logic changes, batch for sweeps
  sensitivity_tests:
    latency: true
    queue_model: true
  interaction_mode: autonomous
```

**Treat `spec.yaml` as the session's contract.** It is the source of truth for what the session is trying to achieve, and the loop must not move its own goalposts. To steer an iteration without changing the contract, tell Claude directly in the session running the loop.

## Session Structure

After the wizard runs and after the first iteration starts:

```
gnomepy_research/sessions/<name>/
  spec.yaml         # DO NOT MODIFY — research goals and constraints
  strategy.py       # modified in-place each iteration
  __init__.py       # empty — makes the session importable as a Python module
  configs/
    iter_001.yaml   # backtest config written for iteration 1
    iter_002.yaml   # ...
    sweep_003.yaml  # sweep config (if remote sweep was used)
  results/
    iter_001/       # backtest outputs: summary.json, report.html, *.parquet
    iter_002/
    walk_forward/   # created by /research-validate
  notes/            # local copies of notes synced from API (Obsidian-compatible)
```

The session lives on its own git branch (`research/<name>`), created automatically on the first iteration. This keeps sessions isolated from each other and from `main`.

---

## Your First Iteration

Run:
```
/research <session_name>
```

What happens internally across the 7 steps:

**Step 1 — Orient:** Reads `spec.yaml`, fetches session state from the API (iteration count, history of last 2-3 iterations, best metrics so far). Creates the git branch `research/<name>` on iteration 1.

**Step 2 — Hypothesize:** Picks up any directive you have given in the conversation. On iteration 1, designs a strategy from scratch based on `spec.description` and `spec.constraints`. On later iterations, reads the previous iteration's analysis and `next_action` to decide what to change.

**Step 3 — Decide:** Chooses local run (logic change) or remote sweep (parameter search) based on `spec.meta.execution_mode`. On iteration 1, always local.

**Step 4A — Write & run (local):** Writes `strategy.py`, writes `configs/iter_001.yaml`, runs:
```bash
poetry run gnomepy backtest run \
  --config gnomepy_research/sessions/<name>/configs/iter_001.yaml \
  --output gnomepy_research/sessions/<name>/results/iter_001
```

**Step 5 — Evaluate:** Reads `results/iter_001/summary.json` for metrics, loads parquet files to understand why (fill timing, order patterns, market context). Checks thresholds and targets. Computes Deflated Sharpe Ratio if thresholds are met.

**Step 6 — Record:** Calls `poetry run research iterations record-from-results` to store the iteration in the API (title, hypothesis, analysis, metrics, changes, next action). Commits `strategy.py` and `configs/` to the session branch.

**Step 7 — Continue or stop:** Outputs a brief summary. Continues if targets not yet met and iterations remain.

---

## Reading Results

After an iteration completes, results land in `results/iter_NNN/`:

- **`summary.json`** — high-level metrics: `final_pnl`, `sharpe`, `sortino`, `fill_count`, `total_fees`, `pct_positive_buckets`, etc.
- **`report.html`** — rendered HTML report with PnL chart, fill distribution, position curve
- **`fills.parquet`**, **`orders.parquet`**, **`intents.parquet`**, **`market.parquet`** — raw event data for deep-dive analysis

Quick check for a result:
```bash
cat gnomepy_research/sessions/<name>/results/iter_001/summary.json | python3 -m json.tool
```

Or open `report.html` in a browser for the visual summary.

**`sharpe` here is a per-bar ratio at 10s bars, not annualized** — multiply by roughly 1776 for the
annualized figure. See `tutorials/02_research_workflow.md` for the full explanation; getting this
wrong is what made every session before 2026-09-30 stall against unreachable targets.

**What good looks like:**
- Per-bar Sharpe of 0.002–0.003 on the dev range (≈ 3.6–5.3 annualized) is a strong result
- Fill count > 20 — fewer fills means the Sharpe estimate is noise
- PnL curve that climbs consistently, not driven by 1-2 large trades

**Red flags to watch for:**
- Very few fills (< 20): the strategy isn't trading — check quoting logic or entry conditions
- Per-bar Sharpe > 0.01 (≈ 18 annualized) with < 50 fills on a 30-minute window: a lucky fluke, not signal
- PnL entirely in one 10-minute window: alpha may be concentrated on a single event

Use `notebooks/02_backtest_deep_dive.ipynb` for a structured post-result analysis.

---

## Continuous Iteration

To run iterations hands-free until targets are met or `max_iterations` is reached:
```
/loop /research <session_name>
```

The loop runs `/research` repeatedly, committing each iteration. It stops when:
- All `goals.targets` are met → status set to `completed`
- `max_iterations` reached → status set to `stalled`
- Primary metric plateaus for 3+ consecutive iterations with no clear path forward → status set to `stalled`

**When to intervene:**
- Type a directive into the session running the loop — it is picked up at the next iteration's hypothesis step
- Interrupt the loop if you see a structural error (wrong listing IDs, broken position logic) — fix it manually, then resume
- If the loop gets stuck after several identical iterations, give it a more structural directive — a different signal or strategy class, not another parameter tweak

---

## CLI Reference

The `research` CLI manages sessions and data outside the iteration loop:

```bash
# List all sessions and their status
poetry run research sessions list

# Get session details (iteration count, best metrics, status)
poetry run research sessions get <name>

# Update session status manually
poetry run research sessions update <name> --status completed

# List iterations for a session
poetry run research iterations list <name>

# Add a note to a session
poetry run research notes add <name> "Observed that spread widens at session open"

# Sync notes to local files (for Obsidian)
poetry run research notes pull <name>
poetry run research notes push <name>

# Run walk-forward validation
poetry run research validate walk-forward <name> \
  --config gnomepy_research/sessions/<name>/configs/iter_015.yaml \
  --start 2026-06-01T00:00:00 \
  --end   2026-06-30T00:00:00 \
  --folds 5
```

---

## Next Steps

- Read **`02_research_workflow.md`** for the full evaluation → iterate → validate lifecycle
- Read **`03_strategy_building.md`** for signal DSL, strategy patterns, and code conventions
- Open `notebooks/01_signal_exploration.ipynb` to explore market data and signals before building a strategy
