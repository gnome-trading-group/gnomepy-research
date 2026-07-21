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

The wizard walks you through 9 rounds of questions:

| Round | What it collects |
|-------|-----------------|
| 1 | Strategy type (`market_maker`, `momentum`, `arb`, `custom`), description, primary metric |
| 2a–2b | Exchange profile: fees (taker/maker), queue model, latency |
| 3 | Listing ID(s), base preset, loop back for additional profiles |
| 4 | Dev date range (keep to 30 min–2 hr), max iterations, signals to explore |
| 5 | Hard thresholds (Sharpe floor, fill floor, PnL floor) |
| 6 | Aspirational targets (Sharpe, Sortino, % positive time buckets) |
| 7 | Market-making targets (time quoting, edge per fill) — MM only |
| 8 | Design constraints, execution mode, sensitivity tests |
| 9 | Interaction mode (autonomous/interactive), final confirmation |

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
    - start: "2026-05-12T23:28:00"    # dev window — keep short
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
    fill_count: ">10"
    final_pnl: ">0"
  targets:                         # aspirational — iteration continues until all met
    sharpe: ">1.0"
    sortino: ">1.5"
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

**Important: never modify `spec.yaml` after the session starts.** It is the source of truth for what the session is trying to achieve. Use `/research-hint` to influence the next iteration without touching the spec.

### Available presets

If you chose a base preset in Round 3, the first iteration starts from that preset's backtest config rather than building from scratch:

- `mm_btc_30m` — market maker on BTC, 30-minute session, 5ms latency, risk_averse queue
- `arb_btc_30m` — cross-exchange arb on BTC, two profiles, 20ms/200ms latency
- `momentum_btc_30m` — momentum taker on BTC, 30-minute session

---

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

**Step 2 — Hypothesize:** Reads `hints.md` if it exists (your directives). On iteration 1, designs a strategy from scratch based on `spec.description` and `spec.constraints`. On later iterations, reads the previous iteration's analysis and `next_action` to decide what to change.

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

**What good looks like:**
- Sharpe > 1.0 on the dev range is a reasonable starting target
- Fill count > 25 — fewer fills means the Sharpe estimate is noisy
- PnL curve that climbs consistently, not driven by 1-2 large trades

**Red flags to watch for:**
- Very few fills (< 10): the strategy isn't trading — check quoting logic or entry conditions
- Sharpe > 3.0 with < 50 fills on a 30-minute window: likely a lucky fluke, not signal
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
- Use `/research-hint <name>` to inject a direction for the next iteration without stopping the loop
- Interrupt the loop if you see a structural error (wrong listing IDs, broken position logic) — fix it manually, then resume
- If the loop gets stuck after several identical iterations, check `hints.md` to make sure the hint was consumed, or restart with a stronger hint about what to change

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
