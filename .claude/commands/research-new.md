# Create New Research Session

Scaffold a new strategy research session. The session name is: **$ARGUMENTS**

## Steps

### 1. Resolve session name
If `$ARGUMENTS` is empty, use `AskUserQuestion` to ask the user for a session name before continuing.

The session name must be a valid Python module identifier: lowercase letters, digits, and underscores only — no hyphens, spaces, or other characters. It must not start with a digit. If the provided name is invalid, reject it and ask for a corrected name before proceeding. Example: `my_strategy` is valid; `my-strategy` and `my strategy` are not.

---

### 2. Round 1 — Core identity
Use `AskUserQuestion` with exactly these 4 questions in a single call:

1. **header: "Strategy type"** (single-select)
   - `market_maker` — Passive quoting on both sides
   - `momentum` — Directional taker strategy
   - `arb` — Cross-exchange or statistical arbitrage
   - `custom` — Other / hybrid

2. **header: "Description"** — Ask the user to describe their strategy idea: signals, logic, and market conditions. Use options:
   - "I'll describe it below (use Other)" — prompts them to type in Other
   - Provide 2-3 short example descriptions as non-selectable placeholders so the "Other" free-text path is obvious

3. **header: "Primary metric"** (single-select)
   - `final_pnl` — Total PnL at end of backtest
   - `sharpe` — Sharpe ratio
   - `sortino` — Sortino ratio
   - `avg_net_edge_captured_bps` — Avg edge per fill in bps (use Other for a different metric name)

4. **header: "Direction"** (single-select)
   - `maximize` — Higher is better
   - `minimize` — Lower is better

---

### 3. Round 2a — Exchange profile: identity + fees
Each profile defines the simulation parameters for one exchange/listing. Most strategies need one profile (`default`); arb strategies need two (e.g. `exchange_a`, `exchange_b`).

Use `AskUserQuestion` with exactly these 4 questions in a single call:

1. **header: "Profile name"** (single-select)
   - `default` — Single-exchange strategy
   - `exchange_a` — First leg of arb
   - `exchange_b` — Second leg of arb (use Other for custom name)

2. **header: "Taker fee"** (single-select) — Fee paid when taking liquidity. Use Other to type an exact decimal value.
   - `0.0005 (5 bps)` — Crypto standard
   - `0.0004 (4 bps)` — Common mid-tier (e.g. Hyperliquid)
   - `0.0002 (2 bps)` — Low-fee tier
   - `0 (zero)` — No taker fee

3. **header: "Maker fee"** (single-select) — Fee (or rebate if negative) when providing liquidity. Use Other to type an exact decimal value.
   - `-0.0001 (-1 bps rebate)` — Crypto standard rebate
   - `0.00012 (1.2 bps fee)` — Small maker fee (e.g. Hyperliquid)
   - `0 (zero)` — No maker fee

4. **header: "Queue model"** (single-select)
   - `risk_averse` *(Recommended)* — Assumes you're last in queue; conservative fill estimates
   - `optimistic` — Assumes you're first in queue; aggressive fill estimates
   - `probabilistic` — Fill probability based on queue position

### 3b. Round 2b — Exchange profile: latency
Use `AskUserQuestion` with exactly these 2 questions in a single call:

1. **header: "Network latency"** (single-select) — Round-trip market data + order submission delay. Use Other to type a value in ms (e.g. `200ms`) or nanos (e.g. `200000000ns`).
   - `1ms (1,000,000 ns)` — Co-located / ultra-low latency
   - `5ms (5,000,000 ns) (Recommended)` — Typical cloud-hosted
   - `20ms (20,000,000 ns)` — Remote / consumer connection
   - `200ms (200,000,000 ns)` — High-latency exchange

2. **header: "Processing latency"** (single-select) — Exchange order processing delay. Use Other to type a value in μs/ms (e.g. `500us`) or nanos.
   - `500μs (500,000 ns)` — Fast exchange
   - `1ms (1,000,000 ns) (Recommended)` — Standard
   - `5ms (5,000,000 ns)` — Slow exchange

---

### 4. Round 3 — Listing ID + loop decision
Use `AskUserQuestion` with exactly these 3 questions in a single call:

1. **header: "Listing ID"** — Which listing_id maps to this profile?
   - `1`
   - `2`
   - `3` (use Other for a different ID)

2. **header: "Add profile?"** (single-select) — Do you need another exchange profile?
   - `Yes` — I have another exchange to configure
   - `No` — Done with profiles

3. **header: "Base preset"** — Start from an existing preset config, or build from scratch?
   - `none` — Build from scratch
   - `mm_btc_30m` — Market maker BTC 30-min preset
   - `momentum_btc_30m` — Momentum BTC 30-min preset
   - `arb_btc_30m` — Cross-exchange arb BTC 30-min preset (use Other for custom preset name)

**If the user answered "Yes" to "Add profile?", loop back and repeat Rounds 2a–2b and Round 3 for the next profile. Continue until they answer "No".**

---

### 5. Round 4 — Date range & iterations
Use `AskUserQuestion` with exactly these 4 questions in a single call:

1. **header: "Range start"** — Start of dev backtest window (keep total window to 30min–2hr). Use Other for custom datetime.
   - `2026-01-23T10:30:00`
   - `2026-02-15T14:00:00`
   - `2026-03-10T09:30:00`

2. **header: "Range end"** — End of dev backtest window. Use Other for custom datetime.
   - `2026-01-23T13:00:00`
   - `2026-02-15T16:30:00`
   - `2026-03-10T12:00:00`

3. **header: "Max iters"** (single-select) — Maximum iterations before the research loop stops.
   - `10`
   - `20` *(Recommended)*
   - `30`
   - `50`

4. **header: "Signals"** — Signals for Claude to explore during research (comma-separated hints, or leave blank). For market_maker suggest `MicropriceFairValue, EWMA(SpreadVolatility())`. For momentum suggest `EWMA(Returns()), TradeImbalance`. For arb suggest `cross-exchange spread, basis`. Use Other for custom list or to skip.
   - "Use defaults for this strategy type"
   - "None / let Claude decide" (use Other to type custom signals)

---

### 6. Round 5 — Thresholds
Hard constraints — if ANY fail the iteration is marked unsuccessful. Use `AskUserQuestion` with these 4 questions:

1. **header: "Sharpe floor"** — Minimum acceptable Sharpe ratio.
   - `>0` *(Recommended)* — Just needs to be positive
   - `>0.5`
   - `>1.0` (use Other for custom)

2. **header: "Fill floor"** — Minimum fill count (ensures strategy is actually trading).
   - `>10` *(Recommended)*
   - `>25`
   - `>50` (use Other for custom)

3. **header: "PnL floor"** — Minimum final PnL.
   - `>0` *(Recommended)* — Just needs to be profitable
   - `>100`
   - `>500` (use Other for custom)

4. **header: "Extra threshold"** — Any additional metric threshold? Free-text as `metric_name: ">value"` (e.g. `max_drawdown: "<500"`), or "None".
   - `None`
   - Use Other to type a custom threshold

---

### 7. Round 6 — Targets
Aspirational targets — keep improving until all are met. Use `AskUserQuestion` with these 4 questions:

1. **header: "Sharpe target"**
   - `>1.0` *(Recommended)*
   - `>1.5`
   - `>2.0` (use Other for custom)

2. **header: "Sortino target"**
   - `>1.5` *(Recommended)*
   - `>2.0`
   - `>3.0` (use Other for custom)

3. **header: "Pct pos buckets"** — Fraction of time buckets with positive PnL.
   - `>0.6` *(Recommended)*
   - `>0.7`
   - `>0.8` (use Other for custom)

4. **header: "Extra target"** — Any additional target metric? Free-text as `metric_name: ">value"`, or "None".
   - `None`
   - Use Other to type a custom target

---

### 8. Round 7 (market_maker only) — MM-specific targets
**Only run this round if strategy_type is `market_maker`.**

Use `AskUserQuestion` with these 2 questions:

1. **header: "Time quoting"** — Minimum fraction of time quoting on both sides.
   - `>0.8` *(Recommended)*
   - `>0.7`
   - `>0.9` (use Other for custom)

2. **header: "Edge target"** — Minimum average net edge captured per fill in bps.
   - `>0.1` *(Recommended)*
   - `>0.2`
   - `>0.05` (use Other for custom)

---

### 9. Round 8 — Constraints & meta
Use `AskUserQuestion` with these 3 questions:

1. **header: "Avoid"** — Hard design constraints Claude must always respect. Use Other to type a comma-separated list of custom rules; multiple rules are supported.
   - `max_position must stay under 20` *(default)*
   - `passive orders only — no taking`
   - Use Other to type your own rules (comma-separated for multiple)

2. **header: "Exec mode"** — How should each iteration be executed?
   - `auto` *(Recommended)* — Claude decides: local for logic changes, batch for parameter sweeps
   - `local` — Always run locally via `poetry run gnomepy backtest run`
   - `batch` — Always submit to AWS Batch as a sweep

3. **header: "Sensitivity"** — Robustness checks run once after the strategy first meets all thresholds:
   - `latency + queue` *(Recommended)* — Test both 2x/0.5x latency and risk_averse queue model
   - `latency only` — Test latency sensitivity only
   - `queue only` — Test queue model sensitivity only
   - `none` — Skip sensitivity testing

---

### 10. Round 9 — Interaction & confirmation
Use `AskUserQuestion` with these 2 questions:

1. **header: "Interact"** — How should the loop handle user input between iterations?
   - `autonomous` *(Recommended)* — Runs hands-off; picks up `hints.md` if present, otherwise decides independently
   - `interactive` — Pauses each iteration with a check-in question before forming a hypothesis

2. **header: "Confirm"** — Display a brief summary of all chosen values (strategy type, profile names, metric, thresholds, iterations, execution mode, sensitivity, interaction mode) and ask to confirm.
   - `Looks good — create it`
   - `Start over` — If selected, restart from Round 1

---

### 10. Create directories
```
gnomepy_research/sessions/<name>/
gnomepy_research/sessions/<name>/configs/
gnomepy_research/sessions/<name>/results/
```

Also create an empty `gnomepy_research/sessions/<name>/__init__.py` so the session is importable as a Python module.

---

### 11. Write spec.yaml
Write `gnomepy_research/sessions/<name>/spec.yaml` with all collected values. Rules:

- `name`: session name
- `description`: from Round 1
- `data.listings`: one entry per profile, with `listing_id` and `profile` name from Round 3 loop
- `data.date_ranges`: single entry with start/end from Round 4
- `profiles`: one block per profile. Parse fee values directly from Round 2a answers:
  - Taker/maker fee: extract the leading decimal from the option label (e.g. `0.0005 (5 bps)` → `0.0005`, `-0.0001 (-1 bps rebate)` → `-0.0001`). If the user chose Other, parse their typed value as a decimal.
  Parse latency values from Round 2b answers:
  - Network / processing latency: extract nanoseconds from the option label (e.g. `5ms (5,000,000 ns)` → `5000000`, `500μs (500,000 ns)` → `500000`). If the user chose Other, parse their typed value — accept `Xms` (multiply by 1,000,000), `Xus`/`Xμs` (multiply by 1,000), or raw nanosecond integers.
- `goals.primary_metric` and `goals.direction`: from Round 1
- `goals.thresholds`: sharpe, fill_count, final_pnl from Round 5 plus any custom threshold
- `goals.targets`: sharpe, sortino, pct_positive_buckets from Round 6 plus any custom target
- `goals.mm_targets`: include ONLY if strategy_type == `market_maker`, using values from Round 7
- `constraints.base_preset`: include ONLY if not "none" (from Round 3)
- `constraints.strategy_type`: from Round 1
- `constraints.max_iterations`: from Round 4
- `constraints.signals_to_explore`: include ONLY if non-empty; if user chose "Use defaults for this strategy type", fill in the appropriate defaults for that strategy type
- `constraints.avoid`: list of all avoid rules collected from Round 8
- `meta.execution_mode`: from Round 8 (`auto`, `local`, or `batch`)
- `meta.sensitivity_tests.latency`: `true` if user chose "latency + queue" or "latency only", otherwise `false`
- `meta.sensitivity_tests.queue_model`: `true` if user chose "latency + queue" or "queue only", otherwise `false`
- Omit `meta.sensitivity_tests` entirely if user chose "none"
- `meta.interaction_mode`: from Round 9 (`autonomous` or `interactive`)

Do NOT create strategy.py, session.json, or run any backtests.

---

### 12. Confirm
Tell the user the session is ready at `gnomepy_research/sessions/<name>/` and that they can run `/research <name>` to start iterating. Mention that the first iteration will automatically create a git branch `research/<name>`. Remind them they can edit `spec.yaml` directly if anything needs adjustment.
