# Create New Research Session

Scaffold a new strategy research session. The session name is: **$ARGUMENTS**

## Steps

### 1. Resolve session name
If `$ARGUMENTS` is empty, use `AskUserQuestion` to ask for a session name before continuing.

The session name must be a valid Python module identifier: lowercase letters, digits, and underscores only — no hyphens, spaces, or other characters. It must not start with a digit. If the provided name is invalid, reject it and ask for a corrected name. Example: `my_strategy` is valid; `my-strategy` and `my strategy` are not.

---

### 2. Round 1 — Core identity
Use `AskUserQuestion` with exactly these 4 questions in a single call:

1. **header: "Strategy type"** (single-select)
   - `market_maker` — Passive quoting on both sides
   - `momentum` — Directional taker strategy
   - `arb` — Cross-exchange or statistical arbitrage
   - `custom` — Other / hybrid

2. **header: "Description"** — Describe the strategy idea: signals, logic, and market conditions. Use options:
   - "I'll describe it below (use Other)"
   - Provide 2-3 short example descriptions as non-selectable placeholders

3. **header: "Primary metric"** (single-select)
   - `final_pnl` — Total PnL at end of backtest
   - `sharpe` — Sharpe ratio
   - `sortino` — Sortino ratio
   - `avg_net_edge_captured_bps` — Avg edge per fill in bps (use Other for a different metric name)

4. **header: "Direction"** (single-select)
   - `maximize` — Higher is better
   - `minimize` — Lower is better

---

### 3. Round 2 — Exchange profiles
Read the available saved profiles by listing `gnomepy_research/profiles/*.yaml`. Each filename (without extension) is a profile name.

Use `AskUserQuestion` with exactly these 3 questions in a single call:

1. **header: "Profile(s)"** — Which exchange profile(s) does this strategy trade on? (multi-select from saved profiles + "Create new")
   - List each saved profile by name (e.g., `hyperliquid`, `polymarket`, `kalshi`, `lighter`)
   - `Create new profile` — Use Other to type a new profile name

   If "Create new profile" is selected, run a sub-round (AskUserQuestion with 5 questions):
   - **"Profile name"** — What to call it (lowercase, e.g. `binance`)
   - **"Taker fee"** — Decimal fee rate (e.g. `0.0005` for 5 bps; negative for rebate). Use Other to type.
   - **"Maker fee"** — Same format. Use Other to type.
   - **"Network latency"** — Round-trip latency. Options: `1ms`, `5ms`, `20ms`, `50ms`, `200ms`. Use Other to type in nanos.
   - **"Queue model"** — `risk_averse` (recommended), `optimistic`, `probabilistic`

   After collecting, write the new profile to `gnomepy_research/profiles/<name>.yaml` so it can be reused in future sessions.

2. **header: "Listing IDs"** — For each selected profile, which listing_id(s) map to it? Free-text as `profile_name: id1,id2` (e.g., `hyperliquid: 1,28` and `lighter: 36`). Use Other to type.

3. **header: "Add another profile?"** (single-select)
   - `Yes` — Configure another exchange
   - `No` — Done with profiles

**If "Yes", loop back and repeat Round 2 for the next profile. Continue until "No".**

---

### 4. Round 3 — Date range & iterations
Use `AskUserQuestion` with exactly these 3 questions in a single call:

1. **header: "Range start"** — Start of dev backtest window (keep window to 30min–2hr). Use Other for custom datetime.
   - `2026-01-23T10:30:00`
   - `2026-02-15T14:00:00`
   - `2026-03-10T09:30:00`

2. **header: "Range end"** — End of dev backtest window. Use Other for custom datetime.
   - `2026-01-23T13:00:00`
   - `2026-02-15T16:30:00`
   - `2026-03-10T12:00:00`

3. **header: "Max iterations"** (single-select)
   - `10`
   - `20` *(Recommended)*
   - `30`
   - `50`

---

### 5. Round 4 — Goals
Use `AskUserQuestion` with exactly these 2 questions in a single call:

1. **header: "Use recommended defaults?"** (single-select) — Show the strategy-type defaults:
   - For `arb`: thresholds `sharpe>0, fill_count>0, final_pnl>0` / targets `sharpe>1.0`
   - For `market_maker`: thresholds `sharpe>0, fill_count>10, final_pnl>0` / targets `sharpe>1.0, sortino>1.5, pct_positive_buckets>0.6` / mm_targets `time_quoting_both_sides>0.8, avg_net_edge_captured_bps>0.1`
   - For `momentum`/`custom`: thresholds `sharpe>0, fill_count>10, final_pnl>0` / targets `sharpe>1.0, sortino>1.5, pct_positive_buckets>0.6`
   
   Options:
   - `Yes — use defaults` *(Recommended)*
   - `No — customize`

2. **header: "Signals to explore"** — Signal hints for Claude (optional, comma-separated). Use Other to type, or skip.
   - `Use defaults for this strategy type`
   - `None / let Claude decide`

**If "No — customize" was selected**, run a customization sub-round (AskUserQuestion with 4 questions):
   - **"Sharpe floor"** — Minimum Sharpe: `>0` *(Recommended)*, `>0.5`, `>1.0`, or Other
   - **"Fill floor"** — Minimum fill count: `>0`, `>10` *(Recommended)*, `>25`, or Other
   - **"Sharpe target"** — Aspirational Sharpe: `>1.0` *(Recommended)*, `>1.5`, `>2.0`, or Other
   - **"Extra targets"** — Any additional targets as `metric: ">value"` (e.g., `max_drawdown: "<500"`), or "None"

---

### 6. Round 5 — Settings & confirm
Use `AskUserQuestion` with exactly these 4 questions in a single call:

1. **header: "Avoid"** — Hard design constraints Claude must respect. Use Other to type comma-separated rules.
   - `None`
   - `max_position must stay under 20` *(default for arb/mm)*
   - `passive orders only — no taking`

2. **header: "Execution mode"**
   - `auto` *(Recommended)* — Claude decides: local for logic changes, batch for parameter sweeps
   - `local` — Always run locally
   - `batch` — Always submit to AWS Batch

3. **header: "Interaction mode"**
   - `autonomous` *(Recommended)* — Runs hands-off; picks up hints.md if present
   - `interactive` — Pauses each iteration with a check-in before proceeding

4. **header: "Confirm"** — Show a brief summary of all collected values and ask to confirm.
   - `Looks good — create it`
   - `Start over` — Restart from Round 1

---

### 7. Create directories
```
gnomepy_research/sessions/<name>/
gnomepy_research/sessions/<name>/configs/
gnomepy_research/sessions/<name>/results/
```

Create an empty `gnomepy_research/sessions/<name>/__init__.py`.

---

### 8. Write spec.yaml
Write `gnomepy_research/sessions/<name>/spec.yaml` with all collected values:

- `name`: session name
- `description`: from Round 1
- `data.listings`: one entry per profile, with `listing_id` and `profile` name. Parse listing IDs from Round 2 — a profile can have multiple listing_ids.
- `data.date_ranges`: single entry with start/end from Round 3
- `profiles`: copy the full YAML content of each selected profile from `gnomepy_research/profiles/<name>.yaml`. For new profiles, use the values just collected.
- `goals.primary_metric` and `goals.direction`: from Round 1
- `goals.thresholds`: from Round 4 (defaults or customized)
- `goals.targets`: from Round 4 (defaults or customized)
- `goals.mm_targets`: include ONLY if strategy_type == `market_maker` (from defaults or omit if customized without mm_targets)
- `constraints.strategy_type`: from Round 1
- `constraints.max_iterations`: from Round 3
- `constraints.signals_to_explore`: include ONLY if non-empty; if "Use defaults", fill in type-appropriate signal hints
- `constraints.avoid`: list of avoid rules from Round 5 (empty list if None)
- `meta.execution_mode`: from Round 5
- `meta.sensitivity_tests.latency`: `true`
- `meta.sensitivity_tests.queue_model`: `true`
- `meta.interaction_mode`: from Round 5

Do NOT create strategy.py or run any backtests.

---

### 9. Register the session in the API
```bash
poetry run research sessions create <name> \
  --spec gnomepy_research/sessions/<name>/spec.yaml \
  --branch research/<name>
```

If the session already exists, the command will print a message and exit cleanly.

---

### 10. Confirm
Tell the user the session is ready at `gnomepy_research/sessions/<name>/` and is now visible in the web UI under Research. They can run `/research <name>` to start iterating, or `/loop /research <name>` for continuous iteration. The first iteration will automatically create git branch `research/<name>`. They can edit `spec.yaml` directly if anything needs adjustment. If they want to explore a different approach with the same setup later, use `/research-branch <name> <suffix>`.
