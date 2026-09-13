# Research Session Status

Show the status of all research sessions, grouped by parent/branch relationships.

If `$ARGUMENTS` is provided, also show the last 3 iterations for that specific session.

---

## Steps

### 1. Fetch all sessions
```bash
poetry run research sessions list --limit 50
```

Parse the JSON output. Each session has: `sessionName`, `status`, `iterationCount`, `bestPnl`, `bestSharpe`, `tags`, `updatedAt`.

### 2. Group sessions by parent/branch relationship
Sessions with a tag matching `parent:<name>` are branches of `<name>`. Build a tree:
- Root sessions: no `parent:` tag
- Branch sessions: have `parent:<name>` tag — nest under their parent

### 3. Display grouped table
Format as a tree with aligned columns. Status symbols: `[running]`, `[completed]`, `[stalled]`, `[paused]`.

```
SESSION STATUS  (as of <timestamp>)
═══════════════════════════════════════════════════════════════════════

n_exchange_arb           [running]    iter 20   sharpe: 0.0001   pnl: $1.32
  |── spread_based       [running]    iter  5   sharpe: 0.4500   pnl: $2.80
  |── latency_arb        [running]    iter  3   sharpe: -0.200   pnl: $0.10
  └── cointegrated       [stalled]    iter 20   sharpe: 0.1200   pnl: $0.45

dutch_book_arb           [running]    iter  1   sharpe:    n/a   pnl:  n/a

equivalent_event_arb     [running]    iter  7   sharpe: 1.2000   pnl: $8.20

prediction_market_mm     [paused]     iter  0   sharpe:    n/a   pnl:  n/a

stablecoin_stat_arb      [stalled]    iter 22   sharpe: 0.3000   pnl: $0.80
```

Branches with `bestSharpe` or `bestPnl` of null/zero show `n/a`.

### 4. If $ARGUMENTS is provided — show iteration detail
Fetch the specific session:
```bash
poetry run research sessions get $ARGUMENTS
```

Display the last 3 iterations with title, metrics, and whether thresholds were met:

```
Last 3 iterations for n_exchange_arb:
─────────────────────────────────────────────────────────────────────
Iter 20 │ [thresholds: FAIL] sharpe=-0.12  pnl=-$0.42  fills=3
        │ EWMA z-score entry with 2-sigma threshold
Iter 19 │ [thresholds: FAIL] sharpe=0.05   pnl=$0.18   fills=12
        │ Added fill timeout guard to prevent leg imbalance
Iter 18 │ [thresholds: PASS] sharpe=0.31   pnl=$1.02   fills=28
        │ Reduced spread threshold to 3bps; added position cap at 10
```

### 5. Output summary line
End with a one-line summary:
- Total sessions: X running, Y completed, Z stalled
- Best performer: `<session_name>` with Sharpe <value> / PnL $<value>
