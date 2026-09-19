# Research Workflow

This tutorial covers the evaluation → iterate → validate lifecycle after your first backtest result.

---

## Evaluating Results

Every iteration produces a `summary.json` with the core metrics. The research loop checks these automatically, but you should understand what they mean:

| Metric | What it measures | When to worry |
|--------|-----------------|---------------|
| `sharpe` | Risk-adjusted return (annualized) | < 0 means losing money on average |
| `sortino` | Like Sharpe but only penalizes downside vol | Low Sortino relative to Sharpe → big losing streaks |
| `final_pnl` | Total PnL in the backtest window | Positive but near zero → fees eating the edge |
| `fill_count` | Number of fills | < 20: Sharpe is statistically meaningless |
| `pct_positive_buckets` | Fraction of time windows with positive PnL | < 0.5 → not consistent |
| `total_fees` | Total fees paid | If fees > gross PnL → strategy has no edge before fees |

**Sharpe thresholds at bar level:** At 10-second bars, the minimum backtest length (at 95% confidence) for a bar-level Sharpe of 0.5 is roughly 80 observations. A 30-minute session has ~180 bars — barely enough to trust a moderate signal. This is why we use a short dev range for fast iteration but validate on longer windows before declaring success.

**Statistical significance** is checked automatically once all thresholds pass. The two key numbers:

- **Deflated Sharpe Ratio (DSR)**: probability that the observed Sharpe reflects genuine skill after accounting for the number of iterations tried. DSR > 0.95 means significant at 5%. If you've run 20 iterations and DSR is 0.7, you need a meaningfully higher Sharpe to clear the multiple-testing hurdle.
- **Bootstrap CI**: `[lo, hi]` confidence interval for the Sharpe. If `lo < 0`, the true Sharpe might be negative.

---

## Deep-Dive Analysis

When a result looks interesting — or suspicious — open `notebooks/02_backtest_deep_dive.ipynb`. It loads the parquet files and walks through:

1. **PnL curve** — is it smooth and consistent, or driven by 1-2 big trades?
2. **Worst drawdown period** — zoom in and see what happened in the market
3. **Fill distribution by hour** — is the strategy only making money in one time window?
4. **Adverse selection** — does the market move against us within 1 second of fills? If yes, we're being picked off
5. **Rolling Sharpe** — is the edge stable across the session, or decaying?
6. **Alpha decay** — do early windows have higher Sharpe than late ones? Signals the strategy's edge degrades as the session progresses
7. **Regime analysis** — does PnL concentrate in one market condition (low vol, tight spread)?

The notebook ends with a **red flags checklist** that runs programmatically. Pay attention to:
- Top 10% of bars driving > 80% of positive returns (lucky outlier, not signal)
- Adverse selection rate > 55% (being picked off consistently)
- PnL from only one regime (not robust)

---

## Making Notes

Notes persist across sessions and sync to Obsidian. Record observations, dead ends, and hypotheses — not just what happened, but why you think it happened and what to try next.

**Add a note from the command line:**
```bash
poetry run research notes add <session_name> "Spread widens dramatically at session open. EWMA needs longer warmup to avoid false signals in the first 5 minutes."
```

**Sync notes to local files** (for Obsidian):
```bash
# Pull API notes → local markdown files in sessions/<name>/notes/
poetry run research notes pull <session_name>

# Push local edits back to the API
poetry run research notes push <session_name>
```

Local note files get YAML frontmatter for Obsidian:
```yaml
---
session: n_exchange_arb
created_at: 2026-07-10T14:30:00
---
Spread widens dramatically at session open...
```

**What to record:**
- Hypotheses you're about to test (before the next iteration)
- What the parquet analysis revealed that `summary.json` didn't show
- Dead ends and why they failed — avoids re-testing the same idea
- Structural observations about the market (regime patterns, fill clustering, staleness patterns)

---

## The Iteration Protocol

Each `/research` run modifies exactly one file: `strategy.py`. Everything else is derived:
- `configs/iter_NNN.yaml` is written fresh from `spec.yaml` + the current `strategy.py` constructor signature
- `results/iter_NNN/` holds backtest output
- The API stores the iteration record (hypothesis, analysis, metrics, next action)

**What the loop tracks between iterations:**

The `next_action` field in each iteration record tells the next iteration what to try. The loop reads the last 2-3 iteration records from the API to understand recent history. If the primary metric hasn't improved by > 5% relative for 3 consecutive iterations, the loop treats it as a plateau and tries something fundamentally different — new signals, different strategy class, or a structural overhaul.

**Why only `strategy.py` changes:**

`spec.yaml` is the user's contract. The loop must not move the goalposts. If you want to try a different date range or change fees, start a new session. If you want to guide the current session, use hints.

---

## Injecting Hints

When the loop is running autonomously and you want to steer it without stopping:
```
/research-hint <session_name>
```

This appends a timestamped directive to `hints.md`:
```
[2026-07-10T15:00:00] Try increasing ewma_alpha to 0.995 — the current warmup is too short for the session open noise
```

On the next iteration, the loop reads `hints.md` first. Hints take priority over the `next_action` from the previous iteration. After consuming them, `hints.md` is cleared.

You can queue multiple hints before the next iteration runs — they are all consumed together.

**When to use hints vs letting the loop run:**
- Use hints when you notice something in the parquet data that the loop can't see (e.g. a specific market event causing fills to cluster)
- Use hints when the loop has been doing minor parameter tweaks for 3+ iterations and needs a structural push
- Let the loop run when the hypothesis is plausible and the metric is improving

---

## Local Runs vs Remote Sweeps

The loop decides automatically when `execution_mode: auto` (the default):

**Local run** — for logic changes:
```bash
poetry run gnomepy backtest run \
  --config gnomepy_research/sessions/<name>/configs/iter_007.yaml \
  --output gnomepy_research/sessions/<name>/results/iter_007
```
Results are available in seconds. Use this to validate that a new signal or structural change actually works before searching parameters.

**Remote sweep** — for parameter search:
```bash
# Loop commits + pushes, then calls:
poetry run gnomepy backtest submit \
  --config gnomepy_research/sessions/<name>/configs/sweep_008.yaml \
  --research-commit <sha>
```
The sweep config uses list or range syntax:
```yaml
strategy:
  args:
    gamma: [0.5, 1.0, 2.0, 4.0]              # 4 values
    delta: {min: 0.5, max: 3.0, step: 0.5}   # 6 values → 24 jobs total
```
Hard cap: the cartesian product must not exceed 100 jobs. AWS Batch runs all jobs in parallel; results download to `results/iter_NNN/` per job, and the loop picks the best by `primary_metric`.

**Multi-scenario configs** — for testing across multiple events or listing sets in a single run:
```yaml
strategy:
  class_name: "gnomepy_research.sessions.my_arb.strategy:MyArb"
  args:
    min_pure_arb_bps: [5, 10]   # optional sweep — crosses with scenarios

scenarios:
  baseball:
    start_date: "2026-08-24T00:23:00"
    end_date: "2026-08-24T02:12:00"
    listings:
      - listing_id: 217772
        profile: kalshi
      - listing_id: 229125
        profile: polymarket
    strategy_args:
      event_ids: [[46259, 48600]]
  football:
    start_date: "2026-08-24T00:26:00"
    end_date: "2026-08-24T03:09:00"
    listings:
      - listing_id: 222852
        profile: kalshi
      - listing_id: 97202
        profile: polymarket
    strategy_args:
      event_ids: [[47431, 18169]]

profiles:
  polymarket: { ... }
  kalshi: { ... }
```
Scenarios compose with sweeps — total jobs = `len(scenarios) × product(sweep_lengths)`. Each scenario gets its own report and summary. Works for both local runs (`gnomepy backtest run`) and AWS Batch (`gnomepy backtest submit`). The hard cap of 100 jobs applies to the combined total.

**When to sweep vs iterate locally:**
- Always validate logic locally first — sweeping a broken strategy wastes time
- Sweep when you have 2+ continuous parameters and the strategy structure is stable
- Don't sweep more than 3 parameters at once — the search space grows exponentially and the best combination may not generalize

---

## Statistical Validation

Once all targets are met, run `/research-validate <session_name>`:
```
/research-validate n_exchange_arb
```

This runs a walk-forward out-of-sample test on a date range that was **never touched during iteration**. The loop is forbidden from running backtests on the holdout range during research — once you've seen it, it's contaminated.

**Walk-forward with 5 expanding folds** (the default):
- Fold 1: OOS on weeks 1-1
- Fold 2: OOS on weeks 1-2
- Fold 3: OOS on weeks 1-3
- ...

**Interpreting the verdict:**
- **PASS**: mean OOS Sharpe > 0, ≥ 60% of folds profitable, holdout consistent with OOS
- **FAIL**: the strategy is overfit to the dev range. Not necessarily a bad idea — may be worth starting fresh on a new date range

**Minimum Backtest Length** — a useful sanity check before validating. For a bar-level Sharpe of `SR`, the minimum number of bars for 95% confidence is approximately:
```
T_min = (1 + SR² / 2) / (SR / 1.645)²
```
For SR = 0.5: ~87 bars (14 minutes at 10s bars). For SR = 0.1: ~2185 bars (6 hours). Low Sharpe strategies need very long windows to validate reliably.

**Deflated Sharpe Ratio** — the DSR adjusts for the number of iterations tried. After 20 iterations with 100 strategy variants each, the expected maximum Sharpe under the null hypothesis is around 2.5. A strategy showing Sharpe = 2.0 after 20 iterations likely reflects lucky parameter fitting, not genuine skill. The loop prints DSR alongside each iteration's metrics once thresholds are met.

---

## Cross-Session Comparison

Once you have multiple sessions with good results, open `notebooks/03_cross_session_comparison.ipynb`:

1. **Summary table** — side-by-side Sharpe, PnL, fill count across sessions
2. **PnL overlay** — normalized curves on the same axes to spot correlation
3. **Correlation matrix** — low correlation between strategies is valuable (diversification)
4. **Combined Sharpe** — if two strategies are 0.3 correlated with individual Sharpe 1.5, the combined Sharpe at equal weight is approximately `1.5 × √2 × 1/√(1 + 0.3) ≈ 1.86`
5. **Per-regime PnL** — does strategy A perform where strategy B doesn't?

---

## When to Stop Iterating

**Stop and declare success** when:
- All `goals.targets` are met (Sharpe, Sortino, % positive buckets)
- Walk-forward PASS with ≥ 60% positive folds
- DSR > 0.95 (significant after multiple-testing correction)

**Stop and restart fresh** when:
- Walk-forward FAIL — the strategy memorized the dev range
- 20 iterations in, primary metric hasn't improved past 0.5 Sharpe — the signal hypothesis is wrong
- DSR < 0.5 despite a superficially good Sharpe — you've exhausted your trial budget

**Don't stop just because results look noisy.** A Sharpe of 0.8 on 30 minutes of data with 40 fills is consistent with a true Sharpe of 1.5 — the window is too short to be sure. Add more date ranges to `spec.yaml` for validation, but don't interpret noise as evidence the strategy doesn't work.

---

## Next Steps

- Read **`03_strategy_building.md`** for signal DSL, strategy patterns, and code conventions
- Open `notebooks/02_backtest_deep_dive.ipynb` for structured result analysis
- Open `notebooks/03_cross_session_comparison.ipynb` to compare sessions
