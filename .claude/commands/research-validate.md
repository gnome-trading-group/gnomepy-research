# Walk-Forward Validation

Run out-of-sample validation for a converged research session: **$ARGUMENTS**

All paths are relative to the gnomepy-research project root.

---

## Step 1: Load session and best iteration

```bash
poetry run research sessions get $ARGUMENTS
```

Parse the JSON to find `bestIteration`. If `bestIteration` is null or the session has no iterations, stop and tell the user to run `/research $ARGUMENTS` first.

Identify the best iteration's config:
```
gnomepy_research/sessions/$ARGUMENTS/configs/iter_NNN.yaml
```
where NNN is the zero-padded best iteration number.

---

## Step 2: Determine validation windows

Read `gnomepy_research/sessions/$ARGUMENTS/spec.yaml`. Check for a `validation:` section.

**If `spec.validation.walk_forward` exists**, use it directly:
- `total_range.start` / `total_range.end` → the total date range to divide into folds
- `n_folds` → number of folds (default 5)
- `step_mode` → `expanding` or `rolling`

**If `spec.validation` is absent**, use `AskUserQuestion` with these questions:

1. **header: "WF start"** — Start of the total walk-forward range (never overlap your dev range). Use Other for a custom datetime.
   - `2026-02-01T00:00:00`
   - `2026-03-01T00:00:00`
   - `2026-04-01T00:00:00`

2. **header: "WF end"** — End of the total walk-forward range.
   - `2026-02-28T23:59:00`
   - `2026-03-31T23:59:00`
   - `2026-04-30T23:59:00`

3. **header: "Folds"** — Number of OOS folds to divide the range into.
   - `3`
   - `5` *(Recommended)*
   - `10`

4. **header: "Step mode"** — How folds are constructed.
   - `expanding` *(Recommended)* — each fold tests on a new window; training grows
   - `rolling` — fixed-width windows rolling forward

**Holdout:** If `spec.validation.holdout` exists, note its `start`/`end` — you'll run it after walk-forward.

---

## Step 3: Run walk-forward

```bash
poetry run research validate walk-forward $ARGUMENTS \
  --config gnomepy_research/sessions/$ARGUMENTS/configs/iter_NNN.yaml \
  --start <total_start> \
  --end <total_end> \
  --folds <n_folds> \
  --mode <step_mode>
```

This will print a per-fold results table and a PASS/FAIL verdict as it finishes. Each fold's output lands in `gnomepy_research/sessions/$ARGUMENTS/results/walk_forward/fold_NNN/`.

---

## Step 4: Run holdout (if applicable)

If `spec.validation.holdout` is present, run the strategy on that date range:

```bash
poetry run gnomepy backtest run \
  --config gnomepy_research/sessions/$ARGUMENTS/configs/iter_NNN.yaml \
  --output gnomepy_research/sessions/$ARGUMENTS/results/holdout
```

Temporarily override `start_date` / `end_date` in the config to the holdout window. Read `holdout/summary.json` for results.

**Important:** If the holdout result is significantly worse than the walk-forward mean, flag overfitting — the strategy may not generalize.

---

## Step 5: Report results

Present a clear summary:

```
Walk-Forward Validation — <session_name>
Strategy: iter NNN (<title>)

Fold  Date Range                    PnL       Sharpe  Fills
----  ----------------------------  --------  ------  -----
1     2026-02-01 → 2026-02-07       +1.23     0.82    42
2     2026-02-01 → 2026-02-14       +0.87     0.61    38
3     2026-02-01 → 2026-02-21       +2.11     1.14    55
...

Mean OOS PnL:      +1.40
Mean OOS Sharpe:   0.86
% Positive folds:  80%

Holdout (2026-04-01 → 2026-04-30):
  PnL: +0.95  Sharpe: 0.71  Fills: 48
  → Consistent with OOS performance

VERDICT: PASS
```

Verdict rules:
- **PASS** if: mean OOS Sharpe > 0, ≥60% of folds profitable, holdout (if run) is not drastically worse
- **FAIL** if: mean OOS Sharpe ≤ 0, <40% folds profitable, or holdout fails while WF passed

---

## Step 6: Record outcome

Add a note to the session summarizing the validation:
```bash
poetry run research notes add $ARGUMENTS "Walk-forward validation (iter NNN): <N> folds, mean OOS Sharpe <X>, mean OOS PnL <Y>, <Z>% positive folds. Holdout: <result>. Verdict: PASS/FAIL."
```

If verdict is PASS and the session status isn't already `completed`, mark it:
```bash
poetry run research sessions update $ARGUMENTS --status completed
```

If verdict is FAIL, update status to `stalled` — the strategy doesn't generalize and the research session should continue or be abandoned:
```bash
poetry run research sessions update $ARGUMENTS --status stalled
```

---

## Important Notes

- **Never run backtests on the holdout range during the iteration loop** — it is for final validation only. If you've seen it before, it's contaminated.
- The walk-forward range must not overlap with the dev range used during research iterations.
- A FAIL verdict means the strategy is overfit to the dev range, not that it's a bad idea — it's worth starting fresh with a different approach or a genuinely new dev range.
