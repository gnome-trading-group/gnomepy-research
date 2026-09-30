# Walk-Forward Validation

Run out-of-sample validation for a converged research session: **$ARGUMENTS**

All paths are relative to the gnomepy-research project root.

---

## Step 1: Load session and best iteration

```bash
poetry run research sessions get $ARGUMENTS
```

Parse the JSON to find `best_iteration` (the API returns snake_case). If it is null or the session has no iterations, stop and tell the user to run `/research $ARGUMENTS` first.

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

4. **header: "Step mode"** — How the evaluation windows are laid out. No refitting happens between
   folds — the strategy's parameters are fixed from the best iteration — so these are evaluation
   windows, not train/test splits.
   - `rolling` *(Recommended)* — equal, disjoint windows marching forward; each fold scores a distinct stretch
   - `expanding` — every window starts at the range start and grows; the last fold covers everything

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

**Scenario configs:** Walk-forward validation currently operates on single-scenario configs only (it patches top-level `start_date`/`end_date`). If the best iteration's config uses a `scenarios` key, create a single-scenario config for the primary scenario before running walk-forward.

---

## Step 4: Run holdout (if applicable)

If `spec.validation.holdout` is present, write a **copy** of the best iteration's config with the
holdout window and run that. Never edit `configs/iter_NNN.yaml` — it is committed session history:

```bash
poetry run python3 - <<'EOF'
import yaml
from pathlib import Path

src = Path("gnomepy_research/sessions/$ARGUMENTS/configs/iter_NNN.yaml")
cfg = yaml.safe_load(src.read_text())
cfg["start_date"] = "<holdout.start>"
cfg["end_date"] = "<holdout.end>"
cfg.pop("scenarios", None)   # holdout runs a single window
out = src.parent / "holdout.yaml"
out.write_text(yaml.safe_dump(cfg))
print(out)
EOF

poetry run gnomepy backtest run \
  --config gnomepy_research/sessions/$ARGUMENTS/configs/holdout.yaml \
  --output gnomepy_research/sessions/$ARGUMENTS/results/holdout
```

Read `holdout/summary.json` for results.

**Important:** If the holdout result is significantly worse than the walk-forward mean, flag overfitting — the strategy may not generalize.

---

## Step 5: Report results

Present a clear summary:

```
Walk-Forward Validation — <session_name>
Strategy: iter NNN (<title>)
Sharpe is per-bar at 10s bars (x ~1776 for annualized)

Fold  Date Range                    PnL       Sharpe  Fills
----  ----------------------------  --------  ------  -----
1     2026-02-01 → 2026-02-07       +1.23    0.0021    42
2     2026-02-07 → 2026-02-14       +0.87    0.0016    38
3     2026-02-14 → 2026-02-21       +2.11    0.0029    55
...

Mean OOS PnL:      +1.40
Mean OOS Sharpe:   0.0022
% Positive folds:  80%

Holdout (2026-04-01 → 2026-04-30):
  PnL: +0.95  Sharpe: 0.0018  Fills: 48
  → Consistent with OOS performance

VERDICT: PASS
```

**The CLI prints the verdict — report what it printed, don't recompute it.** Its rule is
`PASS` when `mean_oos_sharpe > 0` **and** `pct_positive_folds >= 0.6`, otherwise `FAIL`; there is no
undecided band.

Then add the holdout as a separate judgement: if the holdout result is materially worse than the
walk-forward mean, say so and treat the strategy as overfit even when the CLI printed PASS.

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
