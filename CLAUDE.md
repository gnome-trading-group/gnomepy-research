# gnomepy-research

## Commands

```bash
poetry install          # install dev environment
poetry run pytest       # run all tests
```

## Strategy Research Sessions

Research sessions live in `gnomepy_research/sessions/<name>/`. A session is just a directory of
research — the `/research` commands are an **optional** tool for iterating on backtest-parameter
problems, not a requirement. Plenty of sessions (model training, data pipelines, one-off analysis)
never use them, and that is fine: no branch, no iteration records, no accept/reject gate needed.

To use the loop: `/research-new <name>`, then `/research <name>` (or `/loop /research <name>` for
continuous iteration).

A session driven by `/research` runs on its own git branch `research/<name>`, created automatically on
the first iteration. Sessions not driven by it commit wherever you are working.

> **`sharpe` is a per-bar ratio at 10s bars, not annualized** — multiply by ~1776 for the annualized
> figure. Set `goals.targets.sharpe` in per-bar units (0.002 ≈ 3.6 annualized). Targets written as if
> the metric were annualized are unreachable; this stalled every session run before 2026-09-30, so
> treat their Sharpe verdicts in `research_learnings.md` with suspicion.

**Commands:**
- `/research-new <name>` — Create a new session (5-round wizard)
- `/research <name>` — Run one iteration
- `/loop /research <name>` — Run continuously until targets are met or max iterations reached
- `/research-branch <parent> <suffix>` — Fork an existing session to explore a different approach in parallel. Creates session `<parent>__<suffix>` with its own git worktree for parallel execution.
- `/research-status` — Dashboard showing all sessions grouped by parent/branch, with current metrics
- `/research-validate <name>` — Walk-forward out-of-sample validation
- `/research-ablate <name>` — Turn each signal off in turn to see which ones earn their place
- `/research-clean <name>` — Tear down a finished session's worktree, branch and results

**Parallel exploration workflow:**
1. Create a base session: `/research-new my_arb`
2. Run a few iterations to establish a working strategy: `/research my_arb`
3. Branch into parallel explorations: `/research-branch my_arb approach_a`, `/research-branch my_arb approach_b`
4. Each branch gets its own worktree — open separate terminals, cd to each worktree, run `poetry install`, then `/loop /research my_arb__approach_a` and `/loop /research my_arb__approach_b`
5. Monitor all branches: `/research-status`

### Session structure
```
gnomepy_research/sessions/<name>/
  __init__.py       # empty — makes the session importable as a Python module
  spec.yaml         # user-authored goals and constraints — DO NOT MODIFY
  strategy.py       # current working strategy (modified each iteration)
  *.py              # session-local signals or utilities, if needed
  best/             # snapshot of all .py files from the last accepted iteration
  configs/          # per-iteration backtest YAML configs and sweep configs
  results/          # backtest outputs (per-iteration subdirectories)
  notes/            # local note files synced from API (poetry run research notes pull <name>)
```

**Code scope (autonomous iterations only):** during a `/research` iteration, all code changes stay under `sessions/<name>/` — never modify `gnomepy_research/signals/`, `gnomepy_research/strategies/`, or the rest of the shared package. This is a guardrail on the unattended loop, not a rule about sessions generally; working on a session directly, reach for whatever the work needs. If a session needs a modified signal or utility, copy it into the session directory and import from there. Promote session-local code to shared locations only after the session completes.

For loop-driven sessions, state (iterations, notes, status) is stored in the API — viewable at the Research page in the web UI. **API responses are snake_case** (`session_name`, `iteration_count`, `best_iteration`, `best_pnl`, `best_sharpe`). `research sessions list --json` and `research validate significance --json` emit raw JSON for scripting.

**Accept/reject gate:** After each evaluation, `/research` compares the current iteration's primary metric against the best accepted baseline, in the direction given by `spec.goals.direction`. On accept: all session `.py` files (except `__init__.py`) are snapshotted to `best/` and `best_iteration` is set in the API. On reject: session `.py` files are restored from `best/` *before* the commit (new files added in the rejected iteration are removed), so the commit records the reverted state and the tree is left clean. Both outcomes are recorded in iteration metadata (`accepted: true/false`, `returned_to: N`).

**Cross-session memory:** `gnomepy_research/research_learnings.md` holds **one entry per session**, updated in place when a session completes or stalls. `/research` Step 2 reads this file before forming each hypothesis — avoids re-running dead ends and bootstraps from known-good approaches. Learnings are also pushed as session notes (`poetry run research notes add`) for web UI visibility.

**Diagnostics:** `/research` Step 5 runs mandatory per-strategy-type diagnostic checks every iteration via `gnomepy_research.reporting.backtest` (`compute_mm_stats`, `compute_adverse_selection`, `compute_rolling_sharpe`) rather than hand-rolling them from parquet. Results are included in the iteration analysis and directly inform the next hypothesis.

### Exchange profiles
Saved exchange profiles live in `gnomepy_research/profiles/`. Current profiles: `hyperliquid`, `lighter`, `polymarket`, `kalshi`. The `/research-new` wizard picks these up automatically so you don't re-specify fees and latency each time. New profiles created during `/research-new` are saved here for future reuse.

### Iteration modes
- **Local run**: for logic changes — writes a config YAML, runs via `poetry run gnomepy backtest run --config <path>`
- **Remote sweep**: for parameter search — commits+pushes the session branch, submits to AWS Batch via `gnomepy backtest submit --research-commit <sha>`. Parameters to sweep go in a top-level `sweep:` section (see below) — lists in `strategy.args` are always passed as-is, never swept.
- **Multi-scenario run**: for testing across multiple events/listings simultaneously — writes a config with a `scenarios` key; each scenario has its own `listings`, `start_date`, `end_date`, and optional `strategy_args` overrides. Works for both local runs and remote sweeps; scenarios compose with sweep params (total jobs = scenarios × sweep combinations). Each scenario gets its own report and summary.

### Sweep config format
Parameters to sweep are declared in a top-level `sweep:` section, keyed on `strategy` (for strategy args) and `profiles` (for profile values). Everything in `strategy.args` and `profiles` is always fixed — only the `sweep` section is expanded:

```yaml
strategy:
  args:
    gamma: 0.5      # fixed default
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

## Artifacts and Datasets

Trained models and tabular training data are stored in S3 (`gnome-research-{STAGE}`) and registered in DynamoDB. See `tutorials/04_artifacts_and_datasets.md` for full workflows.

**CLI:**
```bash
poetry run research artifacts list [--type TYPE] [--name NAME]
poetry run research artifacts publish <path> --type TYPE --name NAME [--session SESSION]
poetry run research artifacts get <type/name[:version]>

poetry run research datasets list [--name NAME]
poetry run research datasets publish <parquet_path> --name NAME
poetry run research datasets get <name[:version]>
```

**In Python:**
```python
from gnomepy_research.artifacts import ArtifactStore, DatasetStore, resolve_artifact_path

# publish a model
ArtifactStore().publish("model.xgb", artifact_type="xgboost_model", name="cs2_fair_value", session_name="cs2_xgb")

# load in a strategy (artifact://, s3://, or local path all work)
local_path = resolve_artifact_path("artifact://xgboost_model/cs2_fair_value")

# publish/load a dataset
DatasetStore().publish(df, name="cs2_match_features")
df = DatasetStore().load("cs2_match_features")
```

**Artifact URI scheme in YAML configs:**
```yaml
value_function_path: "artifact://value_function/kalshi_cal"    # latest
value_function_path: "artifact://value_function/kalshi_cal:3"  # pinned
```
Old local paths continue to work — `resolve_artifact_path` passes them through unchanged.

## Tutorials

- `tutorials/01_getting_started.md` — session creation through first backtest result
- `tutorials/02_research_workflow.md` — evaluate → iterate → validate, and what the metrics mean
- `tutorials/03_strategy_building.md` — signal catalog (generated) and strategy patterns
- `tutorials/04_artifacts_and_datasets.md` — models and datasets in S3
- `tutorials/05_analysis_toolkit.md` — `explore`, `analysis`, `validation`, `reporting.backtest`, `arb`

The signal tables in `03` are generated: `poetry run python scripts/gen_signal_catalog.py`.
`tests/test_docs_sync.py` fails if they drift from `gnomepy_research.signals.__all__`.

## Code conventions
- All imports at the top of the file — never inside functions or conditionals
- No comments unless the WHY is non-obvious
- Strategies subclass `gnomepy.Strategy`
- Signals compose from `gnomepy_research.signals`
- Prices are scaled integers (divide by 1e9 for display) — never pass floats to `Intent`
- Return `[]` from `on_execution_report` unless reactive logic is required
