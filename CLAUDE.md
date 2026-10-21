# gnomepy-research

## Commands

```bash
poetry install --all-extras   # install dev environment (every dependency family)
poetry run pytest       # run all tests
scripts/release.sh patch|minor|major   # from main: bump, commit, tag vX.Y.Z, push; gnome-pipelines pins the tag
python3 gnomepy_research/dependencies.py constraints   # after every `poetry lock`: rewrite constraints.txt
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
4. Each branch gets its own worktree — open separate terminals, cd to each worktree, run `poetry install --all-extras`, then `/loop /research my_arb__approach_a` and `/loop /research my_arb__approach_b`
5. Monitor all branches: `/research-status`

### Session structure
```
gnomepy_research/sessions/<name>/
  __init__.py       # makes the session importable; declares EXTRAS if its code needs a dependency family
  spec.yaml         # user-authored goals and constraints — DO NOT MODIFY
  strategy.py       # the strategy class itself (modified each iteration); configs name it
                    #   gnomepy_research.sessions.<name>.strategy:<Class>
  *.py              # session-local signals or utilities, if needed
  best/             # snapshot of all .py files from the last accepted iteration
  configs/          # per-iteration backtest YAML configs and sweep configs
  results/          # backtest outputs (per-iteration subdirectories)
  notes/            # local note files synced from API (poetry run research notes pull <name>)
```

`strategy.py` holds the class, never a module-level instance: a config's `strategy.args` are the
parameters. A branch session `<parent>__<suffix>` may subclass its parent's `strategy.py`; no other
session imports from another. Configs record the run they were written for — an old iteration's
args need not match today's constructor.

Only the small files in `results/` are committed (`summary.json`, `sweep_summary.json`,
`metadata.json`, `config.yaml`, and `.txt`/`.csv` reports); parquet, HTML and everything else
stays local. `.gitignore` holds the list.

**Code scope (autonomous iterations only):** during a `/research` iteration, all code changes stay under `sessions/<name>/` — never modify `gnomepy_research/signals/`, `gnomepy_research/arb/`, or the rest of the shared package. This is a guardrail on the unattended loop, not a rule about sessions generally; working on a session directly, reach for whatever the work needs. If a session needs a modified signal or utility, copy it into the session directory and import from there. Promote session-local code to shared locations only after the session completes.

For loop-driven sessions, state (iterations, notes, status) is stored in the API — viewable at the Research page in the web UI. **API responses are snake_case** (`session_name`, `iteration_count`, `best_iteration`, `best_pnl`, `best_sharpe`). `research sessions list --json` and `research validate significance --json` emit raw JSON for scripting.

**Accept/reject gate:** After each evaluation, `/research` compares the current iteration's primary metric against the best accepted baseline, in the direction given by `spec.goals.direction`. On accept: all session `.py` files (except `__init__.py`) are snapshotted to `best/` and `best_iteration` is set in the API. On reject: session `.py` files are restored from `best/` *before* the commit (new files added in the rejected iteration are removed), so the commit records the reverted state and the tree is left clean. Both outcomes are recorded in iteration metadata (`accepted: true/false`, `returned_to: N`).

**Cross-session memory:** `gnomepy_research/research_learnings.md` holds **one entry per session**, updated in place when a session completes or stalls. `/research` Step 2 reads this file before forming each hypothesis — avoids re-running dead ends and bootstraps from known-good approaches. Learnings are also pushed as session notes (`poetry run research notes add`) for web UI visibility.

**Diagnostics:** `/research` Step 5 runs mandatory per-strategy-type diagnostic checks every iteration via `gnomepy_research.analysis` (`compute_mm_stats`, `compute_adverse_selection`, `compute_rolling_sharpe`) rather than hand-rolling them from parquet. Results are included in the iteration analysis and directly inform the next hypothesis.

### Simulation parameters
Each exchange's simulation parameters (network and order latency, market data latency, queue model, self-trade prevention) live in the registry, under the controller's Security Master → Simulation tab. Every backtest and paper session simulates each listing with its exchange's parameters, so a config states nothing about simulation unless an experiment changes it. Fees always come from the registry's fee schedules.

To change how a venue behaves for one experiment, override blocks by exchange code; a block is replaced whole, and the rest keep the registry's values:

```yaml
exchanges:
  KALSHI:
    queue_model: {type: optimistic}
scenarios:
  slow_polymarket:
    exchanges:                       # this scenario only, laid over the top-level blocks for the exchange
      POLYMARKET_INTL:
        order_processing_latency: {type: maker_taker, base_nanos: 5000000, taker_delay_nanos: 3000000000, maker_delay_nanos: 0}
listings:
  - listing_id: 48212
    overrides:                       # one listing, over its exchange's
      network_latency: {type: static, latency_nanos: 1}
```

A run records the time it read the registry (`simulation_as_of` in its metadata) and each listing's resolved parameters; set `simulation_as_of` in a config to rerun with exactly the parameters an earlier run read. On 2026-10-16 every config's assumed latency and queue values (50ms network, the standard maker/taker delays, risk-averse queue) were removed so they simulate with the registry's; only deliberate ones remain (latency runs, optimistic queues, unusual values). Results recorded before then used the old assumptions.

### Iteration modes
- **Local run**: for logic changes — writes a config YAML, runs via `poetry run gnomepy backtest run --config <path>`
- **Remote sweep**: for parameter search — commits+pushes the session branch, submits to AWS Batch via `gnomepy backtest submit --research-commit <sha>`. Parameters to sweep go in a top-level `sweep:` section (see below) — lists in `strategy.args` are always passed as-is, never swept.
- **Multi-scenario run**: for testing across multiple events/listings simultaneously — writes a config with a `scenarios` key; each scenario has its own `listings`, `start_date`, `end_date`, and optional `strategy_args` overrides. Works for both local runs and remote sweeps; scenarios compose with sweep params (total jobs = scenarios × sweep combinations). Each scenario gets its own report and summary.

### Sweep config format
Parameters to sweep are declared in a top-level `sweep:` section, keyed on `strategy` (for strategy args), `exchanges` (for an exchange's simulation blocks) and `listings` (for one listing's, by id). Everything outside `sweep` is always fixed — only the `sweep` section is expanded. A swept simulation block must name its model (`type`) in the config:

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
  exchanges:
    KALSHI:
      network_latency:
        latency_nanos: [5000000, 10000000]   # simulation sweep; needs exchanges.KALSHI.network_latency.type
```

## Where code goes

| Location | What belongs there |
|---|---|
| `models/` | Production models and features, imported by gnome-pipelines and live strategies. Tested, released by tag. |
| `sessions/<name>/` | Everything a session needs that no other session does: its strategy, signals, analysis scripts. Live strategies run from their session too (`sessions/cs2_series`). |
| `signals/` | The composable `Signal` library (catalog in tutorial 03). |
| `arb/` | Pricing, book-walking and leg tracking shared by the prediction-market arb sessions. |
| `fees.py`, `listing.py` | Registry fee rates and the `Listing` key, for any strategy (live ones included). |
| `analysis/` | Reading backtest results: loading and comparing runs, market-making and adverse-selection diagnostics, rolling performance, significance and walk-forward validation, signal attribution. |
| `cli.py`, `api.py`, `notes_sync.py`, `environment.py`, `dependencies.py` | The `research` CLI, the research API client, and packaging. |

Shared code is code more than one session uses. A strategy class used by one session family lives
in that session, not in a shared `strategies/` package; promote code out of a session only once a
second session needs it.

Tests mirror the package: `tests/models/...` for production code, `tests/sessions/` for session
code, and `tests/<module>/` for the shared modules.

## Production code vs sessions

gnome-pipelines and live strategies depend on this package (pinned by git tag). Pipelines import only
from `gnomepy_research/models/`; a live strategy is its session's `strategy.py`, importing only
`models/`, its own session, and the base-only `fees`/`listing` modules:

```
models/cs2/hltv/             HLTV point-in-time features and priors; Bayesian series pricing (base-only)
models/cs2/bayes/            Bayesian series model: data preparation and NUTS fitting
models/cs2/win_probability/  pre-map, round and series models
models/cs2/prematch/         Polymarket market lookup, HLTV team identity
models/cfb/features/         CFB point-in-time features
models/cfb/efficiency/       dynamic efficiency model and posteriors
models/cfb/hyperparameters/  its fitted hyperparameters
```

Sessions build on `models/`, never the other way round: nothing in `models/` imports from
`sessions/`. When a session's model code goes to production, promote it into `models/` first, and treat
changes there as production changes: tested, and released by tagging this repo.

## Dependency families

The base install (`dependencies` in pyproject.toml) is what every live strategy needs. Heavier
libraries are extras named for what they provide, not for a sport, so a session or pipeline
installs only what its code uses:

| Extra | Brings |
|---|---|
| `xgboost` | xgboost (`xgboost-cpu` on Linux), scikit-learn, joblib |
| `pymc` | pymc, `nutpie[pymc]` (our models sample with nutpie; it brings numba) |
| `websocket` | websocket-client |
| `research` | gnomepy[backtest] (plotly, dash, ...) for `analysis` and the CLI |

- A package declares the extras its code needs in its `__init__.py`: `EXTRAS = ("xgboost",)`.
  The nearest enclosing package that declares EXTRAS decides for every module in it; code with
  none runs on the base. Today: `models/cs2/win_probability` → xgboost, `models/cs2/bayes` and
  `models/cfb/efficiency` → pymc, and the sessions that run them. `models/cs2/hltv`, `models/cs2/prematch` and
  `models/cfb/features` are base-only.
- gnome-orchestrator's `start-session.sh` runs
  `python3 gnomepy_research/dependencies.py extras <strategy class>` and installs
  `gnomepy-research[<extras>]` pinned by `constraints.txt`.
- gnome-pipelines groups pipelines into images by domain (cs2, cfb), and each image installs
  the extras its pipelines use.
- `tests/test_dependency_families.py` follows every module's imports through the package and fails
  if one reaches a library its extras don't install. A new heavy import means adding it to an
  extra (or a new one) and declaring that extra on the package.
- `constraints.txt` pins every package to its poetry.lock version. Regenerate it after `poetry lock`;
  the test fails while it is stale.
- Models a live session loads are saved in formats that outlive library versions (xgboost's own
  UBJ format, plain numbers for calibrators), never pickles: a pickle only loads in the exact
  versions that wrote it, and training (gnome-pipelines) and live sessions resolve their
  environments separately. The pre-map bundle (`pre_map_model.write_bundle`) is the example; the
  round model (`win_probability/model.py`) is still a pickle and must be converted before it goes live.

## Artifacts and Datasets

Datasets are append-only Iceberg tables on S3 Tables and models are versioned artifact files indexed
beside them, both through `gnomepy.data`. Research only reads them: every writer (scrapers, ingestion,
priors, feature snapshots, training) lives in gnome-pipelines with the table schemas. See
`tutorials/04_artifacts_and_datasets.md`.

**CLI:**
```bash
poetry run research artifacts list [--type TYPE] [--name NAME]
poetry run research artifacts get <type/name[:version|@alias]>

poetry run research datasets list
poetry run research datasets get <name> [--as-of 2026-10-01T00:00:00+00:00]
```

**In Python:**
```python
from gnomepy.data import DatasetStore, resolve_artifact_path

df = DatasetStore().load("cs2_match_history")                     # newest row per key
df = DatasetStore().load("cs2_match_history", as_of=some_datetime)  # as it was known then

local_path = resolve_artifact_path("artifact://xgboost_model/cs2_pre_map_win_prob")  # artifact://, s3:// or a local path
```

**Artifact URIs in YAML configs:**
```yaml
model_path: "artifact://xgboost_model/cs2_pre_map_win_prob"                          # latest
model_path: "artifact://xgboost_model/cs2_pre_map_win_prob:20261013T165937051464Z"   # pinned
model_path: "artifact://xgboost_model/cs2_pre_map_win_prob@production"               # alias
```

## Research jobs (AWS Batch)

Any module runnable with `python -m` can run on spot AWS Batch at a pushed commit, one task per argument
list, visible to everyone on the controller's Research → Jobs page:

```bash
poetry run research jobs submit <module> --commit research/<topic> [--input NAME=PATH ...] [--array-file tasks.json | -- ARGS...] [--cpu 4 --memory-gib 14]
poetry run research jobs list | get <run_id> | cancel <run_id>
AWS_PROFILE=dev poetry run research jobs download <run_id> <dir> [--flat]
```

`--commit` resolves to the SHA origin has (push first; `research/*` branches are for this). A task reads
`--input` sets at `/work/in/<NAME>/` (uploaded once, content-addressed), datasets and artifacts through
`gnomepy.data`, and writes what it keeps to `$OUTPUT_DIR`; outputs and the log land in
`s3://gnome-research-<stage>/research-jobs/runs/<run_id>/jobs/<i>/`. Defaults: 4 vCPU / 14 GiB per task.
Research runs in prod (`STAGE=prod`, after `STAGE=prod poetry run gnomepy login`), with up to 128 vCPUs across all
research jobs; dev is capped at 16 for testing the runner. The runner image is gnomepy's `docker/research-job/`.

## Tutorials

- `tutorials/01_getting_started.md` — session creation through first backtest result
- `tutorials/02_research_workflow.md` — evaluate → iterate → validate, and what the metrics mean
- `tutorials/03_strategy_building.md` — signal catalog (generated) and strategy patterns
- `tutorials/04_artifacts_and_datasets.md` — reading models and datasets
- `tutorials/05_analysis_toolkit.md` — `analysis`, `arb`, `environment`

The signal tables in `03` are generated: `poetry run python scripts/gen_signal_catalog.py`.
`tests/test_docs_sync.py` fails if they drift from `gnomepy_research.signals.__all__`.

## Code conventions
- All imports at the top of the file — never inside functions or conditionals
- No comments unless the WHY is non-obvious
- Strategies subclass `gnomepy.Strategy`
- Signals compose from `gnomepy_research.signals`
- Prices are scaled integers (divide by 1e9 for display) — never pass floats to `Intent`
- Return `[]` from `on_execution_report` unless reactive logic is required
