# gnomepy-research

## Commands

```bash
poetry install          # install dev environment
poetry run pytest       # run all tests
```

## Strategy Research Sessions

Research sessions live in `gnomepy_research/sessions/<name>/`. Start a new session with `/research-new <name>`, then run `/research <name>` (or `/loop /research <name>` for continuous iteration).

Each session runs on its own git branch `research/<name>`, created automatically on the first iteration.

**Commands:**
- `/research-new <name>` — Create a new session (5-round wizard)
- `/research <name>` — Run one iteration
- `/loop /research <name>` — Run continuously until targets are met or max iterations reached
- `/research-branch <parent> <suffix>` — Fork an existing session to explore a different approach in parallel. Creates session `<parent>__<suffix>` with its own git worktree for parallel execution.
- `/research-status` — Dashboard showing all sessions grouped by parent/branch, with current metrics
- `/research-validate <name>` — Walk-forward out-of-sample validation
- `/research-hint <name>` — Queue a directive for the next autonomous iteration

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

**Code scope:** All code changes during a session stay under `sessions/<name>/`. Never modify files in `gnomepy_research/signals/`, `gnomepy_research/strategies/`, or elsewhere in the shared package — those are stable. If a session needs a modified signal or utility, copy it into the session directory and import from there. Promote session-local code to shared locations only after the session completes.

Session state (iterations, notes, status) is stored in the API — viewable at the Research page in the web UI.

**Accept/reject gate:** After each evaluation, `/research` compares the current iteration's primary metric against the best accepted baseline (`best/` directory). On accept: all session `.py` files (except `__init__.py`) are snapshotted to `best/` and `bestIteration` is set in the API. On reject: session `.py` files are restored from `best/` (new files added in the rejected iteration are removed). Both outcomes are recorded in iteration metadata (`accepted: true/false`, `returned_to: N`).

**Cross-session memory:** `gnomepy_research/research_learnings.md` is appended when a session completes or stalls. `/research` Step 2 reads this file before forming each hypothesis — avoids re-running dead ends and bootstraps from known-good approaches. Learnings are also pushed as session notes (`poetry run research notes add`) for web UI visibility.

**Diagnostics:** `/research` Step 5 runs mandatory per-strategy-type diagnostic checks (fill rate, leg imbalance, fee drag, etc.) every iteration. Results are included in the iteration analysis and directly inform the next hypothesis.

### Exchange profiles
Saved exchange profiles live in `gnomepy_research/profiles/`. Current profiles: `hyperliquid`, `lighter`, `polymarket`, `kalshi`. The `/research-new` wizard picks these up automatically so you don't re-specify fees and latency each time. New profiles created during `/research-new` are saved here for future reuse.

### Iteration modes
- **Local run**: for logic changes — writes a config YAML, runs via `poetry run gnomepy backtest run --config <path>`
- **Remote sweep**: for parameter search — commits+pushes the session branch, submits to AWS Batch via `gnomepy backtest submit --research-commit <sha>`

## Code conventions
- All imports at the top of the file — never inside functions or conditionals
- No comments unless the WHY is non-obvious
- Strategies subclass `gnomepy.Strategy`
- Signals compose from `gnomepy_research.signals`
- Prices are scaled integers (divide by 1e9 for display) — never pass floats to `Intent`
- Return `[]` from `on_execution_report` unless reactive logic is required
