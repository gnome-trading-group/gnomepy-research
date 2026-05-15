# gnomepy-research

## Commands

```bash
poetry install          # install dev environment
poetry run pytest       # run all tests
```

## Strategy Research Sessions

Research sessions live in `gnomepy_research/sessions/<name>/`. Start a new session with `/research-new <name>`, fill in `spec.yaml`, then run `/research <name>` (or `/loop /research <name>` for continuous iteration).

Each session runs on its own git branch `research/<name>`, created automatically on the first iteration.

### Session structure
```
gnomepy_research/sessions/<name>/
  __init__.py       # empty — makes the session importable as a Python module
  spec.yaml         # user-authored goals and constraints — DO NOT MODIFY
  session.json      # iteration history, Claude-managed
  strategy.py       # single strategy file, modified in place each iteration
  configs/          # per-iteration backtest YAML configs and sweep configs
  results/          # backtest outputs (per-iteration subdirectories)
```

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
