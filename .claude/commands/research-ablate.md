# Signal Ablation

Measure which signals actually earn their place in a session's strategy: **$ARGUMENTS**

A parameter sweep tells you the best value for a signal you already committed to. Ablation tells you
whether the signal should be there at all — turn each one off and see what the strategy loses.

---

## Step 1: Identify the baseline and the ablatable switches

```bash
poetry run research sessions get $ARGUMENTS
```

Use `best_iteration` to find the baseline config
`gnomepy_research/sessions/$ARGUMENTS/configs/iter_NNN.yaml` and its results directory.

Read `gnomepy_research/sessions/$ARGUMENTS/strategy.py` and list the constructor kwargs that switch a
signal or component off — a boolean flag, or a weight/coefficient that disables the term at 0.

**If the strategy has no such switches, stop.** Tell the user which signals it uses and that
ablation needs each one behind a flag (e.g. `use_flow_signal: bool = True`) or a zeroable weight.
Adding those flags is a normal iteration; do not edit the strategy from this command.

---

## Step 2: Generate the ablation configs

```bash
poetry run python3 - <<'EOF'
from gnomepy_research.analysis.signal_attribution import generate_ablation_configs

configs = generate_ablation_configs(
    "gnomepy_research/sessions/$ARGUMENTS/configs/iter_NNN.yaml",
    {"use_flow_signal": False, "use_depth_signal": False},   # one entry per switch
    "gnomepy_research/sessions/$ARGUMENTS/configs/ablation",
)
for c in configs:
    print(c)
EOF
```

---

## Step 3: Run each config

Run the baseline (or reuse `results/iter_NNN` if it is still there) and every ablation config, each
with its own `--output`:

```bash
poetry run gnomepy backtest run \
  --config gnomepy_research/sessions/$ARGUMENTS/configs/ablation/<name>.yaml \
  --output gnomepy_research/sessions/$ARGUMENTS/results/ablation/<name>
```

---

## Step 4: Summarize

```bash
poetry run python3 - <<'EOF'
import json
from pathlib import Path
from gnomepy_research.analysis.signal_attribution import summarize_ablation_results

root = Path("gnomepy_research/sessions/$ARGUMENTS/results")
baseline = json.loads((root / "iter_NNN" / "summary.json").read_text())
names = ["use_flow_signal", "use_depth_signal"]
ablated = [json.loads((root / "ablation" / n / "summary.json").read_text()) for n in names]

print(json.dumps(
    summarize_ablation_results(names, baseline, ablated, primary_metric="<spec.goals.primary_metric>"),
    indent=2, default=str,
))
EOF
```

Present a table of signal → baseline metric → ablated metric → delta, ordered by how much removing
the signal costs.

**Reading it:**
- Removing a signal **hurts a lot** → it is carrying the strategy; protect it
- Removing it **changes nothing** → it is dead weight; delete it and simplify
- Removing it **helps** → it is actively harmful on this data; delete it and say so

`sharpe` deltas are per-bar at 10s bars, not annualized (see `tutorials/02_research_workflow.md`).
A small window makes small deltas untrustworthy — say so rather than over-reading them.

---

## Step 5: Record

Ablation is analysis, not an iteration, so do not call `iterations record`. Record it as a note:

```bash
poetry run research notes add $ARGUMENTS "ABLATION (iter NNN): <signal> carries <X> of the primary metric; <signal> is dead weight; <signal> is net negative."
```

If a signal is dead weight or harmful, say in the note that the next iteration should remove it.
