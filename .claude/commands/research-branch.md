# Branch a Research Session

Fork an existing research session into a new parallel exploration. Arguments: **$ARGUMENTS**

Expected format: `<parent_session> <branch_suffix>` — e.g., `/research-branch n_exchange_arb spread_based`

The new session will be named `<parent>__<suffix>` (double underscore). It gets its own git worktree so it can run in a separate terminal via `/loop /research <parent>__<suffix>` without conflicting with the parent.

---

## Steps

### 1. Parse arguments
Parse `$ARGUMENTS` as `<parent_session> <branch_suffix>`. If either part is missing, use `AskUserQuestion` to ask for the missing value(s) before continuing.

Validate:
- Parent session directory `gnomepy_research/sessions/<parent>/` exists and has a `spec.yaml`
- Branch suffix is a valid Python module identifier (lowercase, digits, underscores only; no hyphens or spaces; does not start with a digit)
- Combined name `<parent>__<suffix>` does not already exist as a session directory

If any validation fails, report the specific error and stop.

---

### 2. One round of questions
Use `AskUserQuestion` with exactly these 3 questions in a single call:

1. **header: "Branch focus"** — What approach will this branch explore differently? Free-text via Other is expected. Include placeholder examples as options:
   - "Use EWMA z-score spread entry instead of raw bps threshold"
   - "Switch to maker-only passive orders to reduce fees"
   - "Try a cointegrated pairs approach with Kalman-filtered spread"

2. **header: "Copy strategy?"** — Start from the parent's current strategy.py?
   - `Yes — copy parent strategy as starting point` *(Recommended if parent has run at least one iteration)*
   - `No — start blank (iteration 1 writes from scratch)`

3. **header: "Override spec?"** — Any spec values need changing?
   - `No — use parent spec as-is` *(Recommended)*
   - `Yes — I'll edit spec.yaml after creation`

---

### 3. Copy and modify spec.yaml
Read `gnomepy_research/sessions/<parent>/spec.yaml`. Create a modified version:

- Change `name:` to `<parent>__<suffix>`
- Prepend the branch focus to the description, preserving the original:

```yaml
description: >
  [BRANCH: <suffix>] <branch_focus_text>

  Original: <original description text>
```

All other fields (data, profiles, goals, thresholds, targets, constraints, meta) are copied verbatim.

---

### 4. Create session directory structure
```bash
mkdir -p gnomepy_research/sessions/<parent>__<suffix>/configs
mkdir -p gnomepy_research/sessions/<parent>__<suffix>/results
touch gnomepy_research/sessions/<parent>__<suffix>/__init__.py
```

Write the modified spec.yaml to `gnomepy_research/sessions/<parent>__<suffix>/spec.yaml`.

If "Yes — copy parent strategy" was selected, copy all session `.py` files (except `__init__.py`) and the `best/` directory if it exists:
```bash
for f in gnomepy_research/sessions/<parent>/*.py; do
  [ "$(basename "$f")" != "__init__.py" ] && cp "$f" gnomepy_research/sessions/<parent>__<suffix>/
done
if [ -d gnomepy_research/sessions/<parent>/best ]; then
  cp -r gnomepy_research/sessions/<parent>/best gnomepy_research/sessions/<parent>__<suffix>/best
fi
```

---

### 5. Create git branch
Create a new branch from the parent's research branch:

```bash
# Check if parent branch exists
git show-ref --verify --quiet refs/heads/research/<parent> && \
  git branch research/<parent>__<suffix> research/<parent> || \
  git branch research/<parent>__<suffix> HEAD
```

This branches from the parent's current code (including any committed strategy work), or from HEAD if the parent has never been iterated.

---

### 6. Create git worktree
Create a separate working directory for this branch so it can run in parallel with the parent and other branches:

```bash
WORKTREE_PATH="../gnomepy-research--<parent>__<suffix>"
git worktree add "$WORKTREE_PATH" research/<parent>__<suffix>
```

The worktree path uses double-dash before the session name so it's visually distinct from the main repo.

If the worktree add fails (e.g., worktree already exists), report the error and continue — the session can still be used, just not in parallel with others on the same branch.

---

### 7. Register in the API
```bash
poetry run research sessions create <parent>__<suffix> \
  --spec gnomepy_research/sessions/<parent>__<suffix>/spec.yaml \
  --branch research/<parent>__<suffix> \
  --tags "branch,parent:<parent>"
```

If the session already exists in the API, the command exits cleanly.

---

### 8. Output instructions
Tell the user:

- Session `<parent>__<suffix>` is ready at `gnomepy_research/sessions/<parent>__<suffix>/`
- Git branch `research/<parent>__<suffix>` created from `research/<parent>`
- Worktree created at `<WORKTREE_PATH>`

Provide the exact commands to start iterating in the new worktree:
```
cd <WORKTREE_PATH>
poetry install
claude
# then inside claude:
/loop /research <parent>__<suffix>
```

If "Yes — I'll edit spec.yaml" was selected, remind them to edit `gnomepy_research/sessions/<parent>__<suffix>/spec.yaml` in the worktree before starting.

---

## Notes

- The parent session continues running on its own branch — this branch is fully independent
- Use `/research-status` to monitor all branches and their metrics at a glance
- Use `/research-hint <parent>__<suffix>` to steer the branch's autonomous loop
- Each branch has its own iteration history in the API, tagged with `parent:<parent>` for grouping
