# Clean Up a Research Session

Tear down the worktree, branch and local artifacts for a finished session: **$ARGUMENTS**

`/research-branch` creates a git worktree per branch session and nothing has ever removed them, so
they accumulate along with tens of gigabytes of `results/`.

---

## Step 1: Confirm the session is finished

```bash
poetry run research sessions get $ARGUMENTS
```

Read `status` and `best_iteration` (the API returns snake_case). If `status` is `running`, stop and
ask the user whether to clean it up anyway — a running session may have a loop attached.

Report what will be removed before removing anything.

---

## Step 2: Check for uncommitted work

If the session has a worktree, check it is clean before touching it:

```bash
git worktree list --porcelain
git -C <worktree_path> status --short
```

**If anything is uncommitted or untracked, stop and show the user.** Never discard work that was
never committed — ask whether to commit it to `research/$ARGUMENTS` first.

---

## Step 3: Confirm scope with the user

Use `AskUserQuestion` with one multi-select question, "What should I clean up?":

- `Worktree` — remove the worktree directory (the branch and its commits survive)
- `Results` — delete `gnomepy_research/sessions/$ARGUMENTS/results/` (regenerable; often gigabytes)
- `Local branch` — delete `research/$ARGUMENTS` locally (only offer this if it is merged or pushed)

Never offer to delete the session directory, `spec.yaml`, `configs/`, `best/`, or the remote branch.
Those are the session's record.

Show the size of what will be deleted so the choice is informed:
```bash
du -sh gnomepy_research/sessions/$ARGUMENTS/results 2>/dev/null
```

---

## Step 4: Execute the confirmed actions

```bash
# Worktree
git worktree remove ../gnomepy-research--$ARGUMENTS
git worktree prune

# Results
rm -rf gnomepy_research/sessions/$ARGUMENTS/results

# Local branch — only if merged or pushed; use -d, never -D
git branch -d research/$ARGUMENTS
```

If `git branch -d` refuses because the branch is unmerged, report that and leave it alone. Do not
reach for `-D`.

---

## Step 5: Report

State what was removed, what was kept, and how much disk was reclaimed. If the session status is
still `running`, remind the user to set it:

```bash
poetry run research sessions update $ARGUMENTS --status completed   # or stalled
```

---

## Notes

- Session state, iteration history and notes live in the API and are untouched by any of this
- `best/` and `configs/` stay in git — they are how a session is reproduced later
- Run `/research-status` afterwards to confirm the session list looks right
