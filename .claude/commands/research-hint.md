# Add a Research Hint

Write a hint for the next iteration of a research session: **$ARGUMENTS**

Hints are picked up at the start of Step 2 (Hypothesize) on the next `/research` run and take priority over the session's planned `next_action`. Multiple hints can queue up — they are all consumed and cleared together at the next iteration.

## Steps

### 1. Resolve session name
If `$ARGUMENTS` is empty, use `AskUserQuestion` to ask for the session name before continuing.

### 2. Verify session exists
Check that `gnomepy_research/sessions/$ARGUMENTS/` exists. If not, tell the user the session was not found and list any sessions under `gnomepy_research/sessions/`.

### 3. Ask for the hint
Use `AskUserQuestion` with 1 question:

- **header: "Hint"** — "What direction should the next iteration take?" Use options to prompt common categories, but expect the actual idea via Other:
  - "Try a different signal" — Suggest a specific signal or signal combination to try
  - "Adjust parameters" — Change specific parameter values (e.g. wider spread, lower gamma)
  - "Structural change" — Rethink strategy structure, logic flow, or approach
  - Use Other to type the actual idea directly

### 4. Append to hints.md
Read `gnomepy_research/sessions/$ARGUMENTS/hints.md` if it exists (don't fail if missing). Append the new hint with a timestamp:

```
[2026-05-13T14:30:00] <hint text from user>
```

Write the full file back (existing hints + new hint).

### 5. Confirm
Tell the user: "Hint saved to `gnomepy_research/sessions/$ARGUMENTS/hints.md`. It will be picked up on the next `/research $ARGUMENTS` iteration."
