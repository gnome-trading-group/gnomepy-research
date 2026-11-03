# CFB early-week paper sessions: pre-registration

Written Nov 2, 2026, before any session ran. Paper only; not financial advice.

## Why paper only

- **The signal is real but small.** The model's gap to Kalshi's mid 72 hours out predicts the move to kickoff:
  β = 0.116 (t 6.5) in 2025 and 0.065 (t 2.2) in 2026 (`~/cfb-live-data/early-week-study/SPEC.md`, A2).
- **No version cleared costs in the study.** A taker round trip lost about 4¢ per contract. The resting version (A2b)
  lost 0.13¢ (2025) and 0.38¢ (2026) per game under realistic fills, and won 1.6¢ and 1.5¢ under optimistic fills.
- **The gap between those two is queue position.** Candles and trades cannot show where our order would have sat.

## What runs

- **Strategy:** `early_week.py:CFBEarlyWeek`, the A2b rules:
  - decide at 72 h, up to 6 h late;
  - trade when |model − mid| ≥ 0.05;
  - one contract on the model's side;
  - a post-only bid at the best bid, re-pegged every 6 h;
  - cancel or exit from 6 h out, with a post-only ask at the best ask, re-pegged hourly;
  - sell into the bid 5 min before kickoff.
- **Mid:** the mean of the two legs' mids in home terms, a leg counting only when two-sided and ≤ 6¢ wide (as in A2).
- **Model:** e11 (C2 + one PFF rating). It ties the e11 + e15 overlay over 2019–2026: −0.0000 per game, t 0.07,
  n 5,411. p_home is the live pipeline's weekly value, frozen into the launch trigger.
- **Venue:** Kalshi only.
- **Universe:** FBS v FBS games in the weekly slate with at least 66 h to kickoff at launch, matched and cross-checked
  as `launch.py` does.

## What paper can and cannot answer

Paper fills come from the registry's Kalshi simulation parameters (its queue model), not from a real queue. So these
sessions test:
- the rules running end to end;
- the forward signal (β on weeks 10+, with predictions truly frozen at E, the forward check A2 asked for);
- P&L under the simulator's fills.

They do **not** settle the realistic-vs-optimistic question. Only real orders can do that. Read P&L next to the queue
model the registry used.

## Endpoints, per week and pooled

1. **β:** Δ = logit(K) − logit(E) on g = logit(model) − logit(E).
   - E is the decision row's market mid. K is Kalshi's twin mid 5 min before kickoff.
   - It uses every decision row, traded or not.
2. **P&L per decided game and per traded game,** with the entry fill rate, exit fill rate and share of taker exits.
   These are set against the A2b simulation of the same games, realistic and optimistic.
3. **Adverse selection:** the mid 1 h and 6 h after each entry fill.

Nothing here is promoted to real money from these sessions alone.

## Amendment, Nov 2, 2026 (the user's call, before any session ran): late games traded off protocol

Week 10's Tuesday–Friday games, and any game launched with fewer than 66 h to kickoff, can be traded with
`launch_early_week --include-late`. Such a game:
- is marked `off_protocol` in its trigger;
- is decided at its first usable look, however late, until its entry closes 6 h before kickoff;
- has a decision row with `off_protocol = 1` and `hours_before`, the hours to kickoff when it was decided.

Games with 7 h or less to kickoff at launch are still left out.

**Why it breaks the strategy:** the market has had more time to take in what the model knows, and the bid has less
time to fill. The 72-hour signal was never measured at these horizons; A2 tested only −72, −48 and −24 h.

**How it is reported:**
- Off-protocol games are reported apart.
- The endpoints above use only the protocol games (`off_protocol = 0`).
- Pooled numbers, if shown, are labelled as including off-protocol games.
