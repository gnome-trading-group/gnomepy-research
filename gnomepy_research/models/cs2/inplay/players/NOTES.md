# players: player and team skill in the in-play map model

Direction: point-in-time player and team strength from history, used in the round model (who is alive, not just how
many) and in the map model's pre-map prior.

## Files

- `skill.py`: per-map player counts (rounds, kills, deaths, opening duels, clutches, AWP use; from the event timeline
  where there is one, else the end-of-round player columns), per-map team counts (rounds won by side and map), as-of
  lookups (`merge_asof`, strictly before the target map's begin), `PlayerElo` (online player ratings: overall, per map,
  and how often each player has played the map), `PlusMinus` (ridge adjusted plus-minus refitted each month start).
- `entry.py`: the entry (configuration F2 as defaults in `CFG`). Benchmark round + map models plus
  - pre-map prior, team A minus team B over the roster of the map's first round with player rows: mean/max/min net
    kill rating (kills minus deaths per round, shrunk 40 rounds to the global rate), opening-duel win rate, clutch
    rate, kills and deaths per round, experience (log rounds), roster stability (log maps each player has played for
    this team, mean and min), overall round win rate, round win rate on this map, player Elo (overall and per map), how
    often the roster plays this map; plus Elo. A logistic regression on these, fitted on every finished training map,
    replaces Elo where the map has no kickoff market. Where it does, the market, e12 and the map-specific roster terms
    (per-map Elo, map share, team map rate) are stacked by a second logistic regression once 300+ market-listed maps
    exist (the market price is a series price turned into a map price: it cannot know which map is played or whose
    pick it is);
  - map model: benchmark features without `has_prior` (the prior is calibrated either way), monotone in rounds needed
    and in the prior, depth 6 / min_child_weight 300 / learning rate 0.05 (more capacity only helps once monotone);
  - round model: unchanged from the benchmark in the final entry. `round_feats=True` adds the strength of the players
    still alive on each side (sum and max of net rating, opening rate, clutch rate of a last player alive, best player
    dead, AWPer dead, experience, each team's side record on the map); it improves round log loss slightly but not the
    map price, so it is off.
- `dp.py`: map value from the score under a constant per-round probability matched to the prior (tried as a map model
  feature; off).
- `lab.py`: fast walk-forward lab. Builds every legal map's rows once and refits only the trees per month. Its `base`
  reproduces the harness benchmark exactly on August (0.46560) and its full-dev numbers match the harness within
  ~0.0002.
- `variants/*.py`: one file per scored configuration (each sets `CFG`).

## Point-in-time

- Every count and rating a map sees comes from maps that ended strictly before it began: running totals stamped at
  `max(end, begin)` (a few maps have an end a few seconds before their begin), looked up with
  `merge_asof(allow_exact_matches=False)`; player Elo applies updates at the same stamp and answers lookups at the
  map's begin, lookups first on ties.
- Scored-month maps' outcome columns are blank, so their counts come from their event timelines (never blanked); a map
  of the scored month that ended before this one began therefore counts, exactly as the audit's truncated bundle has it.
- The roster used for a map is that of its first round with player rows; a scored state's own round always has player
  rows, so this round is never after the state.
- Inside the map, the round model uses only deaths with `seq <= event_seq` (as the benchmark does).
- Fitted pieces (priors for shrinkage, LRs, trees, player Elo state at the end of training) are frozen in the model; at
  pricing time ratings are carried forward only through ctx maps that have ended.
- Audit passed (100 states, max diff 0) on every harness run.

### A leak found and fixed

The first player-Elo version sorted (time, kind) tuples of pandas Timestamps; 42 maps have no begin time (NaT), and
NaT comparisons are always False, which silently scrambled the order so that maps saw their own and later results.
Symptom: a pre-map log loss of 0.556 (Elo 0.654, market 0.633), insensitive to delaying every update by 30 or 180 days
and still "working" with the map names shuffled. After sorting integer nanoseconds and leaving NaT maps unrated, the
lag test behaves (hit rate 57.4% → 57.1% → 54.6% at 0/30/180 days) and player Elo is only a small addition. Lab rows
4 (`pelo:*`) below were produced with the bug and are void.

## What worked and what did not

Almost all of the gain is between rounds (round-start difference ≈ overall difference): it comes from a better pre-map
prior and a map model that uses it better. Player skill inside the round did not move the map price.

Worked (harness, full development, entry minus benchmark):
1. Roster-strength prior instead of Elo on maps without a kickoff market (~25% of scored maps): -0.0031. Pre-map log
   loss on those maps 0.654 (Elo) → 0.630. Experience (log rounds played) is the strongest single roster term:
   established players beat newcomers far more than Elo's shrinkage expects.
2. Monotone map model without the has-prior flag: -0.0042 with the e12 stack. Then more capacity (depth 6, min child
   weight 300, learning rate 0.05): -0.0052. More capacity without the constraints is much worse (+0.0048 on the
   benchmark prior in the lab): the per-round rows share one label per map and the trees fit map noise.
3. Market maps: stacking the kickoff market with e12 and the map-specific roster terms (per-map player Elo, how often
   each roster plays the map, team round rate on the map); the market is a series price and cannot know which map is
   played. Pre-map on market maps 0.6343 (market) → 0.6300 (market + e12) → 0.6257 (+ map terms). With roster
   stability and team map rate in the non-market prior: F2, -0.0062.
4. Adjusted plus-minus (ridge on rosters, refitted each month start) in the roster prior: F3, -0.0068.

Did not help (lab unless noted; lab matches the harness within ~0.0002):
- Alive-player strength in the round model (`round_feats`): round log loss -0.0003 (0.38489 → 0.38461), map price
  ±0.0001 (harness: F_rskill vs F +0.0001, F2_rskill vs F2 +0.0001). Individual skill barely moves duels: 1v1s go
  59% → 62% CT across quintiles of the skill gap, while the team gap at round start moves rounds 46% → 59%, and that
  part is already in the map model's prior.
- Team strength (the prior, oriented CT vs T) in the round model: round log loss -0.001, map price +0.0002.
- Raw roster/team features in the map model instead of a combined prior: +0.0011 (trees overfit them).
- Side-specific team strength on the map (CT/T round rates) in the map model: +0.0002.
- In-map kill differential so far (A kills minus B kills per player-round over completed rounds): +0.0010.
- A constant-round-probability score DP as a map model feature: +0.0003.
- Tier, BO and decider flags in the map model: +0.0000 / +0.0004.
- Map model bagging (3 seeds): -0.00003. Round model capacity (1000 trees; depth 8): +0.0000.
- Time-decayed counts (half lives 180/365 days), other shrinkage strengths, other player-Elo K: within ±0.0005 pre-map,
  worse or equal.
- Per-map plus-minus, plus-minus on map results (logistic) instead of round share, and plus-minus inside the market
  stack: no pre-map gain over the ridge round-share version.
- Roster experience as a map model feature (so the trees could learn how far to trust the prior): +0.0042 (lab, vs F3).
- Training the map model only on maps from 2025-06 on (when ratings have history behind them): +0.0000.
- A final logistic recalibration: calibration slopes are 0.91–1.02 by month, nothing systematic to fix.

## Results

Every configuration scored with the harness (full development 2026-04..08 unless noted; entry minus benchmark log
loss, map-bootstrap 95% interval, per-month differences; audit 100 states, all passed with max diff 0). All are in
`results/log.jsonl` under these names.

| name | what | diff | 95% interval | months better | Apr / May / Jun / Jul / Aug |
|---|---|---|---|---|---|
| players-v1 | first try, Aug only: raw roster/team features in the map model + alive skill in the round model | +0.00208 | [-0.0036, +0.0080] | 0/1 | Aug +0.0021 |
| players-prior_only | roster LR prior replaces Elo on non-market maps | -0.00311 | [-0.0047, -0.0014] | 5/5 | -0.0038 -0.0032 -0.0016 -0.0044 -0.0021 |
| players-prior_rskill | + alive-player strength in the round model | -0.00324 | [-0.0049, -0.0015] | 5/5 | -0.0043 -0.0034 -0.0016 -0.0041 -0.0023 |
| players-all | + team prior in the round model | -0.00314 | [-0.0048, -0.0014] | 5/5 | -0.0039 -0.0034 -0.0016 -0.0039 -0.0023 |
| players-elo_mono | benchmark prior, monotone map model only | -0.00110 | [-0.0021, -0.0002] | 4/5 | -0.0016 -0.0003 -0.0010 +0.0002 -0.0028 |
| players-base_mez_nohas_mono | roster prior (no player Elo), market+e12+z stack, no has_prior, monotone | -0.00380 | [-0.0061, -0.0015] | 5/5 | -0.0042 -0.0046 -0.0019 -0.0043 -0.0032 |
| players-full_mez_nohas_mono | + player Elo, per-map Elo, map share in the roster prior | -0.00421 | [-0.0067, -0.0019] | 5/5 | -0.0045 -0.0059 -0.0028 -0.0035 -0.0032 |
| players-full_mez_nohas_mono_rskill | + alive-player strength | -0.00413 | [-0.0067, -0.0017] | 5/5 | -0.0048 -0.0058 -0.0026 -0.0029 -0.0030 |
| players-F | full_mez_nohas_mono + map model depth 6 / mcw 300 / lr 0.05 | -0.00515 | [-0.0077, -0.0027] | 5/5 | -0.0058 -0.0079 -0.0045 -0.0031 -0.0025 |
| players-F_rskill | + alive-player strength | -0.00504 | [-0.0078, -0.0025] | 5/5 | -0.0062 -0.0078 -0.0043 -0.0025 -0.0023 |
| players-F2 | F + roster stability, team map rate; market stacked with e12 + map-specific terms | -0.00623 | [-0.0090, -0.0037] | 5/5 | -0.0057 -0.0111 -0.0022 -0.0063 -0.0030 |
| players-F2_rskill | + alive-player strength | -0.00616 | [-0.0090, -0.0036] | 5/5 | -0.0061 -0.0110 -0.0020 -0.0058 -0.0028 |
| players-F2_seeds2 | F2, two seeds (fit noise) | -0.00623 | [-0.0090, -0.0037] | 5/5 | seeds: -0.00623, -0.00637 |
| **players-F3** | **F2 + adjusted plus-minus in the roster prior (the entry)** | **-0.00680** | **[-0.0095, -0.0043]** | **5/5** | -0.0068 -0.0113 -0.0031 -0.0073 -0.0029 |
FINAL_ROW

Lab-only comparisons (approximate harness, same five months) are listed under "Did not help" above; lab runs `lab1`–`lab4`
predate the NaT fix and their player-Elo numbers are void.

Fit noise: two seeds of F2 differ by 0.00014, far inside the map-bootstrap interval (±0.0027).

## Why the gain is believable

- ~0.007 is below the brief's 0.01 alarm, and its sources are separately measurable before the map starts: on maps
  without a market the pre-map log loss goes 0.654 (Elo) → 0.624; on market maps 0.634 (market) → 0.626. These are
  ordinary sizes for "roster beats team Elo" and "the series market does not know the map".
- The one leak found (NaT ordering) was caught by exactly these checks: delaying every rating update by 30/180 days
  must cost something, and shuffling map names must kill map-specific terms. Both now behave.
- The harness audit (re-pricing on bundles truncated to each state) passes with max diff 0 on every run.
