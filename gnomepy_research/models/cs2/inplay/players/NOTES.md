# players: player and team skill in the in-play map model

Direction: point-in-time player and team strength from history, used in the round model (who is alive, not just how
many) and in the map model's pre-map prior.

## Files

- `skill.py`: per-map player counts (rounds, kills, deaths, opening duels, clutches, AWP use; from the event timeline
  where there is one, else the end-of-round player columns), per-map team counts (rounds won by side and map), as-of
  lookups (`merge_asof`, strictly before the target map's begin), and `PlayerElo` (online player ratings: overall, per
  map, and how often each player has played the map).
- `entry.py`: the entry. Benchmark round + map models plus
  - round model: strength of the players still alive on each side (sum and max of shrunk net kill rating, opening-duel
    rate, the clutch rate of a last player alive, whether the best player / the AWPer is dead, experience) and each
    team's side record on the map (`round_feats`);
  - pre-map prior: a logistic regression on Elo and roster strength (team A minus team B: mean/max/min net rating,
    opening, clutch, kills, deaths, overall round win rate, experience; optionally player Elo, per-map player Elo and
    map share), fitted on every finished training map. Replaces Elo where the map has no kickoff market; where it does,
    the market, e12 and this prior are stacked by a second logistic regression once 300+ market-listed maps exist;
  - map model: benchmark trees, without the `has_prior` flag (the prior is calibrated either way) and monotone in
    rounds needed and in the prior.
- `lab.py`: fast walk-forward lab. Builds every legal map's rows once and refits only the trees per month. Its `base`
  reproduces the harness benchmark exactly on August (0.46560) and its numbers matched the harness within ~0.0002.
- `variants/*.py`: one file per scored configuration (sets `CFG`).

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

## Results

(see the table at the end; filled in as configurations are scored)
