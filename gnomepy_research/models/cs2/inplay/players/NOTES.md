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

## Results

(see the table at the end; filled in as configurations are scored)
