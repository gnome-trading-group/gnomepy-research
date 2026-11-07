# priors: pre-map prior and map context

Direction: improve the pre-map prior the benchmark's map model starts from (kickoff market price where listed, the
provided Elo otherwise), and how it carries through the map.

## What the entry does

`entry.py` = `core.entry(Config(...))` (the best configuration; see the table). Same round model and map model as the
benchmark, with these changes:

1. **Market-listed maps** (about three quarters of scored maps, and of states): the prior is the average of the logits
   of the kickoff market and e12 single-map prices, `(logit p_market + logit p_e12) / 2`. Map level (Apr–Aug, market
   maps): market 0.6306, e12 0.6227, average 0.6213. Fitted logistic combinations (market, e12, pool, rating prior,
   series score so far, team-on-map ratings) never beat the plain average, and it needs no training rows, so it also
   covers April, when no market-listed map is in the training history yet.
2. **Other maps**: a logistic model (`NM_SETS["p2"]`) on point-in-time ratings computed in `ratings.py`:
   - the provided Elo;
   - result Elo for teams, teams on this map, and players (team = roster mean, roster from the map's first round
     with player data);
   - round-share Elo (update by rounds won share, not just the result) for the same three: the strongest addition;
   - each rating times log(1 + the roster's least experienced player's map count), because player ratings of
     newcomers are noise;
   - days since each team's last map; BO1 flag; tier dummies.
   Map level on non-market maps (Apr–Aug): provided Elo alone 0.6459, p2 0.6297.
3. **Map model**: monotone constraints (prior_logit +, need_a −, need_b +).
4. **Round model** (optional `edge_round`): the prior logit oriented to the CT side as one extra feature, so the round
   model knows which side is stronger.

## Point-in-time handling

- `ratings.run` replays maps in time order: each map is *queried* at its `begin` and its result *applied* at its `end`;
  at equal times queries go first, so a map only sees maps with `end < begin` (the harness's rule, strict).
- Results for maps in the scored month come from the final score implied by the last round's score state (the
  outcome columns are blanked); only maps that ended before a map began are ever applied before its query.
- Maps without `begin` are queried at the month start (before any month result), matching the audit's visible set
  (only training maps); maps without `end` never update anything.
- `predict` continues from a copy of the training-time rating state, so calls never affect each other, and nothing
  is fitted on the scored month. Series features (unused in the final prior) take only earlier maps of the match that
  ended before this map began.
- The prior models are fitted in `fit` on the training history only; the xgb variant uses out-of-fold (by match)
  priors for the map model's training rows.
- Audit passes with max |diff| 0.0 on every scored configuration.
- `p_e12`/`p_pool` are taken as provided in the bundle's `priors` table (kickoff prices of the series).

## Results

Fast loop (`exp.py`): fits the round model once per month (cached) and refits only prior + map model, paired against
the benchmark map model on the same round model. Diff = entry minus benchmark log loss per state, pooled Apr–Aug.

RESULTS_TABLE

## What did not work

DID_NOT_WORK
