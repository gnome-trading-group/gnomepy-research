# structure: in-play CS2 map-win model (competition 3)

Direction: model structure and calibration, keeping the benchmark's inputs (plus `tier`).

## The entry

`entry.py` runs `core.py` with `variants.VARIANTS["slow-all"]`. Every scored configuration is a named `Config` in
`variants.py`; research jobs ran each through a two-line `entry_<name>.py` selecting it (removed after scoring; see
git history). Price at a state keeps the benchmark's decomposition

    P(A wins map) = P(A wins round | state) * M(score if A wins) + (1 - P(A wins round | state)) * M(score if A loses)

but replaces the map value M:

1. **Round-win chain (dp.py).** A round-start model q(score state) = P(A wins the next round) from rounds still needed,
   side, overtime, round in half, pistol, map, prior (market map price or Elo), tier and a signed win streak (the
   economy proxy: >0 means A won the last k rounds, reset at each money reset). It is learnt from every round's outcome
   (~750k labels instead of ~40k map labels), then rolled forward by dynamic programming over (score, streak) to
   P(A wins the map) from any score, overtime included (up to 4 OT blocks). The score is a q feature, so in-map form
   (a team beating its prior) carries into the remaining rounds.
2. **One-step economy lookahead.** The round right after the current one is priced with q1, the same model plus the
   money and equipment carried out of the current round (its post-buy money and round-start equipment); the chain
   takes over from there.
3. **Smooth q.** The chain compounds q's errors over many rounds, so smoothness matters more than raw fit: a monotone
   constraint on the prior (stronger team never less likely to win a round), slow learning (lr 0.02, 1000 trees).
4. **Blend.** M = 0.7 * chain + 0.3 * a regression of the map outcome on the next round's score state plus carried
   economy (the benchmark's map model with carry features and min_child_weight 200).
5. **Round model** = the benchmark's, plus the prior oriented to the CT side (monotone), seconds since the last kill,
   which side died last / first, and the bomb site.

## Development result (5 months, 2026-04..08, harness on AWS)

**structure-slow-all: diff -0.00519, 95% interval [-0.00651, -0.00382], better in 5/5 months**
(04 -0.00526, 05 -0.00666, 06 -0.00337, 07 -0.00220, 08 -0.00670); round-start part -0.00483; audit passed
(100 states, max |diff| 0). Fit noise (harness --seeds 3): -0.00519, -0.00518, -0.00504, so differences below ~0.00015
between configurations are noise. Runtime ~11-15 min on 8 vCPU.

Where the gain comes from: almost all of it is the map value (round-start part -0.0048 of -0.0052), and it is spread
evenly over the round (diff by events so far is flat), largest in rounds 7-24 and in tier s/a/b maps.

## Every configuration scored (full development unless noted; diff = entry minus benchmark log loss)

| configuration | diff | 95% interval | months better | what changed |
|---|---|---|---|---|
| dpblend-v1 (quick, Aug only) | -0.00394 | [-0.00601, -0.00189] | 1/1 | chain (q on score+streak) 0.7 + regression with carry 0.3 |
| dpblend-v1 | -0.00411 | [-0.00514, -0.00314] | 5/5 | as above, full development |
| look | -0.00445 | [-0.00547, -0.00348] | 5/5 | + next-round lookahead q1 with carried money/equipment |
| look-w1 | -0.00452 | [-0.00582, -0.00324] | 5/5 | chain only (no regression) |
| look-eco | -0.00444 | [-0.00547, -0.00349] | 5/5 | round model + money left, loss counters, OT, streak |
| look-ctq | -0.00444 | [-0.00546, -0.00349] | 5/5 | round model + q's round-start probability |
| look-eco-ctq | -0.00441 | [-0.00546, -0.00345] | 5/5 | both |
| look-r1000 | -0.00447 | [-0.00549, -0.00351] | 5/5 | round model 1000 trees lr 0.03 |
| look-tier | -0.00453 | [-0.00562, -0.00350] | 5/5 | + tier in q and q1 |
| look-tierbo | -0.00458 | [-0.00571, -0.00345] | 5/5 | + tier, bo, map number |
| look-tier-map | -0.00455 | [-0.00570, -0.00345] | 5/5 | + tier in the regression too |
| look-tier-avg5 | -0.00457 | [-0.00565, -0.00354] | 5/5 | q averaged over 5 seeds |
| look-tier-rtier | -0.00451 | [-0.00561, -0.00348] | 5/5 | + tier in the round model |
| look-tier-w1 | -0.00456 | [-0.00598, -0.00322] | 5/5 | chain only |
| look-tier-rs3 | -0.00455 | [-0.00564, -0.00352] | 5/5 | round model averaged over 3 seeds |
| look-tier-half | -0.00436 | [-0.00543, -0.00334] | 5/5 | chain state + A's first-half score (13 DP copies) |
| look-tier-mono | -0.00470 | [-0.00589, -0.00349] | 5/5 | q monotone in prior (+), need_a (-), need_b (+) |
| look-tier-mono-half | -0.00475 | [-0.00595, -0.00357] | 5/5 | both |
| look-tier-monop | -0.00489 | [-0.00610, -0.00368] | 5/5 | q monotone in prior only |
| monop-avg5 | -0.00492 | [-0.00613, -0.00372] | 5/5 | + 5 q seeds |
| monop-q1 | -0.00489 | [-0.00609, -0.00367] | 5/5 | + q1 monotone in prior |
| monop-slow | -0.00508 | [-0.00628, -0.00387] | 5/5 | q lr 0.02, 1000 trees |
| monop-w1 | -0.00490 | [-0.00642, -0.00331] | 5/5 | chain only |
| monop-all | -0.00500 | [-0.00634, -0.00361] | 5/5 | prior monotone in q, q1, regression, round model |
| monop-d4 | -0.00484 | [-0.00601, -0.00364] | 5/5 | q depth 4 |
| **slow-all** | **-0.00519** | **[-0.00651, -0.00382]** | **5/5** | monop-all + slow q (the entry) |
| slow-all-s3 | -0.00519 | [-0.00651, -0.00382] | 5/5 | same, 3 seeds: -0.00519 / -0.00518 / -0.00504 |
| slower-all | -0.00510 | [-0.00642, -0.00376] | 5/5 | q lr 0.01, 2000 trees |
| slow-all-q1slow | -0.00520 | [-0.00651, -0.00382] | 5/5 | q1 slow too (same within noise) |
| slow-all-mcw300 | -0.00508 | [-0.00637, -0.00374] | 5/5 | q min_child_weight 300 |
| slow-all-ic | -0.00483 | [-0.00611, -0.00354] | 5/5 | q interaction constraints (prior group / state group) |
| slow-all-ens3 | -0.00509 | [-0.00639, -0.00375] | 5/5 | chain averaged over 4 q structures |
| slow-all-ens-ic | -0.00505 | [-0.00633, -0.00374] | 5/5 | chain averaged over slow + interaction-constrained q |
| slow-all-rmono | -0.00518 | [-0.00651, -0.00381] | 5/5 | round model monotone in alive/equipment/kits/plant |
| final | -0.00519 | [-0.00651, -0.00382] | 5/5 | entry.py as handed in (slow-all), audit passed |

Exploration outside the harness (walk-forward on the same months with my own feature cache in `.scratch`, compared to
the cached benchmark; not logged): benchmark reproduced exactly (diff 0.00000 on Aug); round model + CT prior
-0.00035 (Aug); regression map model + carry -0.0020 (Aug), min_child_weight 200 better, depth 7 worse; recency
weighting of q (half-life 180 days) mixed by month, no gain; state-conditional next-round model (P(A wins next round |
this state, this round's result), to see survivors' economy) -0.00004; latent-strength Bayesian chain (q without score
+ Gauss-Hermite strength offset, posterior from past rounds, sigma by marginal likelihood) worse alone (-0.0034 vs
-0.0040) and no help in a blend; earlier maps of the series (map wins, round difference) in q worse; Elo min_games in q
no gain; round model weighted by the map-value swing |M1 - M0| no gain; stacking a map-outcome model on walk-forward
out-of-sample components (logistic or shallow trees with the price as base margin) badly worse (+0.002 to +0.014);
Platt / temperature per phase and blend weight by half/overtime/logit blend all within +-0.0001.

## What worked, what did not

Worked: replacing the map regression with a chain of round probabilities (the largest step, -0.002 to -0.003 over a
regression with the same inputs), the one-step economy lookahead (-0.0003), tier (-0.0001), a monotone prior in q
(-0.0003) and slow learning (-0.0002). The chain is sensitive to q's smoothness: deeper q (depth 6) loses 0.0005-0.001,
because the DP compounds q's small errors over the remaining rounds.

Did not work: anything on the round model (extra economy/score/sequence features, seeds, more trees, monotone
constraints: all within noise; the in-round part of the gain is small), stacking or calibrating on map outcomes (too few
independent maps; month-to-month drift), halftime state in the chain, Bayesian latent strength, ensembles of q
structures, series context, recency weights.

Calibration of the final price is close to identity (slope 1.01, intercept about -0.025 in every phase); the
intercept is a drift in team A's listing advantage that does not look stable enough to correct.

## Point-in-time

* Every per-state feature is a cumulative quantity over the state's own round up to its event_seq (`feats.event_features`:
  cumulative sums/forward fills grouped by game and round, never a whole-round aggregate), or round-start information
  of its own round (equipment and money after the buy, side, score) and earlier rounds (streak and loss counters from
  the score progression, carry from the previous row via shift).
* The chain's statics come from the map's first round row (side at round 1, map, prior, tier); the prior is the
  kickoff market price or the point-in-time Elo.
* Nothing is normalised over the batch: the DP is computed per map, each state's price only reads its own map.
* Training uses only `train` (maps that ended before the month), outcomes from `rounds.a_wins`/`map_a_wins` there.
* The harness audit (100 states re-priced on truncated bundles) passes with max |diff| = 0 for every configuration.
