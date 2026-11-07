# structure: in-play CS2 map-win model (competition 3)

Direction: model structure and calibration, keeping the benchmark's inputs (plus `tier`).

## The entry

`entry.py` -> `core.py` with the configuration named in `entry.py` (all scored configurations are in `variants.py`; each
`entry_<name>.py` selects one). Price at a state keeps the benchmark's decomposition

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

TODO

## Every configuration scored (full development unless noted; diff = entry minus benchmark log loss)

TODO

## What did not work

TODO

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
