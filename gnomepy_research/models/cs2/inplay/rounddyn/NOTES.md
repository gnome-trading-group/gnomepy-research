# rounddyn: richer in-round state (and where the gain actually came from)

## Entry (`entry.py`)

The benchmark's decomposition `P(A wins round) x M(score if A wins) + P(A loses round) x M(score if A loses)`, with:

1. **Round model** (xgboost, benchmark params): the benchmark's CT-vs-T state plus timeline dynamics from
   `feats.dyn_features` (rifles/SMGs/helmets/armor alive, best weapon class alive, weapons seen in kills (pickups),
   plant site, bomb time left, 5 s/10 s defuse feasibility with kits alive, round clock left before a plant, time since
   the last kill, side of the last death, deaths in the last 5 s per side, whether the last kill was a trade, opening
   kill side), the pre-map prior turned to the CT team, and in-map form (`feats.form_features`: kill-death difference
   in earlier rounds, wins against the economy, gun-round win rate) plus `ct_resid_prev` (see below).
   A round already decided on the timeline (a `round_end` or `bomb_exploded` event at or before the state; 1.6% of
   states, mostly kills after the round ended) is priced from its known winner.
2. **Continuation M** learned from round n's rows with their actual outcome (the benchmark's next-score features +
   round n's start-of-round economy: money left after the buy, equipment, money before the buy, AWPs, rifles of both
   teams; next-round cash `money left + win reward / loss bonus`; economy masked to NaN for rounds without player data,
   which carry zeros) and `a_resid_prev`: team A's round wins earlier in this map minus what a small buy-only model
   (`expect`: equipment, money before the buy, side, map, pistol) predicted.

## Development results (harness, walk-forward Apr-Aug 2026)

| config | diff | 95% interval | months better | notes |
|---|---|---|---|---|
| rounddyn-v1-dynform-econcont (quick, Aug) | -0.00221 | | 1/1 | local |
| rounddyn-v1-dynform-econcont | -0.00161 | [-0.00235, -0.00085] | 5/5 | AWS |
| rounddyn-v2-econ4-quick (Aug) | -0.00383 | [-0.00586, -0.00184] | 1/1 | local |
| rounddyn-v2-econ4 | -0.00232 | [-0.00326, -0.00138] | 5/5 | AWS; per month -0.00199 -0.00291 -0.00097 -0.00152 -0.00350 |

## Research-cache experiments (not harness runs)

`build_table.py` caches every 2026 state with features, labels and the benchmark map model's continuations;
`exp.py` (round model variants, map log loss through the benchmark continuation), `exp2.py` (continuation
variants), `exp3.py`/`exp4.py` (state-level continuations). Numbers are map log loss minus benchmark, Apr..Aug.

Round model (benchmark continuation):
- `dyn` (timeline dynamics, no prior): round ll -0.0034..-0.0037 every month, map -0.00016 -0.00013 -0.00014 -0.00007 -0.00020.
- `dyn_prior` (+ CT prior): round ll -0.0041..-0.0051, map -0.00044 -0.00009 -0.00016 +0.00046 -0.00056.
- `dyn_form` (+ in-map form): map -0.00037 -0.00015 -0.00018 +0.00053 -0.00063.
- `dyn_all` (+ headshot/util deaths, pickups, money alive, n kills; Aug only): same as dyn_prior.
- monotone constraints on alive/equipment, 1200 trees: no change (Jun-Aug).
- Lesson: a round-ll gain of 0.005 is only ~0.0002 of map ll; the round model is not where the map price is decided.

Continuation (round model dyn_form):
- trained on outcome rows, score-only (`none`): -0.00008 -0.00122 +0.00037 -0.00003 -0.00109.
- `econ` (round n economy): -0.00095 -0.00214 -0.00017 -0.00092 -0.00175.
- `econ_form` (+ upsets, gun-round win rate): -0.00103 -0.00330 +0.00094 -0.00188 -0.00221 (this is v1).
- `econ_kd` (+ kill-death diff; zero for 2025 maps with no timeline): worse, -0.00138 on Aug.
- `econ_resid` / `econ_r` (residual form): -0.00122 / -0.00141 mean.
- `econ2` (+ masked economy, next-round cash, form): mean -0.00193; seed 1 -0.00203.
- `econ3` (masked economy + cash, no form): -0.00187 -0.00312 -0.00040 -0.00205 -0.00268.
- **`econ4`** (econ3 + residual form): -0.00118 -0.00381 -0.00080 -0.00219 -0.00388 (this is v2).
- `econ5`/`econ6` (+ series context: earlier maps of the series, map wins / round diff): worse (-0.00139 / -0.00157 mean).
- `econ7` (+ Elo logit next to the market prior): worse (-0.00091 mean).
- `econ8` (+ tier): -0.00248 mean, within noise, not kept.
- `econ9` (+ per-player money counts): no gain.
- `econ10`/`econ11` (+ team round-win rates on this map / per side): worse (overfit; map-level features have ~37k
  independent labels). As a linear prior adjustment instead: worse too (June +0.003).
- continuation tree params: depth 4 worse, depth 6 / min_child_weight 300 / 1200 trees same; min_child_weight 1500
  worse; 4-seed bagging same.
- walk-forward Platt recalibration of the final price: slopes 0.97..1.11, no gain.
- state-level continuation (xgboost on in-round states with the round-level M as base margin): much worse Apr-Jul.
- survivor-aware continuation (M with survivors of each team, integrated over multiclass P(survivors | state,
  outcome)): oracle survivors would be worth ~-0.004 more, but the integrated version: Aug -0.00382 vs -0.00388.
  As a delta on the round-level M (minus the same integral at the round's start state): Aug -0.00026, Jul -0.00008.
