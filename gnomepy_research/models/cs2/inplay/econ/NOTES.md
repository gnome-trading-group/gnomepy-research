# econ: next-round economy in the map price

**Entry:** `entry.py` (= `model.make()` with defaults; scored as `econ-v2t_soft03`, re-run as `econ-final`).
**Full development (2026-04..08):** -0.00267 log loss vs the benchmark, map-bootstrap 95% interval
[-0.00342, -0.00194], better in 5/5 months (-0.00283, -0.00300, -0.00106, -0.00237, -0.00338), round-start states
-0.00221. Audit passes on every run. ~24 min per 5-month run on 8 vCPU.

## Idea

The benchmark prices a state as `P(A wins round | s) * M(score if A wins) + P(A loses | s) * M(score if A loses)`,
with M seeing only the score. But this round decides the next round's economy: survivors keep their weapons, dead
players lose theirs, and everybody's money moves by the win reward / loss bonus / kill rewards / plant bonus.

The entry keeps the benchmark round model and replaces M(score) by M(s, o):

- base margin: the benchmark's score-state map model (all maps since 2025) at the next score;
- plus a boosted correction (300 trees, depth 4, min_child_weight 300) learned on in-round states of the 2026 timeline
  maps, each row conditioned on its **realised** round outcome o. Rows are drawn from P(s, o), so this estimates
  P(map | s, o) directly without forecasting survivors: given o, who is alive now is an upper bound on who keeps their
  weapons and the model learns the rest.
- correction features (team A/B view, given o): next score state; alive, equipment alive, rifles and AWPs alive, kills
  so far; money left after this round's buy, round-start equipment; loss counters after the round; projected next-round
  pre-buy money per team (per player: money + kill rewards earned so far + win pay or loss bonus (+ plant bonus for Ts),
  capped at $16,000), "resources" = projected money + equipment alive; number of players who could buy rifle + armor
  next round (survivors keeping a rifle only need a top-up) and with full-buy money; tier. Economy features are blank
  when the next round is a half or overtime-half start (money resets).
- target: 70% map result + 30% M given the *actual* next-round pre-buy economy (money_pre, equipment kept, rifles/AWPs
  kept, buy capability; a model fitted on every map with player data). The distilled part removes the noise of the
  rounds after the next; it costs ~0.0001 in the pooled mean but gives the best worst month and the tightest interval.

## Money rules learned from the data (`players.money_pre` / `money`)

Zero-kill losers of eliminated rounds (2025-06 onward), median income by the team's recent results within the half:
- pistol loss pays $1,900, then $2,400, $2,900, $3,400 (cap): the loss counter starts at 1 at each half start;
- a win lowers the counter by one, it does not reset it (lose, lose, win, lose pays $1,900; win then lose pays $1,400);
- a win pays ~$3,250 (+$250 for bomb objectives: medians 3,700-4,050 with kills); Ts that lose after a plant get extra
  (~$800); Ts that survive a timeout loss get nothing;
- kill rewards by weapon as in the guides (AWP $100, SMG $600, shotgun $900, knife $1,500, most others $300).
- dead players' equipment is gone next round (equip_pre ~0 for 99% of the dead); survivors keep ~90% of theirs.

## What worked, what didn't

Quick checks (August, cached features, `dev.py` + `sweep_local.py`):
- economy correction, first cut: -0.0028; ablations: score-only correction -0.0005, score + alive -0.0011, money-only
  -0.0023: the economy is what helps.
- money rules learned (counter, per-player projection, weapon kill rewards) and buy-capability counts: no change on
  August, top-gain features (`d_nbuy` is the most used split) and they hold up on the full run.
- tier in the correction: -0.0040 on August; on the full run +0.0005 over no tier (5/5 months). The benchmark M is
  miscalibrated by tier (tier b most).
- score-state M as a *feature* ("base") looked good on August (-0.0031) but **hurt June/July on the full run (3/5
  months)**: the in-sample M on the training maps is more accurate than it will be out of sample. Dropped. An
  out-of-sample M (fitted without the timeline maps) as a feature did not help either.
- no gain: round-model economy extras (money left, rifles/helmets alive, utility value alive: -0.00003), tier in the
  round model, augmenting the correction with round-start rows of maps without timelines (2025: worse at round start),
  recency weighting (half-life 2 and 4 months: worse), cross-fitted base margin (worse), side-specific full-buy
  thresholds and "can rebuy twice" counts, a next-round-winner model as a feature (out-of-fold), M at the projected
  next economy as a feature or base margin, prior / map / bo / map number as features, rifles dropped by the dead
  (loot), monotone constraints, 3-seed bagging, depth/trees/min_child_weight/colsample changes.

## Every configuration scored on the full development run (entry minus benchmark)

| name | diff | 95% interval | months | per month (Apr..Aug) | round start |
|---|---|---|---|---|---|
| econ-v1 (first features) | -0.00224 | [-0.00298, -0.00151] | 5/5 | -0.00261 -0.00302 -0.00058 -0.00135 -0.00260 | -0.00176 |
| econ-v2 (+ buy counts) | -0.00220 | [-0.00292, -0.00148] | 5/5 | -0.00260 -0.00293 -0.00051 -0.00127 -0.00262 | -0.00168 |
| econ-v2d6 | -0.00226 | [-0.00312, -0.00141] | 5/5 | -0.00260 -0.00315 -0.00022 -0.00137 -0.00274 | -0.00174 |
| econ-v2rx (+ round-model extras) | -0.00217 | [-0.00291, -0.00145] | 5/5 | -0.00254 -0.00287 -0.00045 -0.00129 -0.00268 | -0.00167 |
| econ-money (money features only) | -0.00201 | [-0.00273, -0.00130] | 5/5 | -0.00228 -0.00281 -0.00040 -0.00125 -0.00230 | -0.00170 |
| econ-v2b (+ base feature) | -0.00173 | [-0.00273, -0.00073] | 3/5 | -0.00326 -0.00198 +0.00031 +0.00125 -0.00323 | -0.00137 |
| econ-v2bt (+ base, tier) | -0.00216 | [-0.00329, -0.00108] | 3/5 | -0.00369 -0.00208 +0.00010 +0.00035 -0.00394 | -0.00182 |
| econ-v2bt_d5 | -0.00206 | [-0.00338, -0.00078] | 3/5 | -0.00385 -0.00196 +0.00038 +0.00103 -0.00411 | -0.00174 |
| econ-v2b_mt (tier in M) | -0.00222 | [-0.00368, -0.00072] | 3/5 | -0.00407 -0.00300 +0.00025 +0.00261 -0.00434 | -0.00187 |
| econ-v2bt_t600 | -0.00192 | [-0.00328, -0.00059] | 3/5 | -0.00391 -0.00187 +0.00078 +0.00155 -0.00414 | -0.00166 |
| econ-v2t (+ tier) | -0.00274 | [-0.00372, -0.00182] | 5/5 | -0.00309 -0.00320 -0.00060 -0.00232 -0.00350 | -0.00226 |
| econ-v1t | -0.00275 | [-0.00373, -0.00184] | 5/5 | -0.00306 -0.00321 -0.00072 -0.00232 -0.00353 | -0.00230 |
| econ-v2_soft05 (50% distilled) | -0.00197 | [-0.00244, -0.00151] | 5/5 | -0.00211 -0.00243 -0.00094 -0.00156 -0.00224 | -0.00148 |
| econ-v2_reg | -0.00209 | [-0.00272, -0.00145] | 5/5 | -0.00245 -0.00269 -0.00066 -0.00130 -0.00243 | -0.00158 |
| econ-v2n (+ next-round model) | -0.00229 | [-0.00304, -0.00155] | 5/5 | -0.00263 -0.00316 -0.00054 -0.00133 -0.00265 | -0.00178 |
| econ-v3nb (+ side buys, rebuy) | -0.00220 | [-0.00291, -0.00148] | 5/5 | -0.00255 -0.00300 -0.00048 -0.00130 -0.00256 | -0.00168 |
| econ-v2t_reg | -0.00255 | [-0.00343, -0.00174] | 5/5 | -0.00279 -0.00284 -0.00080 -0.00223 -0.00333 | -0.00208 |
| econ-v2t_d6 | -0.00262 | [-0.00372, -0.00159] | 5/5 | -0.00286 -0.00328 -0.00017 -0.00211 -0.00356 | -0.00217 |
| econ-v2t_t500 | -0.00275 | [-0.00374, -0.00180] | 5/5 | -0.00310 -0.00321 -0.00056 -0.00233 -0.00356 | -0.00229 |
| econ-v2t_mcw100 | -0.00270 | [-0.00368, -0.00179] | 5/5 | -0.00300 -0.00317 -0.00060 -0.00231 -0.00349 | -0.00223 |
| **econ-v2t_soft03 (entry)** | **-0.00267** | **[-0.00342, -0.00194]** | **5/5** | -0.00283 -0.00300 -0.00106 -0.00237 -0.00338 | -0.00221 |
| econ-v2t_cs1 | -0.00277 | [-0.00375, -0.00184] | 5/5 | -0.00303 -0.00332 -0.00060 -0.00234 -0.00357 | -0.00232 |
| econ-v2tc (+ bo, map_no, has_prior) | -0.00247 | [-0.00367, -0.00133] | 5/5 | -0.00244 -0.00310 -0.00104 -0.00151 -0.00341 | -0.00204 |
| econ-v2tq (+ M at projected economy) | -0.00265 | [-0.00356, -0.00174] | 5/5 | -0.00337 -0.00281 -0.00070 -0.00174 -0.00362 | -0.00223 |
| econ-v2tne (+ out-of-sample base) | -0.00260 | [-0.00351, -0.00170] | 5/5 | -0.00321 -0.00287 -0.00073 -0.00157 -0.00358 | -0.00215 |
| econ-v2t_mono | -0.00273 | [-0.00374, -0.00179] | 5/5 | -0.00302 -0.00327 -0.00057 -0.00227 -0.00354 | -0.00223 |
| econ-v2t_bag3 | -0.00276 | [-0.00373, -0.00185] | 5/5 | -0.00313 -0.00322 -0.00062 -0.00232 -0.00353 | -0.00230 |
| econ-v2t_soft03_mono | -0.00266 | [-0.00343, -0.00192] | 5/5 | -0.00283 -0.00301 -0.00104 -0.00230 -0.00341 | -0.00218 |
| econ-v2t_soft015 | -0.00271 | [-0.00356, -0.00189] | 5/5 | -0.00296 -0.00312 -0.00086 -0.00231 -0.00346 | -0.00224 |
| econ-final_quick (entry, August only, local) | -0.00342 | [-0.00524, -0.00164] | 1/1 | -0.00342 | -0.00281 |

The top configurations (v2t, v1t, v2t_t500, v2t_cs1, v2t_bag3, v2t_soft03) are within 0.0001 of each other, well inside
noise. The entry is v2t_soft03 because it has the best worst month (June) and the tightest interval, which matters for
a 3-month holdout that must be better in 2 of 3 months. Configurations were run as research jobs from commits on this
branch; the losing options were removed from the code afterwards (see git history, e.g. `9f1f5a0`, `ba5621b`).

Gains by tier (v1, all months): every tier improves; tier b most (-0.007), c/d (82% of states) -0.002. Gains grow with
the number of events in the round (-0.0023 at round start, -0.0030 after 5-7 events): survivors carry information.

## Point-in-time

All features of a state in round r come from: round r's round-start columns (`players.money`, `equip`, `prim`, `armor`,
`kit`; `rounds` economy and score), earlier rounds' scores (loss counters, score state), the tier of the map, and round
r's events with `seq <= event_seq` (deaths, kills and their weapons, plant). Round r+1's `money_pre`/`equip_pre` and
all outcomes are used only in training, on maps that ended before the scored month. Nothing depends on other states in
the batch. The harness truncation audit passes on every run (max abs diff 0.0).

## Files

- `entry.py`: the entry; `model.py`: the model; `econ_features.py`: state/economy features and money rules.
- `dev.py`, `sweep_local.py`: cache one month's features and compare variants quickly (development only).
- `mkcfg.py`: writes per-configuration entry files and the research-job array file for sweeps.
