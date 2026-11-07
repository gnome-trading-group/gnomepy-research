# free: in-map form in the map model

## Direction and argument

Free direction. I started from error analysis of the benchmark and ended with **in-map form**: evidence about
how the two teams are actually playing *this map*, beyond the score. None of the other five directions covers it:

- it is not next-round economy: nothing here prices the next buy (no money, no survivors' weapons), and the kill
  features lag a round, so the current round's survivors never enter;
- it is not player/team skill: no player ids and no history across maps; everything resets at each map;
- it is not a richer in-round state: the round model is untouched (see "what didn't work": form in the round model
  adds nothing);
- it is not model structure/calibration: the benchmark's two trees and their combination formula are kept (only the
  map model's `min_child_weight` goes from 50 to 200, needed once it has more features);
- it is not a pre-map prior: the prior is untouched, and the one pre-map idea I tried (form from earlier maps of the
  series) made things worse and is not in the entry.

## Error analysis (benchmark, development months)

- The round model is well calibrated in every alive-count cell (1v1 ... 5v5), before and after the plant, at every
  time bucket, by tier and for odd economies: predicted and realized CT win rates agree to ~0.005 everywhere. There
  is little to fix there without new information.
- Most of the loss is at the map level: 0.60–0.62 log loss in rounds 1–3 (prior quality), falling to ~0.37 late in
  regulation, and 0.50 in overtime. The map model only knows the score, side, loss-bonus counters and the prior. Two
  maps at 6–6 look the same to it whether one team won all its rounds as economic underdog and out-killed its
  opponent, or the reverse.
- Calibration slopes by tier differ (0.90 tier a to 1.21 tier c in August), but a tier feature only ever moved the
  result by noise (+0.0001).

## The entry (`entry.py` → `core.py`)

Round model: the benchmark's, unchanged. Map model: the benchmark's features plus eight in-map form features, all
from team A's view, all computed by `core.history`:

| feature | definition |
|---|---|
| `res_mean` | Σ over previous rounds of (A won − P(A wins \| round-start equipment)), / (rounds + 4) |
| `res_ct_mean`, `res_t_mean` | the same over the previous rounds A played on CT / on T |
| `kill_mean` | Σ (kills − deaths of A) over the rounds before the previous one, / (rounds + 4) |
| `kill_ct_mean`, `kill_t_mean` | the same per side of A |
| `res_dec`, `kill_dec` | recency-weighted versions (weight 0.2^age), / (weights + 4) |

The expectation P(A wins | equipment) is a logistic regression on both sides' round-start equipment (log, product,
squares), AWPs, rifles, helmets, pistol flag and map, refit on the training months in `fit`. Who won a previous round
comes from the score progression (`score_a` of the next round), so the residuals exist for every map since 2025 and
the map model trains on all 800k rounds. Kills come from kill victims in the event timeline (2026 only; NaN before,
which XGBoost routes on its own). In the combination formula, each branch (A wins / loses the current round) updates
the residual features exactly (the current round's expectation is known at its start), and the kill features use
rounds up to the previous one, which are complete.

Why per side: residuals on CT and on T separate "A is stronger today" from "this matchup plays CT-sided today", which
matters most right before and after the side swap. Why recency: the weight on the last round or two picks up
momentum and tilt that the long averages smooth away.

## Point-in-time cleanliness

- Every feature of round r uses: rounds ≤ r of the same map for scores and round-start equipment (allowed), events of
  rounds < r (all before the state's `event_seq`), and the expectation model fitted in `fit` on the training bundle.
  Nothing reads `a_wins`, `outcome`, `a_alive_end`/`b_alive_end`, `map_a_wins`, `kills`, `hp_end` or `alive_end` at
  prediction time (all blank in `ctx` anyway); the previous rounds' winners come from the next round's score.
- Predictions use only the states' own maps (`ctx.subset(states.game_id)`), so they don't depend on the batch.
- The harness audit (100 states re-priced on truncated bundles) passed with max |diff| = 0.0 on every official run.
- The gain is ~0.003, i.e. in the range the brief calls plausible; it is spread over all months and all tiers and
  grows with the number of rounds played (as it should if it is in-map evidence), and starts at round 2 (round 1 can
  only gain through the branch updates).

## Results

### Official harness, full development (2026-04..08), research jobs

| name | config | diff | 95% interval | months better | audit |
|---|---|---|---|---|---|
| free-reskill-mcw200 | res_mean + kill_mean, mcw 200 | −0.00158 | [−0.00258, −0.00058] | 4/5 | pass |
| free-sideform-mcw200 | + per-side res/kill (side6) | −0.00260 | [−0.00393, −0.00125] | 4/5 | pass |
| free-side6-dec04 | side6 + decay 0.4, rich expectation | RESULT_D4 | | | |
| free-side6-dec02 | side6 + decay 0.2, rich expectation (**entry**) | RESULT_D2 | | | |

Quick (August only, local): free-reskill-mcw200 −0.00145 [−0.00341, +0.00044], audit pass.

### Research walk-forward (all scored configurations)

Fast replica of the harness on cached features (`work/`): same months, same training cut-off (maps ended before the
month), same benchmark round model; it reproduces the benchmark exactly (diff 0.00000 on August). Diff vs benchmark,
month order Apr May Jun Jul Aug. Fit noise (seed change) is about ±0.0001–0.0002.

August only (first screen):

| config | diff |
|---|---|
| tier in both models | −0.00073 |
| res_mean in map model | −0.00165 |
| res_mean in round model | −0.00003 |
| survivors history in round model | −0.00001 |
| prior (CT-oriented) in round model | −0.00035 |

Five months:

| config | diff | months |
|---|---|---|
| res_mean (map) | −0.00105 | +.0004 −.0024 +.0005 −.0015 −.0016 |
| res_sum (map) | −0.00095 | |
| res_mean + tier | −0.00102 | |
| res + kill (lag 1 round) | −0.00152 | −.0011 −.0035 +.0005 −.0009 −.0013 |
| res + opening kills | −0.00118 | |
| res + kill + opening kills | −0.00147 | |
| res + economy-adjusted kills | −0.00152 | |
| res + kill + tier | −0.00150 | |
| res + kill, map depth 3 | −0.00050 | |
| res + kill, mcw 100 / 200 / 400 / 1000 | −0.00142 / −0.00179 / −0.00180 / −0.00148 | |
| res + kill mcw200, seeds 1, 2 | −0.00161, −0.00161 | |
| res + kill with current-round kills so far (no lag) | −0.00191 | |
| res + kill + 2025 round starts in the round model | −0.00169 | |
| + per-side residuals | −0.00231 | −.0024 −.0042 −.0004 −.0024 −.0008 |
| + per-side residuals and kills (side6) | −0.00296 | −.0023 −.0057 +.0005 −.0034 −.0022 |
| per-side only + overall kill | −0.00213 | |
| side + CT/T team side form in round model | −0.00230 | |
| per-side only (4) | −0.00284 | |
| side6 + per-side plant rate | −0.00264 | |
| side6 + per-side opening kills | −0.00284 | |
| side6 on core.py (check) | −0.00267 | −.0023 −.0055 +.0009 −.0028 −.0016 |
| side6 shrink 2 / 8 | −0.00272 / −0.00273 | |
| side6 rich expectation | −0.00284 | −.0020 −.0056 −.0001 −.0030 −.0020 |
| rich + current-round kills (no lag) | −0.00277 | |
| rich + tier | −0.00294 | |
| rich, 2026 rows weighted ×3 | −0.00174 | |
| rich, 2025 kills from survivors (alive_end) | −0.00197 | |
| same, no lag | −0.00202 | |
| rich, no-player-data rounds get map-only expectation | −0.00273 | |
| same, mcw 400 | −0.00274 | |
| side6 + series form (earlier maps of the match) | −0.00209 | |
| side6 + series map score only | −0.00213 | |
| side6 + kill form in the round model | −0.00293 | |
| side6 + decay 0.8 / 0.6 / 0.4 / 0.2 / 0.1 | −0.00297 / −0.00307 / −0.00316 / −0.00332 / −0.00331 | |
| decay 0.6 seed 1, 0.4 seed 1, 0.2 seed 1 | −0.00301, −0.00307, −0.00321 | |
| side4 + decay 0.6 (no overall) | −0.00290 | |
| side6 + res decay only (0.4) | −0.00310 | |
| side6 + decay 0.2 + last-round expectation/outcome | −0.00325 | |
| side6 + last-round expectation/outcome (no decay) | −0.00315 | |
| side6 + decay 0.2 + tier | −0.00338 | −.0025 −.0058 −.0008 −.0037 −.0028 |
| side6 + decay 0.2 + no-player fix | −0.00315 | |

## What worked, what didn't

Worked: economy-adjusted round residuals and kill differentials of the map so far in the map model (−0.0016),
splitting them by side (−0.0027), and a recency-weighted copy (−0.0032). A heavier `min_child_weight` once features
were added.

Didn't: any form feature in the round model (team form says nothing about the current round once alive counts and
equipment are known); opening kills, plants, economy-adjusted kills (no gain beyond plain kills); counting the current
round's kills so far (noise); filling 2025 kills from survivors (worse: the model then treats eras alike); upweighting
2026; series form from earlier maps (worse); 2025 round-start rows in the round model (noise); tier (noise);
shrinkage constant (irrelevant to trees).

## Files

- `entry.py`: the entry (config); `core.py`: features, fit, predict.
- `work/`: research scripts (cache builders, the fast walk-forward replica `exp.py`/`wc.py`, experiment batches
  `e*.py`, error analyses `err*.py`). `work/cache/` is not committed.
