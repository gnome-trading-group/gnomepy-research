# Research Learnings

Cross-session knowledge base. Read by `/research` Step 2 before forming each hypothesis — check for a
matching `strategy_type` and listing IDs. Written by `/research` Step 7 when a session completes or
stalls.

**One entry per session.** Step 7 *updates* the existing entry for a session rather than appending a
second one; append only when the session has no entry yet. This file previously held contradictory
duplicate verdicts for three sessions because Step 7 blind-appended.

**Format** — every entry uses this shape:

```
### [YYYY-MM-DD] <session_name> — <STATUS> (best=iter_NNN)
**Type**: arb | mm | momentum | custom | **Venues**: ... | **Listings**: ... | **Window**: ...
**Best**: iter_NNN — PnL $X, N fills, per-bar Sharpe Y

**Worked:** / **Failed:** / **Insights:** / optional **Market:**
```

> ### ⚠️ Sharpe verdicts dated before 2026-09-30 are contaminated
>
> Every "Sharpe target structurally unachievable" conclusion in this file was measured against a
> target mis-scaled by ~1776x. `summary.json`'s `sharpe` is a **per-bar** ratio at 10s bars
> (`gnomepy.reporting.metrics.compute_sharpe` defaults to `annualize=False`, and
> `BacktestReport.sharpe` never overrides it), but targets were written as though it were annualized.
> At 10s bars there are 3,153,600 bars/year, so the annualized equivalent is ~1776x the per-bar value.
>
> These sessions were therefore chasing annualized Sharpe of 178 to 1776 and were recorded as
> `stalled` no matter how they actually performed:
>
> | Session | best per-bar | ≈ annualized | target | target ≈ annualized |
> |---|---|---|---|---|
> | `cross_prediction_arb` | 0.0920 | 163 | >0.5 | 888 |
> | `informed_pmm` | 0.1882 | 334 | >1.0 | 1776 |
> | `kalshi_sports_mm` | 0.0784 | 139 | >0.1 | 178 |
> | `oracle_spread_maker` | 0.0739 | 131 | >0.3 | 533 |
> | `informed_pmm__bid_ask_spread_pnl_only` | 0.0088 | 16 | >1.0 | 1776 |
>
> A per-bar Sharpe of 0.002 is roughly 3.6 annualized, so several of these were respectable results
> recorded as failures. **Re-read their Sharpe conclusions before trusting them**; the PnL, fill and
> structural findings are unaffected. Targets have been in per-bar units since 2026-09-30.

## Learnings

<!-- Entries updated in place here by /research Step 7 -->

### [2026-09-19] cross_prediction_arb — STALLED (best=iter_054)
**Type**: arb | **Venues**: polymarket + kalshi | **Window**: multi-scenario, 6 event types
**Best**: iter_054 — aggregate PnL $158.66, per-bar Sharpe 0.008 (53 iterations against spec max 25)

> Supersedes two earlier conflicting entries — one recording "stalled, 49 iterations, $158.42
> aggregate" and one recording "CONVERGED, best=iter_043, $106.03". The $106.03 figure is the
> single-event (Seahawks/Titans) result; $158.66 is the multi-scenario aggregate and the final
> recorded best.

**Worked:**
- Custom state machine (SCANNING → ENTERING → PARTIAL_FILL → HEDGED/UNWINDING) with `base_qty` tracking for persistent base positions across re-entries.
- PM legs as TAKER (immediate fill at ask), Kalshi legs as MAKER (`join_best_bid`, rest until filled).
- Optimal params: `min_contract_price=0.10`, `imbalance_timeout=60s`, `max_position=2000`, `min_edge_cents=1.0`, `unwind_spread_mult=2.0`, `price_model=join_best_bid`, `pm_taker_mode=True`, `allow_dutch_book=False`.
- Per-event PnL: seahawks/titans $106.03, tennis geerts/albot $32.50, baseball atlanta/milwaukee $7.96, csgo $9.47, tennis zverev/paul $2.34, baseball det/pit $0.14.

**Key fix — cancel/reentry race:** when a PM taker rejects → Kalshi cancel + SCANNING reset → on the same tick a new Kalshi maker order and the cancel arrive together, and the exchange cancels the *new* order. Fixed with `_CANCEL_SETTLE_NS=200ms` cooldown via `ps.last_cancel_ts`. Also needed: a PM rejection handler in `on_execution_report` (REJECT/EXPIRE → cancel Kalshi + reset), an orphan position check in `_on_scanning`, and `_close_all_positions` must not send a cancel for PM legs in taker mode.

**Failed:**
- `allow_scaling` + `base_quantities` dict — overly complex, replaced by a simple `base_qty` int.
- HEDGED phase — unnecessary; go straight to SCANNING once all legs fill.
- `min_contract_price=0.25` loses 26% PnL vs 0.10. `imbalance_timeout=120s` loses fills vs 60s.
- `max_entry_size` parameter — replaced by the natural `max_position` cap; sub-entries of 500 add overhead and partial-fill timeout risk.
- Aggressive Kalshi bids (`price_improve_bps=50-200`) — any bid above best_bid costs edge without enough fill benefit.
- Dutch book P2 (K_SEA + K_TIT) creates Kalshi leg conflicts with P0/P1 (−$806).
- `max_position>2000` causes catastrophic imbalance: PM taker fills large qty, Kalshi maker only fills ~700.
- `min_edge_cents<1.0` allows marginal arbs that are negative after fees.

**Insights:**
- Cross-prediction arb (PM_YES + K_TIT = $1, PM_NO + K_SEA = $1) works across NFL, tennis, baseball and CS:GO. NFL dominates PnL — a 2.75h game with scoring plays gives sustained in-game probability swings.
- **Kalshi liquidity ceiling is the binding constraint**: the book fills at most ~2000 contracts at our maker bid over a 3-hour game. The entire $106 single-event PnL comes from exactly one entry cycle per pairing, held to resolution.
- Parametric fees (7% taker × price × (1−price)) hit hardest at low-price contracts — tennis_zverev_paul filled in [0.15, 0.20] with 80% fee drag. Kalshi's 1.75% maker fee is negligible against PM's 7% taker.
- Data coverage varies by event: CS:GO produced 256 S3 missing-data warnings.
- Low Sharpe here is real but structural to the *design*, not the venue: hold-to-resolution means all PnL arrives at one settlement, leaving ~80% of time buckets at zero. An early-exit mechanism or multi-game data would change this. (The original "target >0.5 unreachable" framing is contaminated — see the warning above.)

**Market:** PM Seahawks YES 222852 / NO 222853, Kalshi Seahawks 97203 / Titans 97202, 2026-08-24 00:25–03:11 UTC. PM opened Seahawks at 0.54, Kalshi at 0.73 — a 19% structural disagreement all game. Titans won, confirming PM had the better fair value. P0 net $38 + P1 net $68 on $3892 notional (2.7% in 3 hours).

### [2026-09-14] informed_pmm — COMPLETED (best=iter_021)
**Type**: mm | **Venues**: kalshi (ref) + polymarket (quote) | **Listings**: ref=129651, quote=130435/130436 (YES/NO) | **Window**: 2026-08-19 14:30–14:48
**Best**: iter_021 — PnL $15.43, 167 fills, per-bar Sharpe 0.125

> Supersedes an earlier entry recording "stalled at max iterations (20/20), best=iter_014, $14.50".
> iter_021 added the short-YES model fix below and is the final recorded best.

**Critical model fix (iter_021):**
- Prediction markets do not allow naked short YES. "Shorting YES" means buying the NO contract — a different `security_id` on the same exchange.
- Auto-discover the NO listing: `registry.get_event_contracts(event_id=...)` returns both YES and NO; filter for the differing `security_id`.
- Track net position as `yes_qty - no_qty`, both clamped ≥ 0 before dividing by `lot_size`.
- Intent routing: decrease-position → close YES longs first (ASK YES), then BID NO with the remainder. Increase-position → close NO longs first (ASK NO), then BID YES. Always return both Intents; zero price/size acts as a cancel.
- Include the NO `listing_id` in the config's `listings` so the backtester has NO order book data.
- Binary price complement: `no_price = PRICE_SCALE - yes_price`.

**Worked:**
- `value_function_kalshi_cal.npz` dominates all other value functions (g20/g50/g100/Q10) — it quotes tighter spreads that generate fills throughout the window, not just at resolution.
- `size=3_000_000` (3 contracts/order) is optimal: 1M < 2M < 3M > 4M > 5M.
- `max_position=100` is the saturation point — liquidity caps position at ~97 contracts regardless of a higher setting. Below 50 limits pre-spike long accumulation.
- `kalman_Q=1e-4, kalman_R=1e-2` optimal. Lower R (1e-3) adds noise; higher Q (1e-3) tested and worse.
- `inventory_fade=0.5` (default) optimal. fade=1.0 with max_position=100 is mathematically equivalent since position never reaches 100.

**Failed:**
- Swapping ref/quote venues (Polymarket-as-ref) = −$11.82. Kalshi-as-ref is the correct direction.
- `size=5M`: overshoots the cap to −118 contracts. **The hard position cap is only checked on intent, not on fill**, so large orders overfill past the limit.
- `max_position=30`: limits pre-spike long building. The pre-spike long built at ~0.63 is the primary PnL source when price spikes to 0.99.
- Faster Kalman (Q=1e-3): more noise in the reference signal → worse fill quality.
- `processing_time_ns=1ms` (vs 5ms): slightly worse — faster quoting doesn't help in this maker context.

**Insights:**
- The window is a single resolution event (0.63 → 0.99 at 14:46). All PnL comes from holding a long YES position at resolution; post-spike short accumulation (NO at ~$0.01) is MTM-neutral.
- Exchange ID mapping: `exchange_id=4` = Polymarket, `exchange_id=5` = Kalshi. Verify via the registry, not the spec description.
- **Significance is not achievable on this window**: 107 time buckets against 176 needed, DSR=0.0. Results are real within the backtest but indistinguishable from noise. Validate on a longer range covering multiple resolution events.
- The HJB `max_position` acts as both a hard cap and an inventory risk normalizer — a larger cap makes HJB treat current inventory as less risky, giving tighter spreads and more fills. Saturation occurs when position never approaches the cap.
- Order size scaling near the cap: `bid_size_scaled = scale × size if position > 0 else size`. Same-direction orders fade, opposite-direction stays full. Intentional — it fades in the direction that increases risk.

### [2026-09-15] informed_pmm__bid_ask_spread_pnl_only — STALLED (best=iter_012)
**Type**: mm | **Venues**: kalshi (ref) + polymarket (quote) | **Listings**: ref=97203/97202 (Kalshi Seahawks YES/NO), quote=222852/222853 (PM Seahawks YES/NO) | **Window**: 2026-08-24 00:25–03:11 (Seahawks vs Titans)
**Best**: iter_012 — PnL $2.50, 18 fills, per-bar Sharpe 0.00883 (stalled at iter_018)

**Worked:**
- `size=5_000_000` (5 contracts/quote) and `max_position=20` are both necessary for correct sizing. Larger size (10M) causes a naked short via the engine order-update overfill race below.
- `yes_bid_tau_threshold=0.20` is a **critical safety guard** — it stops YES bid quoting in the final ~33 min of the game. At 0.10 it allows YES bids in the final ~16 min, accumulating +21 YES contracts that resolve at $0 when Seahawks lose ($8.29 loss).
- `kalman_Q=0.01, kalman_R=0.01`. Q has **zero effect on fills** — 0.001 produced identical 18 fills, PnL and Sharpe. Fills are driven by participants crossing our quotes, not by reference-signal precision.
- `close_spread_ticks=1` closes at fill_price + 1 tick; the effective close price is HJB-optimal because the periodic update overrides the reactive close within 1s.

**Failed:**
- `size=10_000_000` — naked short. See the engine order-update behaviour below; oversells by ~8.59 contracts.
- `get_effective_quantity` cap for ask size — two competing close paths (reactive `on_execution_report` + periodic `_on_market_data_impl`) reset each other's order sizes even with an accurate cap. Both paths must coexist as designed.
- Removing the reactive close from `on_execution_report` — removes the `_last_quote_ts` reset, so bid quoting gets more frequent, accumulating more position and *more* naked short exposure.

**Insights:**
- **Engine order-update behaviour**: a new Intent for the same `(eid, sid, side)` slot **resets the remaining size** of the existing order. A 1.078433M remaining order updated to 5M allows 5M *more* fills. Cap ask size using `_filled_qty`, not `get_effective_quantity` — and even then only size=5M makes the timing race rare.
- **Fill pattern**: all profitable fills are on the NO contract (223653, Titans NO ≈ "Seahawks lose"), bought at 0.41–0.48 in the first 4 minutes and sold at 0.51–0.61 in the last ~3 minutes, held 2+ hours as maker asks.
- **Directional, not spread capture**: PnL comes from NO contracts rallying in the final period — effectively a 2-hour carry with a maker exit at 0.51–0.61 against a maker entry at 0.41–0.48.
- **Football market thinness**: only 18 fills over 2.75 hours despite quoting at 1s intervals. Polymarket participants for this game are very sparse; more frequent quoting will not help.
- Only 5/8 positive time buckets, 107 against the 176 needed, DSR=0.0. The window cannot support a significance claim either way. The original "56x improvement impossible with single-game data" framing is contaminated by the unit bug above — the target was the problem, not only the data.
- Shares the HJB `max_position` dynamics and order-size scaling behaviour documented under `informed_pmm`.

### [2026-09-13] equivalent_event_arb__fix_unfilled_legs — COMPLETED (best=iter_019)
**Type**: arb | **Venues**: polymarket + kalshi | **Listings**: 117905/117906 (PM), 115763/115764 (Kalshi), event_ids [22765, 22167] | **Window**: 2026-08-19, resolution ~22:09 UTC (46 min in)
**Best**: iter_019 — PnL +$6.653, per-bar Sharpe 0.056

> Supersedes an earlier "running (first threshold pass at iter 3)" milestone entry, whose latency
> finding is folded in below.

**Worked:**
- Maker orders (`join_best_bid`, 120s imbalance timeout) fixed the fill-rate problem. Maker bids rest in the book and fill when sellers cross, capturing the same opportunities without a latency race.
- `min_contract_price=0.25` in `compute_target` blocks new entries when any leg's BID is below the threshold. Cut end-of-match pair-2 adverse selection from −2.864 to −0.475 (iter_013 → iter_019); pair-1 entries (both legs bid ≥ 0.31) unaffected.

**Failed:**
- Taker-first mode (iter 1) caused 60% leg imbalance — orders arrived after brief ask dips had closed.
- `DepthCoverageCost` penalty (iter 2) had zero effect: raw arb edge (20%+) dwarfs a 188bps penalty.
- `cancel_grace_ns` (iter_014) keeps both legs' targets active during the grace period, accumulating more fills.
- Selective cancel in `compute_target` (iter_015/016) — the engine batches fill events before the next `compute_target` cycle.
- Pre-entry ASK check (iter_017) — both pairs show 60–70bps ASK edges at entry, indistinguishable.
- `min_pure_arb_bps=250` (iter_018) affects entry *and* exit validity, causing premature exits that wipe pair-1 profit.

**Insights:**
- These CS:GO prediction market arbs have sub-second ask dips. At 50ms latency taker orders consistently miss; maker bids are the right approach.
- **Very latency-sensitive**: 2x latency (100ms) → thresholds fail (PnL −26.71 vs +6.63 at 50ms).
- The PM ask book for low-probability outcomes is thin (12.74 shares within 2 cents), so large taker orders fail immediately. Maker bids sidestep this.
- End-of-match adverse selection: near resolution one contract converges to 0, creating stale bids that fill immediately. The contract's own BID price is the right filter — below 0.25 means the market is nearly decided.
- 6.06 shares residual persists — a structural race: order placed when bid=0.25–0.30, filled after the bid drops to 0.20 in a ~1.7s window post-pair-1-exit. Not eliminable with a simple price threshold.
- Tennis markets (Zverev vs Paul, 46-min window) produce fewer and smaller arbs than CS:GO (12-min window).

### [2026-09-13] equivalent_event_arb__low_vol_changes — STALLED (best=iter_011)
**Type**: arb | **Venues**: polymarket + kalshi | **Listings**: 253844/253845 (PM) + 246127/246126 (Kalshi), CS:GO IEM Beijing 2026 | **Window**: 12 min
**Best**: iter_011 — PnL $19.47, per-bar Sharpe 0.099

**Worked:**
- `max_position` is the primary PnL lever. `max_position=115` captures the natural ~115-unit batch fill at match resolution (iter_11 $19.47 vs iter_4 $16.40 baseline).
- Scale-up logic fills in waves based on available depth; the first wave holds ~115 units for this window.

**Failed:**
- `max_position>115` triggers a second scale-up wave (~85bps, 14 units) that cannot complete before window end → the whole position stays ENTERING and EXIT never fires → −$0.35 (iter 6 at 200, iter 10 at 130).
- `imbalance_timeout_ns=120s` cost −$18.58 (iter 2). Taker mode structurally unviable (14% round-trip fees).
- `fill_prob_target` price model −$10.31 (iter 7). Pairing cooldowns create orphan positions and overfills (iter 5). `fill_risk_lambda>0` blocks profitable entries (iter 7).

**Insights:**
- **Depth boundary**: the book fills ~115 units in one batch at 17:05:54 (resolution). `max_position` 110 and 115 fill at identical timestamps; 116+ likely triggers a second wave. Specific to this 12-minute window.
- 5 imbalance timeout events per window, all at 56–177bps — structural, not filterable via vol regime or edge threshold.
- The 30s timeout is critical to the winning trade: Kalshi 1-unit fills at exactly 30.562s, triggering the 115-unit scale-up; both legs fill within 62s → EXIT.
- PnL concentrates in the final 90s when the outcome clarifies (exit_edge 1724–1740 bps).
- The vol regime EWMA filter (alpha=0.999, warmup=50, threshold=1.5) adds modest protection but does not reduce timeout events — all occur in the "normal" regime.
- Original note "Sharpe 0.099 vs target >1.0 structurally unachievable in a 72-bar window, DSR=0.000 (need 282+ bars)" — the bar-count point stands; the target comparison is contaminated.

### [2026-09-16] oracle_spread_maker — STALLED (best=iter_019 local / iter_006 API)
**Type**: mm | **Venues**: polymarket (quote) + kalshi (ref) | **Listings**: PM Seahawks YES 222852 / NO 222853, Kalshi ref 97203/97202 | **Window**: 2026-08-24 00:25–03:11 UTC
**Best**: per-bar Sharpe 0.016, PnL +$0.688

> Local and API disagree on the best iteration (iter_019 vs iter_006) — a symptom of the accept/reject
> gate recording a best that `sessions update` never received. Trust the local `best/` snapshot.

**Worked:**
- Simultaneous YES+NO bidding at `poly_mid ± base_spread` with a vol gate. Optimal: `base_spread=0.01`, `vol_gate=0.005` at 250ms cadence, force-cancel every 5 quotes, `divergence_gate=0.025`, `max_position=30`.
- Achieved perfectly balanced 29.7 YES = 29.7 NO fills — pure spread capture, outcome-independent (+$0.69 regardless of winner).

**Failed:**
- Adaptive divergence spread (`divergence_spread_coeff=1.0`) reaches +$0.68 matched-pair PnL in isolation but creates temporal mismatch: sparse fills mean YES fills early (high prices) and NO fills late, causing 2 stale-bid YES fills at 0.20–0.31 during the game crash = −$2.02 directional loss.
- Kalshi-guided YES bid reduction → directional NO excess (5.5 lots): +$4.63 when Seahawks lost, but −$0.86 had they won — violates the spec.
- Removing force-cancel (iter_023) → 84 fills but worse average prices. `vol_gate=0.006` → YES excess during Seahawks domination. `base_spread=0.02` → YES excess.

**Insights:**
- **Temporal pairing rule (critical)**: a YES+NO matched pair's profit is `1 − YES_fill − NO_fill`, which equals `1 − 2×spread` *only* if both fill at the same market state. If YES fills at p1 and NO at p2 after a decline, combined cost is `1 + (p1−p2) − 2×spread`, which exceeds 1 when `p1−p2 > 2×spread`. **More fills = better temporal pairing** — a narrow spread means fills cluster in time, so p1 ≈ p2.
- **Vol gate timing**: must be computed at quoting cadence (250ms), not per market event (~5.8ms). Per-event log-returns have std ≈ 0.0004, far below the 0.005 gate, so the gate never fires. At 250ms it correctly identifies the crash period.
- **Force-cancel** every 5 quotes (1.25s) serves two purposes: OMS ring buffer management (256 slots), and a 250ms blank window that prevents stale-bid fills during rapid moves. Removing it degrades fill quality even though it raises fill count.
- **Divergence gate** at 2.5% prevents 5 adversely-selected fills during sustained Kalshi-PM disagreement. Should not be replaced by adaptive spread widening (temporal mismatch above).
- Fill ceiling: ~73 fill events per game (market-activity-driven), ~36 matched pairs × $0.02 = ~$0.72 max spread PnL per game. Raising PnL requires more venues or more games, not better parameters.

### [2026-09-16] kalshi_sports_mm — STALLED (best=iter_010)
**Type**: mm | **Venues**: kalshi (quote) + polymarket (ref) | **Listings**: 97203 (Kalshi Seahawks), 97202 (Kalshi Titans), 222852 (PM Seahawks) | **Window**: 2026-08-24 00:25–03:11
**Best**: iter_010 — PnL $9.15, ~254 fills, per-bar Sharpe 0.0784 (stalled at 21/30 iterations)

**Worked:**
- Dual-contract Kalshi quoting using the Polymarket microprice (Kalman-filtered, Q=1e-4, R=1e-2) as the fair-value oracle. Best params: `base_spread=0.03`, `inventory_skew=0.002`, `divergence_gate=0.020`, `overround_gate=0.05`, `max_exposure=30`.
- PnL decomposes as ~$1.50 spread capture + ~$7.65 directional MTM from the final position at resolution.

**Failed:** all 11 attempts to beat iter_010 were rejected — raw divergence (noisy), raw vol returns (noisy), EMA vol (too slow), `vol_ema_alpha=0.3` + `vol_gate=0.003`, jump detector `halt_ticks=20`, 30% Kalshi blend in FV (fills 254 → 221, Sharpe 0.060), `max_exposure=50` (more fills, worse Sharpe), `max_exposure=20` + higher skew (fewer fills, worse Sharpe).

**Insights:**
- **Vol gate bug (critical)**: `_ref_returns` tracks Kalman log returns ≈ 0.001/tick. At Q=1e-4 a 10% price move gives a Kalman delta ≈ 0.0001/tick, so vol std ≈ 0.001 ≪ `vol_gate=0.005` — the gate is effectively disabled. Every vol-filtering variant made things worse via false positives from microprice noise.
- **Scoring plays** cause instantaneous permanent probability jumps (10–15%), not sustained volatility, so vol-based gates cannot detect them. They are the source of the 2 negative PnL buckets out of 10, and no filter was found that removes them without also removing the spread-capture alpha.
- **Alpha structure**: the edge is cross-venue PM–Kalshi price discrepancy, not pure spread capture. PM microprice differs from Kalshi mid by 1–3%; the strategy undercuts one side of the Kalshi spread on that basis. Ask fills at mean 0.519 vs book ask 0.525 (~0.6% undercut), 73% of asks at or below market ask. Blending Kalshi mid into FV dilutes the oracle advantage — always use pure PM microprice.
- **Inventory risk**: `D = q_seahawks - q_titans` is the single net directional exposure; 0.002/lot skew manages per-contract accumulation. The final position at game end is inherently directional.
- **Not saturated**: 254 fills / 79,528 Kalshi trades = 0.32% participation. The 3% spread covers the 1.75% maker fee at p=0.43 but limits fill frequency; tighter spreads raise adverse selection from PM-Kalshi divergence.
- Overround (Seahawks_mid + Titans_mid) ranged 0.90–1.18 (mean 1.001, std 0.024); `overround_gate=0.05` fires ~3% of the time.

## Sessions with no entry

These ran or were scaffolded but never produced a learnings entry. Listed so the gap is visible
rather than silent: `n_exchange_arb`, `stablecoin_stat_arb`, `cs2_win_probability`,
`equivalent_event_arb`, `dutch_book_arb`, `implication_boundary_arb`, `prediction_market_mm`.
