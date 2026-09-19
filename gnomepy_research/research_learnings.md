# Research Learnings

Cross-session knowledge base. Appended automatically by `/research` when a session completes or stalls,
or when a significant milestone (first threshold-passing iteration) is reached.

### [2026-09-19] cross_prediction_arb (arb, polymarket + kalshi)
- STATUS: stalled (49 iterations, sharpe target >0.5 unreachable for hold-to-resolution binary arb)
- WORKED: Custom state machine (SCANNING/ENTERING/PARTIAL_FILL/UNWINDING) with base_qty tracking for persistent base positions across re-entries. Maker-first Kalshi legs + PM taker at best ask. join_best_bid price model. Optimal params: min_contract_price=0.10, imbalance_timeout=60s, max_position=2000, min_edge_cents=1.0, unwind_spread_mult=2.0. Multi-scenario config covers 6 event types. Best aggregate PnL: $158.42 across seahawks/titans ($106.03), tennis geerts/albot ($32.50), baseball atlanta/milwaukee ($7.96), csgo ($9.47), tennis zverev/paul ($2.34), baseball det/pit ($0.14).
- FAILED: allow_scaling + base_quantities dict (overly complex, replaced by simpler base_qty int). HEDGED phase (unnecessary — go straight SCANNING after all-filled). min_contract_price=0.25 (loses 26% PnL vs 0.10). imbalance_timeout=120s (loses fills vs 60s). max_entry_size parameter (replaced by natural max_position cap).
- INSIGHT: Cross-prediction arb (PM_YES + K_TIT always = $1, PM_NO + K_SEA always = $1) works across NFL, tennis, baseball, CS:GO. NFL (2.75h game with scoring plays) dominates PnL due to sustained in-game probability swings. Tennis 20-min window gave $32.50. Parametric fees (7% taker × price × (1-price)) hit hardest at low-price contracts — tennis_zverev_paul all fills in [0.15,0.20] price range with 80% fee drag. Data coverage varies significantly by event: CS:GO had 256 S3 missing-data warnings. Kalshi fee (1.75% maker) is negligible vs PM taker (7%); strategy correctly focuses on PM taker entries. Sharpe metric is low (~0.008) for this strategy class because hold-to-resolution means large intraday mark-to-market swings relative to final realized PnL.

Read by `/research` Step 2 before forming each hypothesis — check for matching `strategy_type` and listing IDs.

## Learnings

<!-- Entries appended here by /research Step 7 -->

### informed_pmm__bid_ask_spread_pnl_only — 2026-09-15 — stalled at iter_018 (best=iter_012)

**Session**: informed_pmm__bid_ask_spread_pnl_only | **Strategy type**: mm | **Listing IDs**: ref=97203/97202 (Kalshi Seahawks YES/NO), quote=222852/222853 (Polymarket Seahawks YES/NO)
**Best result**: iter_012 — PnL $2.50, 18 fills, Sharpe 0.00883 | **Date range**: 2026-08-24 00:25–03:11 (Seahawks vs Titans football game)

**What worked:**
- `size=5_000_000` (5 contracts per quote), `max_position=20`: both necessary for correct sizing. Larger size (10M) causes naked short via engine order-update overfill race condition (see below).
- `yes_bid_tau_threshold=0.20`: CRITICAL safety guard — prevents YES bid quoting in the final ~33 min of game. Setting to 0.10 causes catastrophic +21 naked long ($8.29 loss) when game outcome becomes clear.
- `kalman_Q=0.01, kalman_R=0.01`: Kalman Q parameter has zero effect on fills (changing to 0.001 produced identical 18 fills, same PnL, same Sharpe). Fills are driven by market participants crossing quotes, not reference signal precision.
- `close_spread_ticks=1`: Closes at fill_price + 1 tick; effective close price is HJB-optimal (periodic update overrides reactive close within 1s).

**What failed / dead ends:**
- `size=10_000_000`: Engine order-update behavior causes naked short. When new Intent resets remaining of a partially-filled order, the entire new size can fill, overselling by ~8.59 contracts.
- `yes_bid_tau_threshold=0.10`: Allows YES bids in final ~16 min → accumulates +21 YES contracts that resolve at $0 when Seahawks lose.
- `get_effective_quantity` cap for ask size: Two competing close paths (on_execution_report reactive + _on_market_data_impl periodic) reset each other's order sizes even with accurate cap. Both paths must coexist as designed.
- Removing reactive close from `on_execution_report`: Removes `_last_quote_ts` reset → more frequent bid quoting → more position accumulation → MORE naked short exposure.
- `kalman_Q=0.001` (and other Kalman Q values): Zero effect. Fills are market-driven.

**Structural insights:**
- **Engine order-update behavior**: When a new Intent arrives for the same (eid, sid, side) slot, it RESETS the remaining size of the existing order. A 1.078433M remaining order updated to 5M allows 5M MORE fills. Hard cap on ask_size using `_filled_qty` (not `get_effective_quantity`) avoids this only with size=5M where timing races are rare.
- **Fill pattern**: ALL profitable fills are on NO contract (223653, Titans NO ≈ "Seahawks lose"). NO contracts bought at 0.41-0.48 in first 4 minutes, sold at 0.51-0.61 in last ~3 minutes of game. Holds 2+ hours as maker asks waiting for market to rally.
- **Sharpe structural limit**: Only 5/8 positive time buckets. Sharpe=0.00883 vs target 0.5 requires 56x improvement impossible with single-game data. Test on multi-game window to achieve statistical significance.
- **Football market thinness**: Only 18 fills over 2.75 hours despite quoting at 1s intervals. Market participants on Polymarket for this football game are very sparse. More frequent quoting won't help.
- **Directional element**: PnL comes from NO contracts rallying during final game period. Not pure spread capture — equivalent to 2-hour carry with maker exit at 0.51-0.61 vs maker entry at 0.41-0.48.

### informed_pmm — 2026-09-14 — completed (iter_021 accepted, $15.43 PnL)

**Session**: informed_pmm | **Strategy type**: mm | **Listing IDs**: ref=129651 (Kalshi), quote=130435/130436 (Polymarket YES/NO)
**Best result**: iter_021 — PnL $15.43, 167 fills, Sharpe 0.125 | **Date range**: 2026-08-19 14:30–14:48

**Critical model fix (iter_021):**
- Prediction markets do not allow naked short YES. "Shorting YES" = buying NO contract (different security_id, same exchange).
- Auto-discover the NO listing: `registry.get_event_contracts(event_id=...)` returns both YES+NO contracts; filter for the one with a different `security_id`.
- Track net position as `yes_qty - no_qty` (both clamped ≥ 0 before dividing by lot_size).
- Intent routing: decrease-position → close YES longs first (ASK YES), then BID NO with remainder. Increase-position → close NO longs first (ASK NO), then BID YES. Both Intents always returned; zero price/size acts as cancel.
- Include the NO listing_id in the backtest config's `listings` section so the backtester has NO order book data.
- Binary price complement: `no_price = PRICE_SCALE - yes_price`.

**What worked:**
- `value_function_kalshi_cal.npz` dominates all other value functions (g20/g50/g100/Q10).
- `size=3_000_000` (3 contracts per order) is the optimum: 1M < 2M < 3M > 4M > 5M. At size=5M, orders overshoot the max_position cap by ~18 contracts.
- `max_position=100` is the saturation point: market liquidity caps position at ~97 contracts regardless of cap setting above 100. Reducing below 50 limits pre-spike long accumulation.
- `kalman_Q=1e-4, kalman_R=1e-2` are the optimal Kalman parameters. Lower R (1e-3) introduces noise.
- `inventory_fade=0.5` (default) is optimal.

**What failed / dead ends:**
- Swapping ref/quote venues: Polymarket-as-ref, Kalshi-as-quote = -$11.82 PnL. Kalshi-as-ref is the correct direction.
- size=5M: position cap overshoot to -118 contracts.
- max_position=30: limits pre-spike long building, hurts PnL.
- faster Kalman (Q=1e-3): more noise in reference signal → worse fill quality.

**Structural insights:**
- The 2026-08-19 14:30–14:48 window is a single resolution event (price 0.63→0.99 at 14:46). ALL PnL comes from holding a long YES position at resolution. Post-spike short accumulation (NO contracts at ~$0.01) is MTM-neutral.
- Exchange ID mapping: exchange_id=4 = Polymarket, exchange_id=5 = Kalshi (verify via registry, not spec description).
- **Statistical significance is NOT achievable on this window**: only 107 time buckets, need 176. DSR=0.0. Test on longer date range covering multiple resolution events.
- The HJB position parameter (`max_position`) acts as BOTH a hard cap AND an inventory risk normalizer. Saturation occurs when position never approaches the cap.

### informed_pmm — 2026-09-13 — stalled at max iterations (20/20)

**Session**: informed_pmm | **Strategy type**: mm | **Listing IDs**: ref=129651 (Kalshi), quote=130435 (Polymarket)
**Best result**: iter_014 — PnL $14.50, 163 fills, Sharpe 0.125 | **Date range**: 2026-08-19 14:30–14:48

**What worked:**
- `value_function_kalshi_cal.npz` dominates all other value functions (g20/g50/g100/Q10). It quotes tighter spreads that generate more fills throughout the window, not just at resolution.
- `size=3_000_000` (3 contracts per order) is the optimum: 1M < 2M < 3M > 4M > 5M. At size=5M, orders overshoot the max_position cap by ~18 contracts (catastrophic).
- `max_position=100` is the saturation point: market liquidity caps position at ~97 contracts regardless of cap setting above 100. Reducing below 50 limits pre-spike long accumulation.
- `kalman_Q=1e-4, kalman_R=1e-2` are the optimal Kalman parameters. Lower R (1e-3) introduces noise. Higher Q (1e-3) was tested and worse.
- `inventory_fade=0.5` (default) is optimal. Setting fade=1.0 with max_position=100 is mathematically equivalent to fade starting at 100 — no change since position never hits 100.

**What failed / dead ends:**
- Swapping ref/quote venues: Polymarket-as-ref, Kalshi-as-quote = -$11.82 PnL. Kalshi-as-ref is the correct direction.
- size=5M: position cap overshoot to -118 contracts. Hard position cap is only checked on intent, not on fill; large orders overfill past the limit.
- max_position=30: limits pre-spike long building, hurts PnL. The pre-spike long (built at ~0.63) is the primary PnL source when price spikes to 0.99.
- faster Kalman (Q=1e-3): more noise in reference signal → worse fill quality.
- processing_time_ns=1ms (vs 5ms): slightly worse — faster quoting doesn't help in this maker context.

**Structural insights:**
- The 2026-08-19 14:30–14:48 window captures a single resolution event (price 0.63→0.99 at 14:46). ALL PnL comes from holding a long position at resolution time. Post-spike short accumulation (-97 contracts at 0.99) is MTM-neutral within the window.
- **Statistical significance is NOT achievable on this window**: only 107 time buckets, need 176. DSR=0.0. The results are real within the backtest but not statistically distinguishable from noise.
- To validate this strategy statistically, test on a longer date range covering multiple contract resolution events.
- The HJB position parameter (`max_position`) acts as BOTH a hard cap AND an inventory risk normalizer. Larger cap → HJB treats current inventory as less risky → tighter spreads → more fills. Saturation occurs when position never approaches the cap.
- Order size scaling near the cap: `bid_size_scaled = scale × size if position > 0 else size`. Same-direction orders fade; opposite-direction stays full. This is intentional — fades in the direction that increases risk.

### [2026-09-13] equivalent_event_arb__fix_unfilled_legs (arb, polymarket+kalshi)
- STATUS: running (first threshold pass at iter 3)
- WORKED: Maker orders (join_best_bid, 120s imbalance timeout) fixed the fill rate problem. Taker orders (iter 1) failed because 55ms latency meant orders arrived after brief ask dips had closed. Maker bids rest in the book and fill when sellers cross us, capturing the same opportunities without latency race.
- FAILED: Taker-first mode (iter 1) caused 60% leg imbalance — order cancelled on (4,254608) when ask dipped then recovered before order arrived. DepthCoverageCost penalty (iter 2) had zero effect because raw arb edge (20%+) was too large for 188bps penalty to push below threshold.
- INSIGHT: These CS:GO prediction market arbs on Kalshi+Polymarket have very brief (sub-second) ask dips that create entry signals. With 50ms latency, taker orders consistently miss. Maker bids are the right approach. Strategy is VERY latency-sensitive: 2x latency (100ms) → threshold fails (PnL -26.71 vs +6.63 at 50ms). The Polymarket ask book for low-probability outcomes is thin (12.74 shares within 2 cents), so taker orders for large qty fail immediately. Maker bids avoid this problem entirely.

### [2026-09-13] equivalent_event_arb__fix_unfilled_legs — SESSION COMPLETE (iter 020, best=iter_019)
- STATUS: complete — best accepted: iter_019, final_pnl=+6.653, Sharpe=0.056
- WORKED: `min_contract_price=0.25` parameter in compute_target blocks new entries when any leg's BID price is below the threshold. Reduced end-of-match pair 2 adverse selection losses from -2.864 to -0.475 (iter_013 → iter_019). Pair 1 entries (both legs bid ≥ 0.31) unaffected.
- FAILED: cancel_grace_ns (iter_014) — keeps BOTH legs' targets active during grace, accumulating more fills. Selective cancel in compute_target (iter_015/016) — engine batches fill events before next compute_target cycle. Pre-entry ASK check (iter_017) — both pairs have ASK edges of 60-70bps at entry, indistinguishable. High min_pure_arb_bps=250 (iter_018) — affects both entry AND exit validity checks, causes premature exits and wipes pair 1 profit.
- INSIGHT: End-of-match adverse selection in prediction markets: near resolution, one contract converges to 0, creating stale bids that get filled immediately. The contract's BID price itself is the right filter — below 0.25 means the market is nearly decided. Note: 6.06 shares residual persists (structural race condition: order placed when bid=0.25-0.30, filled after bid drops to 0.20 in ~1.7s window post-pair-1-exit). Cannot be eliminated with a simple price threshold.
- DATA: Tennis markets (Zverev vs Paul, 46-min window) produce fewer/smaller arbs than CS:GO (12-min window). Sharpe is severely limited by short data windows (0.056 vs target 1.0) — focus on multi-event datasets for Sharpe improvement.
- MARKET: Listings 117905/117906 (Polymarket), 115763/115764 (Kalshi), event_ids [22765,22167], 2026-08-19. The match resolution triggers the end-of-match dynamics around 22:09 UTC (46 min into the window).

### [2026-09-13] equivalent_event_arb__low_vol_changes (arb, polymarket+kalshi)
- STATUS: stalled (best accepted iter 11, PnL=$19.47, Sharpe=0.099)
- WORKED: max_position is the primary PnL lever. Optimal max_position=115 captures the natural ~115-unit batch fill at match resolution (iter 11 $19.47 vs iter 4 $16.40 baseline). Scale-up logic fills positions in waves based on available depth; first wave holds ~115 units for this window.
- FAILED: max_position>115 triggers a second scale-up wave (~85 bps, 14 units) that cannot complete before window end → entire position stays ENTERING and EXIT never fires → PnL=-$0.35 (iter 6 at 200, iter 10 at 130 both confirmed). imbalance_timeout_ns=120s caused -$18.58 (iter 2). taker mode structurally unviable (14% round-trip fees). fill_prob_target price model caused -$10.31 (iter 7). pairing cooldowns create orphan positions and overfills (iter 5). fill_risk_lambda>0 blocks profitable entries (iter 7).
- DEPTH BOUNDARY: Book fills ~115 units in a single batch at 17:05:54 (resolution time). max_position=110 and max_position=115 both fill in same timestamps. max_position=116+ likely triggers second wave. This boundary is specific to this 12-minute IEM Beijing CS:GO window.
- STRUCTURAL LIMITS: 5 imbalance timeout events per window (structural, not filterable via vol regime or edge threshold — all occur at 56-177 bps). 30s timeout is CRITICAL for winning trade (Kalshi 1-unit fills at exactly 30.562s, triggering the 115-unit scale-up). Sharpe=0.099 (target >1.0) structurally unachievable in 72-bar window. DSR=0.000 (need 282+ bars for significance).
- INSIGHT: Listings 253844/253845 (polymarket) + 246127/246126 (kalshi) — CS:GO match, IEM Beijing 2026. Parametric fee model (7% taker, 0%/1.75% maker) makes taker mode economically unviable. PnL concentrated in final 90s when match outcome clarifies (exit_edge=1724-1740 bps). Kalshi 1-unit initial order fills at ~30.562s → triggers 114-unit scale-up → both legs fill within 62s → EXIT. The vol regime EWMA filter (alpha=0.999, warmup=50, threshold=1.5) adds modest protection but doesn't reduce timeout events (all occur in "normal" regime).

### [2026-09-16] oracle_spread_maker — SESSION STALLED (best=iter_019 local / iter_006 API, Sharpe=0.016)
- STATUS: stalled — simplified vol-gate approach best at Sharpe=0.016, PnL=+$0.688, far from target 0.3
- WORKED: Simplified simultaneous YES+NO bidding at poly_mid ± base_spread with vol_gate. Optimal params: base_spread=0.01, vol_gate=0.005 computed at 250ms cadence (not per-event), force-cancel every 5 quotes, divergence_gate=0.025, max_position=30. Achieved PERFECTLY BALANCED 29.7 YES = 29.7 NO fills — pure spread capture, outcome-independent (+$0.69 regardless of who wins).
- FAILED: Adaptive divergence spread (divergence_spread_coeff=1.0) — achieves +$0.68 matched-pair PnL in isolation but creates temporal mismatch: sparse fills mean YES fills early (high prices) and NO fills late (lower prices), causing 2 late stale-bid YES fills at 0.20-0.31 during game crash = -$2.02 directional loss. Kalshi-guided YES bid reduction → directional NO excess (5.5 lots), PnL=+$4.63 when Seahawks lost but -$0.86 if they'd won (violates spec). Removing force-cancel (iter_023) → 84 fills but worse avg prices. Looser vol_gate=0.006 → YES excess during Seahawks-domination phase. wider base_spread=0.02 → YES excess.
- FUNDAMENTAL CEILING: 73 fill events per game (market-activity-driven), ~36 matched pairs × $0.02/pair = $0.72 max spread PnL per game. Sharpe=0.016 on single-game dataset is the ceiling. Time-series Sharpe is dominated by M2M variance from mid-game price swings (60% of buckets negative as YES price falls while holding YES positions). Target Sharpe=0.3 requires ~19× more PnL per unit variance — structurally unachievable on single-game data.
- TEMPORAL PAIRING RULE (critical insight): Profitability of a YES+NO matched pair = 1 - YES_fill_price - NO_fill_price. This equals 1 - 2×spread ONLY if YES and NO fill at the SAME market state (same poly_mid). If YES fills at p1 and NO fills at p2 (after price decline), combined cost = p1 + (1-p2) - 2×spread = 1 + (p1-p2) - 2×spread > 1 when p1-p2 > 2×spread. MORE FILLS = BETTER TEMPORAL PAIRING. Narrow spread → more fills → fills happen close in time → p1≈p2 → guaranteed profitability.
- VOL GATE TIMING: Must be computed at quoting cadence (250ms), NOT per market event (~5.8ms). Per-event log-returns have std≈0.0004, far below 0.005 gate threshold → gate never fires. At 250ms cadence the gate correctly identifies the game-crash period.
- FORCE-CANCEL MECHANISM: Every 5 quotes (1.25s cycle) has dual purpose: (1) OMS ring buffer management (256 slots), (2) 250ms blank window that prevents stale-bid fills during rapid price moves. Removing it degrades fill quality even though it reduces fill count.
- DIVERGENCE GATE: Hard gate at 2.5% prevents 5 adversely-selected fills during periods of sustained Kalshi-Poly disagreement. Should NOT be replaced by adaptive spread widening (temporal mismatch problem above).
- MARKET: Polymarket Seahawks YES (222852) / NO (222853), Kalshi ref (97203/97202), 2026-08-24 00:25-03:11 UTC, Seahawks-Titans NFL game. Seahawks started at YES=0.73, lost the game (YES→0). NO had majority of fills in final hour as odds shifted. ~73 fill events deterministic from market taker activity.

### [2026-09-16] kalshi_sports_mm — SESSION STALLED (best=iter_010, Sharpe=0.0784, PnL=$9.15)
- STATUS: stalled at 21/30 iterations — best accepted iter_010, could not improve beyond 0.0784 Sharpe (target 0.10)
- WORKED: Dual-contract Kalshi quoting using Polymarket microprice (Kalman-filtered, Q=1e-4, R=1e-2) as fair value oracle. Best params: base_spread=0.03, inventory_skew=0.002, divergence_gate=0.020, overround_gate=0.05, max_exposure=30. ~254 fills across both Kalshi contracts (44k available trades). PnL=$9.15 = ~$1.50 spread capture + ~$7.65 directional MTM from final position at game resolution.
- FAILED: All 11 attempts to improve on iter_010 were rejected: raw divergence (vol noisy), raw vol returns (vol noisy), EMA vol (too slow, same as Kalman), vol_ema_alpha=0.3 + tighter vol_gate=0.003 (same), jump detector halt_ticks=20 (same), 30% Kalshi blend in FV (reduced fills from 254 to 221, Sharpe 0.060). Max_exposure=50 (iter_014, more fills but worse Sharpe), max_exposure=20+higher skew (iter_015, fewer fills worse Sharpe).
- VOL GATE BUG (critical): `_ref_returns` tracks Kalman log returns ≈ 0.001/tick. At Q=1e-4, a 10% price move → Kalman delta ≈ 0.0001 per tick → vol std ≈ 0.001 << vol_gate=0.005. Vol gate is effectively DISABLED. All vol-filtering approaches (raw returns, EMA) made things worse (false positives from microprice noise).
- SCORING PLAY STRUCTURE: NFL scoring plays cause instantaneous permanent probability jumps (10-15%), NOT sustained volatility. They cannot be detected by vol-based gates. They are the source of the 2 negative PnL buckets out of 10. No filter found that removes these without also removing the spread-capture alpha.
- ALPHA STRUCTURE: Edge is cross-venue Polymarket-Kalshi price discrepancy, not pure spread capture. Polymarket microprice differs from Kalshi mid by 1-3%. Strategy undercuts one side of the Kalshi spread based on this discrepancy. Ask fills at mean 0.519 vs book ask 0.525 (we undercut ~0.6%). 73% of asks at or below market ask. Blending Kalshi mid into the FV dilutes this oracle advantage — always use pure Polymarket microprice.
- INVENTORY RISK: D = q_seahawks - q_titans is the single net directional exposure. Inventory skew of 0.002/lot effectively manages per-contract accumulation. The final position at game end is inherently directional (2 of 10 time buckets lose from adverse outcome); this is structural to single-game data.
- SATURATION: 254 fills / 79,528 Kalshi trades = 0.32% participation rate — the strategy is not hitting the fill ceiling. The spread (3%) covers the 1.75% maker fee at p=0.43 but is wide enough to limit fill frequency. Tighter spreads increase adverse selection from Polymarket-Kalshi divergence.
- MARKET: Listings 97203 (Kalshi Seahawks), 97202 (Kalshi Titans), 222852 (Polymarket Seahawks), 2026-08-24 00:25-03:11. Seahawks opened ~0.43, finished losing (~0.0). Overround (Seahawks_mid + Titans_mid) observed 0.90-1.18 (mean 1.001, std 0.024). overround_gate=0.05 fires ~3% of the time.
- SHARPE CEILING: Sharpe=0.0784 on single-game data. The 2 negative buckets (scoring plays) are structural. Achieving Sharpe>0.10 requires either better scoring-play filtering (not found) or multi-game data to diversify the 2-bucket loss.

### [2026-09-19] cross_prediction_arb — SESSION CONVERGED (best=iter_043, PnL=$106.03)
- STATUS: complete — best accepted: iter_043 ($106.03, Sharpe=0.0082), 45 iterations (spec max=25)
- WORKED: Pure arb strategy (P0=PM_YES+K_TIT, P1=PM_NO+K_SEA). Core design: PM legs as TAKER (immediate fill at ask), K legs as MAKER (join_best_bid, rest until filled). Critical state machine: SCANNING→ENTERING→PARTIAL_FILL→HEDGED/UNWINDING. `max_position=2000` captures K book depth limit (~2000 contracts profitable at our bid price). `min_contract_price=0.10` allows near-resolution entries with lower parametric fees.
- KEY FIX: Cancel/reentry race condition caused persistent position imbalances. When PM taker rejects → K cancel + SCANNING reset → same tick: new K maker order AND the cancel arrive simultaneously → exchange cancels the NEW order. Fix: `_CANCEL_SETTLE_NS=200ms` cooldown via `ps.last_cancel_ts` prevents re-entry for 200ms after any K cancel. Also needed: PM rejection handler in `on_execution_report` (REJECT/EXPIRE exec_type → cancel K + reset), orphan position check in `_on_scanning`, fix `_close_all_positions` to not send cancel for PM legs in taker mode.
- FAILED: Aggressive K bids (price_improve_bps=50-200) hurt PnL — any bid above best_bid costs edge without enough fill benefit. Dutch book P2 (K_SEA+K_TIT) creates K leg conflicts with P0/P1 (-$806 loss). max_position>2000 causes catastrophic imbalance (PM taker fills large qty, K maker only fills ~700-700 contracts). Smaller sub-entries max_entry_size=500 add overhead and partial-fill timeout risk. min_edge_cents<1.0 allows marginal arbs that are negative after fees.
- K LIQUIDITY CEILING: K book can fill at most ~2000 contracts at our maker bid price over the 3-hour game window. This is the binding constraint. The entire PnL ($106) comes from exactly one entry cycle per pairing (P0 fills 2000 PM_YES + 2000 K_TIT, P1 fills 2000 PM_NO + 2000 K_SEA) held to resolution.
- ARBITRAGE STRUCTURE: This game (Seahawks vs Titans, 2026-08-24) had a fundamental PM-Kalshi price discrepancy: PM priced Seahawks at 54% but Kalshi priced them at 73%. This 19% disagreement was structural throughout the game. P1 edge (PM_NO+K_SEA) = ~3.4 cents gross, P0 edge (PM_YES+K_TIT) = ~1.9 cents gross. Titans won, confirming PM had the better fair value. Total arb PnL = P0 net $38 + P1 net $68 = $106 on $3892 notional (2.7% net return in 3 hours, risk-free).
- SHARPE CEILING: 0.0082 is structural for hold-to-resolution pure arb. All PnL arrives at game end (single settlement event), creating 80% of time buckets with zero PnL → very low Sharpe. Target >0.5 unachievable with single-game data and resolution-at-end design. Sharpe improves significantly only with early-exit mechanism or multi-game data.
- PARAMETERS: max_position=2000, min_edge_cents=1.0, min_contract_price=0.10, price_model=join_best_bid, pm_taker_mode=True, imbalance_timeout=60s, allow_dutch_book=False.
- MARKET: Listings 222852 (PM Seahawks YES), 222853 (PM Seahawks NO), 97203 (Kalshi Seahawks), 97202 (Kalshi Titans). Date 2026-08-24 00:25-03:11 UTC. Seahawks lost (Titans won). PM started at Seahawks=0.54, Kalshi started at Seahawks=0.73 — 19% structural disagreement throughout game.
