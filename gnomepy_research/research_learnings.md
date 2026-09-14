# Research Learnings

Cross-session knowledge base. Appended automatically by `/research` when a session completes or stalls,
or when a significant milestone (first threshold-passing iteration) is reached.

Read by `/research` Step 2 before forming each hypothesis — check for matching `strategy_type` and listing IDs.

## Learnings

<!-- Entries appended here by /research Step 7 -->

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
