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
