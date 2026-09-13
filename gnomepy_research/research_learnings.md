# Research Learnings

Cross-session knowledge base. Appended automatically by `/research` when a session completes or stalls,
or when a significant milestone (first threshold-passing iteration) is reached.

Read by `/research` Step 2 before forming each hypothesis — check for matching `strategy_type` and listing IDs.

## Learnings

<!-- Entries appended here by /research Step 7 -->

### [2026-09-13] equivalent_event_arb__low_vol_changes (arb, polymarket+kalshi)
- STATUS: running (new best accepted iter 4, PnL=$16.40)
- WORKED: max_position=100 scaled the 266 bps arb 2x vs max_position=50 (iter 1 $5.79 → iter 4 $16.40, +183%). CS:GO match markets have brief high-edge windows as the match resolves; scaling position size is the primary PnL lever.
- FAILED: Extending imbalance_timeout_ns from 30s to 120s (iter 2) caused -$18.58 by holding an unhedged 50-unit PM ex-RUBY position while the market moved sharply. Enabling taker mode (iter 3) had zero effect because 7% parametric taker fees per leg (14% round-trip) exceed any observed maker edge.
- INSIGHT: Listings 253844/253845 (polymarket) + 246127/246126 (kalshi) — CS:GO match, IEM Beijing 2026. Parametric fee model (7% taker, 0%/1.75% maker) makes taker mode economically unviable for any arb under ~1400 bps. Maker-only execution creates consistent leg-imbalance risk (~28.6% cancel rate, 5 timeout events per 12-min window costing ~$4-5). PnL concentrated in final 2 min when match outcome clarifies (exit_edge=1740 bps). Sharpe structurally limited to ~0.094 by 72 available time bars.
