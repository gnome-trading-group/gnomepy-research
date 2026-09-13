# Research Learnings

Cross-session knowledge base. Appended automatically by `/research` when a session completes or stalls,
or when a significant milestone (first threshold-passing iteration) is reached.

Read by `/research` Step 2 before forming each hypothesis — check for matching `strategy_type` and listing IDs.

## Learnings

<!-- Entries appended here by /research Step 7 -->

### [2026-09-13] equivalent_event_arb__low_vol_changes (arb, polymarket+kalshi)
- STATUS: stalled (best accepted iter 11, PnL=$19.47, Sharpe=0.099)
- WORKED: max_position is the primary PnL lever. Optimal max_position=115 captures the natural ~115-unit batch fill at match resolution (iter 11 $19.47 vs iter 4 $16.40 baseline). Scale-up logic fills positions in waves based on available depth; first wave holds ~115 units for this window.
- FAILED: max_position>115 triggers a second scale-up wave (~85 bps, 14 units) that cannot complete before window end → entire position stays ENTERING and EXIT never fires → PnL=-$0.35 (iter 6 at 200, iter 10 at 130 both confirmed). imbalance_timeout_ns=120s caused -$18.58 (iter 2). taker mode structurally unviable (14% round-trip fees). fill_prob_target price model caused -$10.31 (iter 7). pairing cooldowns create orphan positions and overfills (iter 5). fill_risk_lambda>0 blocks profitable entries (iter 7).
- DEPTH BOUNDARY: Book fills ~115 units in a single batch at 17:05:54 (resolution time). max_position=110 and max_position=115 both fill in same timestamps. max_position=116+ likely triggers second wave. This boundary is specific to this 12-minute IEM Beijing CS:GO window.
- STRUCTURAL LIMITS: 5 imbalance timeout events per window (structural, not filterable via vol regime or edge threshold — all occur at 56-177 bps). 30s timeout is CRITICAL for winning trade (Kalshi 1-unit fills at exactly 30.562s, triggering the 115-unit scale-up). Sharpe=0.099 (target >1.0) structurally unachievable in 72-bar window. DSR=0.000 (need 282+ bars for significance).
- INSIGHT: Listings 253844/253845 (polymarket) + 246127/246126 (kalshi) — CS:GO match, IEM Beijing 2026. Parametric fee model (7% taker, 0%/1.75% maker) makes taker mode economically unviable. PnL concentrated in final 90s when match outcome clarifies (exit_edge=1724-1740 bps). Kalshi 1-unit initial order fills at ~30.562s → triggers 114-unit scale-up → both legs fill within 62s → EXIT. The vol regime EWMA filter (alpha=0.999, warmup=50, threshold=1.5) adds modest protection but doesn't reduce timeout events (all occur in "normal" regime).
