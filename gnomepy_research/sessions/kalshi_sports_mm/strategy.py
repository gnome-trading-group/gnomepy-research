from gnomepy_research.sessions.kalshi_sports_mm.kalshi_sports_mm import KalshiSportsMM

strategy = KalshiSportsMM(
    ref_listing_id=222852,
    team_a_listing_id=97203,
    team_b_listing_id=97202,
    size=3_000_000,
    max_exposure=30,
    base_spread=0.02,
    divergence_gate=0.025,
    overround_gate=0.05,
    vol_gate=0.005,
    vol_horizon=20,
    kalman_Q=1e-4,
    kalman_R=1e-2,
    warmup_ticks=50,
    max_ref_staleness_ns=5_000_000_000,
    min_quote_interval_ns=250_000_000,
    tau_pull_threshold=0.05,
    tau_widen_threshold=0.15,
    resolution_time_override_ns=1_787_541_060_000_000_000,
    processing_time_ns=5_000_000,
)
