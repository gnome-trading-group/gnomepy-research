from __future__ import annotations

from gnomepy_research.strategies.equivalent_event_arb import EquivalentEventArb


class EquivalentEventArbFillFix(EquivalentEventArb):
    """
    Branch of EquivalentEventArb focused on improving fill rate.
    Uses taker-first order mode to guarantee fills against existing ask-side liquidity
    instead of waiting as a maker at the back of the risk_averse queue.
    """

    def __init__(
        self,
        event_ids: list[int],
        max_position: int = 100,
        exchange_fees: dict[str, dict[str, float]] | None = None,
        min_pure_arb_bps: float = 5.0,
        taker_edge_threshold_bps: float = 5.0,
        use_taker_orders: bool = True,
        exit_edge_threshold_bps: float = 5.0,
        taker_order_type: str = "LIMIT",
        max_staleness_ns: int = 60_000_000_000,
        processing_time_ns: int = 5_000_000,
        allow_scaling: bool = False,
        allow_early_exit: bool = True,
        imbalance_timeout_ns: int = 30_000_000_000,
        max_unhedged_qty: int = 0,
        fill_risk_lambda: float = 0.0,
        unwind_spread_mult: float = 2.0,
        depth_coverage_mult: float = 0.0,
        price_window_cents: int = 2,
        price_model: str = "aggressive",
        price_improve_bps: float = 50.0,
        price_edge_share: float = 0.5,
        target_fill_prob: float = 0.7,
        optimal_ev_lambda: float = 30.0,
        maker_reprice_cooldown_ns: int = 0,
        cancel_grace_ns: int = 0,
        debug: bool = False,
    ):
        super().__init__(
            event_ids=event_ids,
            max_position=max_position,
            exchange_fees=exchange_fees,
            min_pure_arb_bps=min_pure_arb_bps,
            taker_edge_threshold_bps=taker_edge_threshold_bps,
            use_taker_orders=use_taker_orders,
            exit_edge_threshold_bps=exit_edge_threshold_bps,
            taker_order_type=taker_order_type,
            max_staleness_ns=max_staleness_ns,
            processing_time_ns=processing_time_ns,
            allow_scaling=allow_scaling,
            allow_early_exit=allow_early_exit,
            imbalance_timeout_ns=imbalance_timeout_ns,
            max_unhedged_qty=max_unhedged_qty,
            fill_risk_lambda=fill_risk_lambda,
            unwind_spread_mult=unwind_spread_mult,
            depth_coverage_mult=depth_coverage_mult,
            price_window_cents=price_window_cents,
            price_model=price_model,
            price_improve_bps=price_improve_bps,
            price_edge_share=price_edge_share,
            target_fill_prob=target_fill_prob,
            optimal_ev_lambda=optimal_ev_lambda,
            maker_reprice_cooldown_ns=maker_reprice_cooldown_ns,
            cancel_grace_ns=cancel_grace_ns,
            debug=debug,
        )
