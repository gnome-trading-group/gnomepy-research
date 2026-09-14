from __future__ import annotations

from gnomepy_research.arb import ArbPhase
from gnomepy_research.strategies.equivalent_event_arb import EquivalentEventArb
from gnomepy_research.strategies.target_portfolio import Listing


class LowVolEquivalentEventArb(EquivalentEventArb):
    def __init__(
        self,
        event_ids: list[int],
        max_position: int = 0,
        exchange_fees: dict[str, dict[str, float]] | None = None,
        min_pure_arb_bps: float = 2.0,
        taker_edge_threshold_bps: float = 99999.0,
        use_taker_orders: bool = False,
        exit_edge_threshold_bps: float = 2.0,
        taker_order_type: str = "LIMIT",
        max_staleness_ns: int = 60_000_000_000,
        processing_time_ns: int = 5_000_000,
        allow_scaling: bool = True,
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
        vol_regime_alpha: float = 0.999,
        vol_regime_warmup: int = 50,
        vol_regime_threshold: float = 1.5,
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
        self._vol_regime_threshold = vol_regime_threshold
        self._vol_regime_warmup = vol_regime_warmup
        self._vol_regime_alpha = vol_regime_alpha

        # Per-listing EWMA spread tracking (spread in bps relative to mid)
        self._ewma_spread: dict[Listing, float] = {}
        self._last_spread: dict[Listing, float] = {}
        self._spread_ticks: dict[Listing, int] = {}

    def _on_book_updated(
        self,
        listing: Listing,
        bids: list[tuple[int, int]],
        asks: list[tuple[int, int]],
        timestamp: int,
    ) -> None:
        super()._on_book_updated(listing, bids, asks, timestamp)

        if not bids or not asks:
            return
        bid = bids[0][0]
        ask = asks[0][0]
        if bid <= 0 or ask <= 0:
            return

        mid = (bid + ask) / 2.0
        spread_bps = (ask - bid) / mid * 10000.0
        self._last_spread[listing] = spread_bps
        n = self._spread_ticks.get(listing, 0)
        if n == 0:
            self._ewma_spread[listing] = spread_bps
        else:
            alpha = self._vol_regime_alpha
            self._ewma_spread[listing] = alpha * self._ewma_spread[listing] + (1.0 - alpha) * spread_bps
        self._spread_ticks[listing] = n + 1

    def _is_high_vol_regime(self) -> bool:
        for lst in self._group.all_listings:
            ticks = self._spread_ticks.get(lst, 0)
            if ticks < self._vol_regime_warmup:
                continue
            ewma = self._ewma_spread.get(lst, 0.0)
            current = self._last_spread.get(lst, 0.0)
            if ewma > 0.0 and current / ewma > self._vol_regime_threshold:
                return True
        return False

    def _compute_group_target(self, timestamp: int) -> dict[Listing, int]:
        if self._portfolio is None and self._is_high_vol_regime():
            return {}
        return super()._compute_group_target(timestamp)
