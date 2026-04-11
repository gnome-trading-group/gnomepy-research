# Re-export everything from reporting.backtest so existing imports keep working.
from gnomepy_research.reporting.backtest import (  # noqa: F401
    DEFAULT_SECTIONS,
    BacktestReport,
    Curves,
    ReportSection,
    adverse_selection_section,
    build_curves,
    market_making_section,
    compute_sharpe,
    plot_pnl,
    plot_pnl_by_symbol,
    plot_position,
    plot_spread,
)

__all__ = [
    "BacktestReport",
    "Curves",
    "build_curves",
    "compute_sharpe",
    "plot_pnl",
    "plot_position",
    "plot_pnl_by_symbol",
    "plot_spread",
    "ReportSection",
    "DEFAULT_SECTIONS",
    "adverse_selection_section",
    "market_making_section",
]
