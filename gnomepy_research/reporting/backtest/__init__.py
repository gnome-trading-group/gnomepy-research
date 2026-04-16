from gnomepy_research.reporting.backtest.adverse_selection import adverse_selection_section
from gnomepy_research.reporting.backtest.market_making import market_making_section
from gnomepy_research.reporting.backtest.metrics import Curves, build_curves, compute_sharpe
from gnomepy_research.reporting.backtest.plots import (
    DEFAULT_SECTIONS,
    SECTION_REGISTRY,
    ReportSection,
    plot_pnl,
    plot_pnl_by_symbol,
    plot_position,
    plot_spread,
    resolve_sections,
)
from gnomepy_research.reporting.backtest.report import BacktestReport

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
    "SECTION_REGISTRY",
    "resolve_sections",
    "adverse_selection_section",
    "market_making_section",
]
