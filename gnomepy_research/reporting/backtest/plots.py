"""Re-exported from gnomepy.reporting.plots."""
from gnomepy.reporting.plots import (  # noqa: F401
    DEFAULT_SECTIONS,
    MAX_FILLS_DISPLAY,
    SECTION_REGISTRY,
    ReportSection,
    _fills_table_html,
    _kpi_cards_html,
    _summary_table_html,
    assemble_html,
    plot_cross_exchange_spread,
    plot_pnl,
    plot_pnl_by_symbol,
    plot_position,
    plot_spread,
    resolve_sections,
)

__all__ = [
    "ReportSection",
    "DEFAULT_SECTIONS",
    "SECTION_REGISTRY",
    "plot_pnl",
    "plot_position",
    "plot_pnl_by_symbol",
    "plot_spread",
    "plot_cross_exchange_spread",
    "assemble_html",
    "resolve_sections",
]
