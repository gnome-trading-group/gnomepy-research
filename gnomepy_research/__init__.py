# Re-export commonly used types at top level for convenience
from gnomepy_research.research.replay import SignalReplay, DataSlice, AssetConfig, ReplayResult
from gnomepy_research.research.returns import Horizon, compute_forward_returns
from gnomepy_research.research.metrics import SignalMetrics, MetricResult
from gnomepy_research.research.pnl import SignalPnL, PnLResult
from gnomepy_research.research.discovery import SignalDiscovery
from gnomepy_research.research.report import SignalReport

__all__ = [
    # Research
    "SignalReplay",
    "DataSlice",
    "AssetConfig",
    "ReplayResult",
    "Horizon",
    "compute_forward_returns",
    "SignalMetrics",
    "MetricResult",
    "SignalDiscovery",
    "SignalPnL",
    "PnLResult",
    "SignalReport",
]
