from __future__ import annotations

import pandas as pd

from gnomepy.java.recorder import BacktestResults
from gnomepy.reporting.report import BacktestReport


def load_results(path: str) -> BacktestReport:
    """Load a backtest result directory as a BacktestReport.

    Args:
        path: Local directory path or S3 URI containing backtest results.

    Returns:
        BacktestReport with all curves and metrics ready to use.
    """
    results = BacktestResults.from_parquet(path)
    return BacktestReport(results)


def compare_results(*paths: str, metric: str = "sharpe") -> pd.DataFrame:
    """Load multiple backtest result directories and return a comparison table.

    Args:
        *paths: Local or S3 paths to backtest result directories.
        metric: Column to sort by (descending). Default: "sharpe".

    Returns:
        DataFrame with one row per backtest, sorted by metric descending.
    """
    rows = []
    for path in paths:
        try:
            report = load_results(path)
            summary = report.summary()
            pnl = report.pnl_curve
            max_dd = float((pnl - pnl.cummax()).min()) if not pnl.empty else float("nan")
            rows.append({
                "path": path,
                "final_pnl": summary.get("final_pnl"),
                "sharpe": summary.get("sharpe"),
                "sortino": summary.get("sortino"),
                "fill_count": summary.get("fill_count"),
                "max_drawdown": max_dd,
                "total_fees": summary.get("total_fees"),
                "pct_positive_buckets": summary.get("pct_positive_buckets"),
            })
        except Exception as exc:
            rows.append({"path": path, "error": str(exc)})

    df = pd.DataFrame(rows)
    if metric in df.columns and df[metric].notna().any():
        df = df.sort_values(metric, ascending=False)
    return df.reset_index(drop=True)
