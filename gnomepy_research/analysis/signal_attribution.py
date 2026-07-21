"""Signal attribution — post-hoc PnL decomposition and ablation config generation."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def attribute_pnl_by_signal(
    fills_df: pd.DataFrame,
    custom_metrics_df: pd.DataFrame,
    signal_columns: list[str],
    mid_column: str = "mid_price",
) -> pd.DataFrame:
    """Decompose per-fill edge into signal contributions.

    For each fill, the total edge (fill_price vs mid) is decomposed
    proportionally to each signal's magnitude at fill time.

    custom_metrics_df must be indexed by timestamp and contain signal_columns.
    Returns a DataFrame indexed by fill timestamp with one column per signal
    ('pnl_attr_<signal>') plus 'total_edge' and 'unexplained_edge'.
    """
    from gnomepy.reporting.metrics import _is_buy

    if fills_df.empty or custom_metrics_df.empty:
        return pd.DataFrame()

    required = set(signal_columns) | {mid_column}
    if not required.issubset(custom_metrics_df.columns):
        missing = required - set(custom_metrics_df.columns)
        raise ValueError(f"custom_metrics_df missing columns: {missing}")

    fill_ts = pd.DataFrame({"timestamp": fills_df.index}).reset_index(drop=True)
    metrics_reset = custom_metrics_df[[mid_column] + signal_columns].sort_index().reset_index()
    metrics_reset.columns = ["timestamp"] + [mid_column] + signal_columns

    merged = pd.merge_asof(
        fill_ts.sort_values("timestamp"),
        metrics_reset.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    mid_at_fill = merged[mid_column].astype(float).values
    fill_price = fills_df["fill_price"].astype(float).values
    sign = fills_df["side"].map(lambda s: 1.0 if _is_buy(s) else -1.0).values

    valid = mid_at_fill > 0
    total_edge_bps = np.where(
        valid,
        (mid_at_fill - fill_price) * sign / np.where(valid, mid_at_fill, 1.0) * 10_000,
        0.0,
    )

    signal_vals = merged[signal_columns].astype(float).values
    signal_magnitudes = np.abs(signal_vals)
    total_magnitude = signal_magnitudes.sum(axis=1, keepdims=True)
    total_magnitude = np.where(total_magnitude > 0, total_magnitude, 1.0)
    attribution_weights = signal_magnitudes / total_magnitude

    result = pd.DataFrame(index=fills_df.index)
    result["total_edge_bps"] = total_edge_bps
    for i, col in enumerate(signal_columns):
        result[f"pnl_attr_{col}"] = total_edge_bps * attribution_weights[:, i]

    attributed = sum(result[f"pnl_attr_{col}"] for col in signal_columns)
    result["unexplained_edge_bps"] = total_edge_bps - attributed

    return result


def generate_ablation_configs(
    base_config_path: str | Path,
    ablation_params: dict[str, Any],
    configs_output_dir: str | Path,
) -> list[Path]:
    """Generate N+1 configs for signal ablation: one baseline + one per param zeroed.

    ablation_params: {strategy_arg_name: zero_value}
        e.g., {'flow_weight': 0.0, 'vol_weight': 0.0}

    Returns [baseline_config, ablated_flow_config, ablated_vol_config, ...]
    The /research command runs each one and compares Sharpe/PnL to baseline.
    """
    base_config_path = Path(base_config_path)
    configs_output_dir = Path(configs_output_dir)
    configs_output_dir.mkdir(parents=True, exist_ok=True)

    with open(base_config_path) as f:
        base_config = yaml.safe_load(f)

    baseline_path = configs_output_dir / "ablation_baseline.yaml"
    with open(baseline_path, "w") as f:
        yaml.dump(base_config, f)

    config_paths = [baseline_path]

    for param_name, zero_value in ablation_params.items():
        ablated = yaml.safe_load(yaml.dump(base_config))
        strategy_args = ablated.setdefault("strategy", {}).setdefault("args", {})
        strategy_args[param_name] = zero_value

        ablated_path = configs_output_dir / f"ablation_no_{param_name}.yaml"
        with open(ablated_path, "w") as f:
            yaml.dump(ablated, f)
        config_paths.append(ablated_path)

    return config_paths


def summarize_ablation_results(
    param_names: list[str],
    baseline_summary: dict[str, Any],
    ablated_summaries: list[dict[str, Any]],
    primary_metric: str = "sharpe",
) -> dict[str, Any]:
    """Compute marginal contribution of each signal from ablation results.

    Returns a dict mapping param_name to marginal contribution
    (positive = signal helps; negative = signal hurts).
    """
    baseline_val = float(baseline_summary.get(primary_metric, 0.0))
    contributions: dict[str, float] = {}

    for param_name, ablated_summary in zip(param_names, ablated_summaries):
        ablated_val = float(ablated_summary.get(primary_metric, 0.0))
        contributions[param_name] = baseline_val - ablated_val

    return {
        "baseline": {primary_metric: baseline_val},
        "contributions": contributions,
        "signals_to_remove": [p for p, c in contributions.items() if c <= 0],
    }
