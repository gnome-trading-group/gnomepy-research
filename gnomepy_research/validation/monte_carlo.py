"""Monte Carlo robustness checks for simulation parameters.

Generates sweep configs for latency and queue sensitivity analysis,
and provides a fill-based bootstrap that requires no engine re-runs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def generate_latency_sweep_config(
    base_config_path: str | Path,
    output_path: str | Path,
    latency_values_nanos: list[int] | None = None,
    profile_names: list[str] | None = None,
) -> Path:
    """Generate a sweep YAML varying each profile's network latency.

    Values go in the top-level ``sweep.profiles`` section, which is the only part
    ``gnomepy.sweep.expand_sweep`` expands; ``profiles`` itself stays fixed and keeps a
    scalar default so the config is still runnable on its own.

    Profiles sweep independently, so the job count is
    ``len(latency_values_nanos) ** len(profile_names)``. Pass ``profile_names`` to vary one
    venue at a time and keep that from exploding past the 100-job cap.
    """
    if latency_values_nanos is None:
        latency_values_nanos = [2_000_000, 5_000_000, 10_000_000, 20_000_000]

    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    sweep_profiles: dict[str, Any] = {}
    for profile_name, profile in config.get("profiles", {}).items():
        if profile_names is not None and profile_name not in profile_names:
            continue
        network_latency = profile.setdefault("network_latency", {})
        network_latency["type"] = "static"
        network_latency.setdefault("latency_nanos", latency_values_nanos[0])
        sweep_profiles[profile_name] = {"network_latency": {"latency_nanos": list(latency_values_nanos)}}

    if sweep_profiles:
        config.setdefault("sweep", {})["profiles"] = sweep_profiles

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        yaml.dump(config, f)

    return output_path


def generate_queue_sweep_config(
    base_config_path: str | Path,
    output_path: str | Path,
    cancel_ahead_probs: list[float] | None = None,
    profile_names: list[str] | None = None,
) -> Path:
    """Generate a sweep YAML varying queue ``cancel_ahead_probability``.

    Switches every profile to the probabilistic queue model and sweeps the
    probability via the top-level ``sweep.profiles`` section.

    Profiles sweep independently, so the job count is
    ``len(cancel_ahead_probs) ** len(profile_names)``. Pass ``profile_names`` to vary one
    venue at a time.
    """
    if cancel_ahead_probs is None:
        cancel_ahead_probs = [0.1, 0.3, 0.5, 0.7, 0.9]

    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    sweep_profiles: dict[str, Any] = {}
    for profile_name, profile in config.get("profiles", {}).items():
        if profile_names is not None and profile_name not in profile_names:
            continue
        profile["queue_model"] = {
            "type": "probabilistic",
            "cancel_ahead_probability": cancel_ahead_probs[0],
        }
        sweep_profiles[profile_name] = {
            "queue_model": {"cancel_ahead_probability": list(cancel_ahead_probs)}
        }

    if sweep_profiles:
        config.setdefault("sweep", {})["profiles"] = sweep_profiles

    output_path = Path(output_path)
    with open(output_path, "w") as f:
        yaml.dump(config, f)

    return output_path


def bootstrap_pnl_paths(
    fills_df: pd.DataFrame,
    market_df: pd.DataFrame,
    n_paths: int = 500,
    block_minutes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Block-bootstrap PnL paths from a single backtest run.

    Resamples fills in temporal blocks to generate a distribution of PnL
    outcomes without re-running the engine. Each path is a plausible
    realization of the same strategy under different fill timing.

    Returns a DataFrame with n_paths columns, each a bootstrapped cumulative
    PnL series aligned to market_df's timestamps.
    """
    from gnomepy.reporting.metrics import build_curves

    if fills_df.empty or market_df.empty:
        return pd.DataFrame(index=market_df.index)

    rng = np.random.default_rng(seed=seed)
    block_ns = block_minutes * 60 * 1_000_000_000

    fills_sorted = fills_df.sort_index().copy()
    ts_ns = fills_sorted.index.astype(np.int64).values
    t_start, t_end = int(ts_ns.min()), int(ts_ns.max())
    duration_ns = t_end - t_start

    if duration_ns <= 0 or len(fills_sorted) == 0:
        return pd.DataFrame(index=market_df.index)

    n_blocks = max(1, int(duration_ns / block_ns))
    block_starts_ns = np.linspace(t_start, t_end - block_ns, max(n_blocks, 1)).astype(np.int64)

    paths: dict[int, pd.Series] = {}
    for path_idx in range(n_paths):
        sampled_starts = rng.choice(block_starts_ns, size=n_blocks, replace=True)
        offset = int(t_start - sampled_starts.min())

        chunks = []
        for bs in sampled_starts:
            mask = (ts_ns >= bs) & (ts_ns < bs + block_ns)
            if mask.any():
                chunk = fills_sorted.iloc[mask].copy()
                chunk.index = pd.DatetimeIndex(
                    chunk.index.astype(np.int64) + offset + int(bs - t_start)
                )
                chunks.append(chunk)

        if not chunks:
            paths[path_idx] = pd.Series(0.0, index=market_df.index)
            continue

        bootstrapped_fills = pd.concat(chunks).sort_index()
        curves = build_curves(market_df, bootstrapped_fills)
        paths[path_idx] = curves.pnl

    result = pd.DataFrame(paths)
    result.index = market_df.index
    return result


def summarize_mc_paths(pnl_paths: pd.DataFrame) -> dict[str, Any]:
    """Compute percentile Sharpe/PnL stats from a set of MC PnL paths."""
    from gnomepy.reporting.metrics import compute_sharpe

    if pnl_paths.empty:
        return {}

    sharpes = []
    final_pnls = []
    for col in pnl_paths.columns:
        path = pnl_paths[col].dropna()
        if len(path) >= 2:
            sharpes.append(compute_sharpe(path)["sharpe"])
            final_pnls.append(float(path.iloc[-1]))

    if not sharpes:
        return {}

    sharpes_arr = np.array(sharpes)
    pnls_arr = np.array(final_pnls)
    return {
        "n_paths": len(sharpes),
        "sharpe_p5": float(np.percentile(sharpes_arr, 5)),
        "sharpe_p50": float(np.percentile(sharpes_arr, 50)),
        "sharpe_p95": float(np.percentile(sharpes_arr, 95)),
        "pnl_p5": float(np.percentile(pnls_arr, 5)),
        "pnl_p50": float(np.percentile(pnls_arr, 50)),
        "pnl_p95": float(np.percentile(pnls_arr, 95)),
    }
