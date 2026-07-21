"""Command-line interface for gnomepy-research."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click
import yaml

from gnomepy_research import api
from gnomepy_research.notes_sync import pull_notes, push_notes


@click.group()
def main() -> None:
    """gnomepy-research — manage research sessions."""


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@main.group()
def sessions() -> None:
    """Manage research sessions."""


@sessions.command(name="list")
@click.option("--status", type=click.Choice(["running", "completed", "stalled", "paused"]), default=None, help="Filter by status")
@click.option("--limit", default=20, show_default=True, help="Max results")
def sessions_list(status: str | None, limit: int) -> None:
    """List research sessions."""
    try:
        result = api.list_sessions(status=status, limit=limit)
    except RuntimeError as e:
        raise click.ClickException(str(e))

    sessions_data = result.get("sessions", [])
    if not sessions_data:
        click.echo("no sessions found")
        return

    header = f"{'SESSION':<30} {'STATUS':<12} {'ITERS':>5} {'BEST PNL':>10} {'BEST SHARPE':>12} {'OWNER':<20} {'UPDATED'}"
    click.echo(header)
    click.echo("-" * len(header))
    for s in sessions_data:
        name = s.get("sessionName", "")[:30]
        status_val = s.get("status", "")
        iters = s.get("iterationCount", 0)
        best_pnl = s.get("bestPnl")
        best_sharpe = s.get("bestSharpe")
        owner = (s.get("owner") or "")[:20]
        updated = (s.get("updatedAt") or "")[:19].replace("T", " ")
        pnl_str = f"{best_pnl:.4f}" if best_pnl is not None else "—"
        sharpe_str = f"{best_sharpe:.4f}" if best_sharpe is not None else "—"
        click.echo(f"{name:<30} {status_val:<12} {iters:>5} {pnl_str:>10} {sharpe_str:>12} {owner:<20} {updated}")


@sessions.command(name="get")
@click.argument("session_name")
def sessions_get(session_name: str) -> None:
    """Get a research session (JSON output)."""
    try:
        result = api.get_session(session_name)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    click.echo(json.dumps(result, indent=2))


@sessions.command(name="create")
@click.argument("session_name")
@click.option("--spec", required=True, type=click.Path(exists=True), help="Path to spec.yaml")
@click.option("--description", default="", help="Session description")
@click.option("--tags", default="", help="Comma-separated tags")
@click.option("--branch", default=None, help="Git branch (defaults to research/<session_name>)")
def sessions_create(session_name: str, spec: str, description: str, tags: str, branch: str | None) -> None:
    """Register a new research session."""
    spec_path = Path(spec)
    spec_yaml = spec_path.read_text()

    if not description:
        parsed = yaml.safe_load(spec_yaml)
        description = parsed.get("description", "") if isinstance(parsed, dict) else ""

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    try:
        result = api.create_session(
            session_name=session_name,
            spec_yaml=spec_yaml,
            description=description,
            tags=tag_list,
            branch=branch or f"research/{session_name}",
        )
    except RuntimeError as e:
        if "409" in str(e):
            click.echo(f"session '{session_name}' already exists")
            return
        raise click.ClickException(str(e))

    click.echo(f"created session '{result.get('session_name', session_name)}'")


@sessions.command(name="update")
@click.argument("session_name")
@click.option("--status", type=click.Choice(["running", "completed", "stalled", "paused"]), default=None)
@click.option("--description", default=None)
@click.option("--tags", default=None, help="Comma-separated tags (replaces existing)")
@click.option("--best-iteration", type=int, default=None)
@click.option("--best-pnl", type=float, default=None)
@click.option("--best-sharpe", type=float, default=None)
def sessions_update(
    session_name: str,
    status: str | None,
    description: str | None,
    tags: str | None,
    best_iteration: int | None,
    best_pnl: float | None,
    best_sharpe: float | None,
) -> None:
    """Update a research session's metadata."""
    fields: dict = {}
    if status is not None:
        fields["status"] = status
    if description is not None:
        fields["description"] = description
    if tags is not None:
        fields["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
    if best_iteration is not None:
        fields["best_iteration"] = best_iteration
    if best_pnl is not None:
        fields["best_pnl"] = best_pnl
    if best_sharpe is not None:
        fields["best_sharpe"] = best_sharpe

    if not fields:
        raise click.UsageError("provide at least one field to update")

    try:
        api.update_session(session_name, **fields)
    except RuntimeError as e:
        raise click.ClickException(str(e))

    click.echo(f"updated session '{session_name}'")


# ---------------------------------------------------------------------------
# Iterations
# ---------------------------------------------------------------------------

@main.group()
def iterations() -> None:
    """Record research iterations."""


@iterations.command(name="record")
@click.argument("session_name")
@click.option("--iteration", required=True, type=int, help="Iteration number")
@click.option("--type", "iter_type", required=True, type=click.Choice(["local", "sweep", "manual"]))
@click.option("--title", required=True, help="One-line hypothesis summary")
@click.option("--description", default="", help="Markdown description (Hypothesis / Changes / Analysis / Next)")
@click.option("--metrics", required=True, help="JSON object of metric scalars")
@click.option("--metadata", default="{}", help="JSON object of iteration metadata")
@click.option("--environment", default="{}", help="JSON object of environment info")
@click.option("--timestamp", default=None, help="ISO 8601 timestamp (defaults to now)")
def iterations_record(
    session_name: str,
    iteration: int,
    iter_type: str,
    title: str,
    description: str,
    metrics: str,
    metadata: str,
    environment: str,
    timestamp: str | None,
) -> None:
    """Record an iteration result for a research session."""
    try:
        metrics_dict = json.loads(metrics)
    except json.JSONDecodeError as e:
        raise click.UsageError(f"--metrics is not valid JSON: {e}")
    try:
        metadata_dict = json.loads(metadata)
    except json.JSONDecodeError as e:
        raise click.UsageError(f"--metadata is not valid JSON: {e}")
    try:
        environment_dict = json.loads(environment)
    except json.JSONDecodeError as e:
        raise click.UsageError(f"--environment is not valid JSON: {e}")

    try:
        result = api.record_iteration(
            session_name=session_name,
            iteration=iteration,
            type=iter_type,
            title=title,
            description=description,
            metrics=metrics_dict,
            metadata=metadata_dict,
            environment=environment_dict,
            timestamp=timestamp,
        )
    except RuntimeError as e:
        raise click.ClickException(str(e))

    click.echo(f"recorded iteration {result.get('iteration', iteration)} for '{session_name}'")


@iterations.command(name="record-from-results")
@click.argument("session_name")
@click.option("--iteration", required=True, type=int, help="Iteration number")
@click.option("--type", "iter_type", required=True, type=click.Choice(["local", "sweep", "manual"]))
@click.option("--title", required=True, help="One-line hypothesis summary")
@click.option("--description", default="", help="Markdown description (Hypothesis / Changes / Analysis / Next)")
@click.option("--results-dir", required=True, type=click.Path(exists=True), help="Results directory containing summary.json")
@click.option("--extra-metrics", default="{}", help="JSON object of additional metrics to merge (e.g. custom derived metrics)")
@click.option("--extra-metadata", default="{}", help="JSON object of metadata (config_name, thresholds_met, changes, etc.)")
@click.option("--timestamp", default=None, help="ISO 8601 timestamp (defaults to now)")
def iterations_record_from_results(
    session_name: str,
    iteration: int,
    iter_type: str,
    title: str,
    description: str,
    results_dir: str,
    extra_metrics: str,
    extra_metadata: str,
    timestamp: str | None,
) -> None:
    """Record an iteration by reading metrics and environment from a results directory.

    Reads summary.json for metrics and metadata.json for environment capture.
    Only requires human-authored fields: title, description, and optional extras.
    """
    from gnomepy_research.environment import capture_environment

    results_path = Path(results_dir)
    summary_path = results_path / "summary.json"
    metadata_path = results_path / "metadata.json"

    if not summary_path.exists():
        raise click.ClickException(f"summary.json not found in {results_dir}")

    with open(summary_path) as f:
        metrics_dict = json.load(f)

    try:
        extra_metrics_dict = json.loads(extra_metrics)
    except json.JSONDecodeError as e:
        raise click.UsageError(f"--extra-metrics is not valid JSON: {e}")
    metrics_dict.update(extra_metrics_dict)

    try:
        extra_metadata_dict = json.loads(extra_metadata)
    except json.JSONDecodeError as e:
        raise click.UsageError(f"--extra-metadata is not valid JSON: {e}")

    environment_dict = capture_environment(
        metadata_json_path=metadata_path if metadata_path.exists() else None
    )

    try:
        result = api.record_iteration(
            session_name=session_name,
            iteration=iteration,
            type=iter_type,
            title=title,
            description=description,
            metrics=metrics_dict,
            metadata=extra_metadata_dict,
            environment=environment_dict,
            timestamp=timestamp,
        )
    except RuntimeError as e:
        raise click.ClickException(str(e))

    click.echo(f"recorded iteration {result.get('iteration', iteration)} for '{session_name}'")


# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------

@main.group()
def notes() -> None:
    """Manage research notes."""


@notes.command(name="list")
@click.argument("session_name")
def notes_list(session_name: str) -> None:
    """List notes for a research session."""
    try:
        note_list = api.get_notes(session_name)
    except RuntimeError as e:
        raise click.ClickException(str(e))

    if not note_list:
        click.echo("no notes")
        return

    for note in note_list:
        ts = (note.get("timestamp") or "")[:19].replace("T", " ")
        author = note.get("author", "")
        content = note.get("content", "").replace("\n", " ")[:80]
        click.echo(f"[{ts}] {author}: {content}")


@notes.command(name="add")
@click.argument("session_name")
@click.argument("content")
def notes_add(session_name: str, content: str) -> None:
    """Add a note to a research session."""
    try:
        result = api.add_note(session_name, content)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    click.echo(f"note added at {result.get('timestamp', '')}")


@notes.command(name="pull")
@click.argument("session_name")
def notes_pull(session_name: str) -> None:
    """Download API notes to local notes/ directory."""
    session_dir = Path("gnomepy_research") / "sessions" / session_name
    if not session_dir.exists():
        raise click.ClickException(f"session directory not found: {session_dir}")
    try:
        n = pull_notes(session_name, session_dir)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    click.echo(f"pulled {n} note(s) for '{session_name}'")


@notes.command(name="push")
@click.argument("session_name")
def notes_push(session_name: str) -> None:
    """Upload new local notes to API."""
    session_dir = Path("gnomepy_research") / "sessions" / session_name
    if not session_dir.exists():
        raise click.ClickException(f"session directory not found: {session_dir}")
    try:
        n = push_notes(session_name, session_dir)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    click.echo(f"pushed {n} note(s) for '{session_name}'")


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

@main.group()
def validate() -> None:
    """Significance testing and walk-forward validation."""


@validate.command(name="significance")
@click.argument("session_name")
@click.option("--fills", required=True, type=click.Path(exists=True), help="Path to fills.parquet")
@click.option("--market", required=True, type=click.Path(exists=True), help="Path to market.parquet")
@click.option("--n-trials", required=True, type=int, help="Number of iterations tested so far")
def validate_significance(session_name: str, fills: str, market: str, n_trials: int) -> None:
    """Compute bootstrap Sharpe CI and Deflated Sharpe Ratio."""
    import numpy as np
    import pandas as pd
    from gnomepy.reporting.metrics import build_curves, compute_sharpe
    from gnomepy_research.validation.statistics import bootstrap_sharpe_ci, deflated_sharpe_ratio, minimum_backtest_length

    fills_df = pd.read_parquet(fills)
    market_df = pd.read_parquet(market)
    curves = build_curves(market_df, fills_df)
    sharpe_info = compute_sharpe(curves.pnl)

    observed_sharpe = sharpe_info["sharpe"]
    n_bars = sharpe_info["n_bars"]

    bar_returns = curves.pnl.resample("10s").last().diff().dropna().values
    ci_lo, ci_hi = bootstrap_sharpe_ci(bar_returns)
    dsr = deflated_sharpe_ratio(observed_sharpe=observed_sharpe, n_trials=n_trials, n_bars=n_bars)
    min_length = minimum_backtest_length(observed_sharpe) if observed_sharpe > 0 else 0

    click.echo(f"\nSignificance test for '{session_name}'")
    click.echo(f"  Observed Sharpe:     {observed_sharpe:.4f}")
    click.echo(f"  95% Bootstrap CI:    [{ci_lo:.4f}, {ci_hi:.4f}]")
    click.echo(f"  Deflated Sharpe:     {dsr:.4f}  (p-value: {1 - dsr:.4f})")
    click.echo(f"  Significant (DSR>0.95): {'YES' if dsr > 0.95 else 'NO'}")
    click.echo(f"  CI excludes zero:    {'YES' if ci_lo > 0 else 'NO'}")
    click.echo(f"  Min bars for sig.:   {min_length} (have {n_bars})")
    click.echo(f"  Trials tested:       {n_trials}")


@validate.command(name="walk-forward")
@click.argument("session_name")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True), help="Base iteration config YAML")
@click.option("--start", "total_start", required=True, help="Total range start (ISO 8601)")
@click.option("--end", "total_end", required=True, help="Total range end (ISO 8601)")
@click.option("--folds", default=5, show_default=True, help="Number of OOS folds")
@click.option("--mode", default="expanding", show_default=True, type=click.Choice(["expanding", "rolling"]))
@click.option("--output", "output_dir", default=None, type=click.Path(), help="Output directory for fold results")
def validate_walk_forward(
    session_name: str,
    config_path: str,
    total_start: str,
    total_end: str,
    folds: int,
    mode: str,
    output_dir: str | None,
) -> None:
    """Run walk-forward validation across OOS date folds."""
    from gnomepy_research.validation.walk_forward import WalkForwardConfig, run_walk_forward_local

    try:
        start_dt = datetime.fromisoformat(total_start)
        end_dt = datetime.fromisoformat(total_end)
    except ValueError as e:
        raise click.UsageError(f"Invalid date format: {e}")

    if output_dir is None:
        output_dir = str(Path("gnomepy_research") / "sessions" / session_name / "results" / "walk_forward")

    cfg = WalkForwardConfig(
        total_start=start_dt,
        total_end=end_dt,
        n_folds=folds,
        step_mode=mode,
    )

    click.echo(f"Running {folds}-fold walk-forward for '{session_name}'...")
    try:
        result = run_walk_forward_local(
            base_config_path=config_path,
            walk_forward_config=cfg,
            output_base_dir=output_dir,
        )
    except Exception as e:
        raise click.ClickException(str(e))

    click.echo(f"\n{'FOLD':<6} {'START':<22} {'END':<22} {'PNL':>10} {'SHARPE':>8} {'FILLS':>6}")
    click.echo("-" * 78)
    for fold in result.folds:
        pnl = fold.summary.get("final_pnl", 0.0)
        sharpe = fold.summary.get("sharpe", 0.0)
        fills = fold.summary.get("fill_count", 0)
        start_str = fold.test_start.strftime("%Y-%m-%d %H:%M")
        end_str = fold.test_end.strftime("%Y-%m-%d %H:%M")
        click.echo(f"{fold.fold_index:<6} {start_str:<22} {end_str:<22} {pnl:>10.4f} {sharpe:>8.4f} {fills:>6}")

    click.echo("-" * 78)
    click.echo(f"{'Mean OOS':<6} {'':22} {'':22} {result.mean_oos_pnl:>10.4f} {result.mean_oos_sharpe:>8.4f}")
    click.echo(f"\n% positive folds: {result.pct_positive_folds * 100:.0f}%")
    verdict = "PASS" if result.mean_oos_sharpe > 0 and result.pct_positive_folds >= 0.6 else "FAIL"
    click.echo(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
