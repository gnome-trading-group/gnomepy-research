"""Command-line interface for gnomepy-research."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

import click
import yaml

from gnomepy_research import api
from gnomepy_research.notes_sync import pull_notes, push_notes


def _session_dir(session_name: str) -> Path:
    """Locate a session directory, following the worktree /research-branch created for it.

    Branch sessions live only inside their own worktree, so the path relative to the
    current repo root does not exist when invoked from anywhere else.
    """
    local = Path("gnomepy_research") / "sessions" / session_name
    if local.exists():
        return local

    branch = f"research/{session_name}"
    try:
        out = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        raise click.ClickException(f"session directory not found: {local}")

    worktree = None
    for line in out.splitlines():
        if line.startswith("worktree "):
            worktree = line[len("worktree "):]
        elif line.startswith("branch ") and worktree:
            if line[len("branch "):] == f"refs/heads/{branch}":
                candidate = Path(worktree) / "gnomepy_research" / "sessions" / session_name
                if candidate.exists():
                    return candidate

    raise click.ClickException(
        f"session directory not found: {local} (and no worktree for branch '{branch}')"
    )


def _flag(value: object) -> str:
    """Render a tri-state boolean from iteration metadata, which may be absent."""
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "\u2014"


def _num(value: object, places: int = 4) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.{places}f}"
    return "\u2014"


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
@click.option("--json", "as_json", is_flag=True, help="Emit the raw API response as JSON")
def sessions_list(status: str | None, limit: int, as_json: bool) -> None:
    """List research sessions."""
    try:
        result = api.list_sessions(status=status, limit=limit)
    except RuntimeError as e:
        raise click.ClickException(str(e))

    if as_json:
        click.echo(json.dumps(result, indent=2))
        return

    sessions_data = result.get("sessions", [])
    if not sessions_data:
        click.echo("no sessions found")
        return

    header = f"{'SESSION':<38} {'STATUS':<12} {'ITERS':>5} {'BEST PNL':>10} {'BEST SHARPE':>12} {'TAGS':<26} {'UPDATED'}"
    click.echo(header)
    click.echo("-" * len(header))
    for s in sessions_data:
        name = (s.get("session_name") or "")[:38]
        status_val = s.get("status") or ""
        iters = s.get("iteration_count") or 0
        best_pnl = s.get("best_pnl")
        best_sharpe = s.get("best_sharpe")
        tags = ",".join(s.get("tags") or [])[:26]
        updated = (s.get("updated_at") or "")[:19].replace("T", " ")
        pnl_str = f"{best_pnl:.4f}" if best_pnl is not None else "—"
        sharpe_str = f"{best_sharpe:.4f}" if best_sharpe is not None else "—"
        click.echo(f"{name:<38} {status_val:<12} {iters:>5} {pnl_str:>10} {sharpe_str:>12} {tags:<26} {updated}")


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
@click.option("--best-metric", default=None, help="Name of the session's primary metric (e.g. sortino)")
@click.option("--best-metric-value", type=float, default=None, help="Best accepted value of --best-metric")
def sessions_update(
    session_name: str,
    status: str | None,
    description: str | None,
    tags: str | None,
    best_iteration: int | None,
    best_pnl: float | None,
    best_sharpe: float | None,
    best_metric: str | None,
    best_metric_value: float | None,
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
    if (best_metric is None) != (best_metric_value is None):
        raise click.UsageError("--best-metric and --best-metric-value must be given together")
    if best_metric is not None:
        fields["best_metric"] = best_metric
        fields["best_metric_value"] = best_metric_value

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


@iterations.command(name="list")
@click.argument("session_name")
@click.option("--limit", default=10, show_default=True, help="Show the most recent N iterations")
@click.option("--json", "as_json", is_flag=True, help="Emit the raw iteration records as JSON")
def iterations_list(session_name: str, limit: int, as_json: bool) -> None:
    """List recorded iterations for a session, most recent last."""
    try:
        session = api.get_session(session_name)
    except RuntimeError as e:
        raise click.ClickException(str(e))

    records = sorted(session.get("iterations") or [], key=lambda r: r.get("iteration", 0))
    if limit > 0:
        records = records[-limit:]

    if as_json:
        click.echo(json.dumps(records, indent=2))
        return

    if not records:
        click.echo(f"no iterations recorded for '{session_name}'")
        return

    header = f"{'ITER':>5} {'TYPE':<8} {'ACC':<4} {'THR':<4} {'PNL':>12} {'SHARPE':>10} {'FILLS':>7}  TITLE"
    click.echo(header)
    click.echo("-" * len(header))
    for r in records:
        metrics = r.get("metrics") or {}
        metadata = r.get("metadata") or {}
        accepted = metadata.get("accepted")
        thresholds = metadata.get("thresholds_met")
        click.echo(
            f"{r.get('iteration', 0):>5} "
            f"{(r.get('type') or ''):<8} "
            f"{_flag(accepted):<4} "
            f"{_flag(thresholds):<4} "
            f"{_num(metrics.get('final_pnl')):>12} "
            f"{_num(metrics.get('sharpe')):>10} "
            f"{_num(metrics.get('fill_count'), 0):>7}  "
            f"{(r.get('title') or '')[:60]}"
        )


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
    session_dir = _session_dir(session_name)
    try:
        n = pull_notes(session_name, session_dir)
    except RuntimeError as e:
        raise click.ClickException(str(e))
    click.echo(f"pulled {n} note(s) for '{session_name}'")


@notes.command(name="push")
@click.argument("session_name")
def notes_push(session_name: str) -> None:
    """Upload new local notes to API."""
    session_dir = _session_dir(session_name)
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
@click.option("--json", "as_json", is_flag=True, help="Emit results as JSON")
def validate_significance(session_name: str, fills: str, market: str, n_trials: int, as_json: bool) -> None:
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

    if as_json:
        click.echo(json.dumps({
            "session_name": session_name,
            "observed_sharpe": observed_sharpe,
            "sharpe_ci_95": [ci_lo, ci_hi],
            "deflated_sharpe": dsr,
            "dsr_significant": bool(dsr > 0.95),
            "ci_excludes_zero": bool(ci_lo > 0),
            "min_bars_for_significance": min_length,
            "n_bars": n_bars,
            "n_trials": n_trials,
        }, indent=2))
        return

    click.echo(f"\nSignificance test for '{session_name}'")
    click.echo(f"  Observed Sharpe:     {observed_sharpe:.4f}  (per-{sharpe_info.get('bar', '10s')} bar)")
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
@click.option("--mode", default="rolling", show_default=True, type=click.Choice(["expanding", "rolling"]),
              help="rolling: disjoint windows marching forward. expanding: windows anchored at --start that grow.")
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


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------

@main.group()
def artifacts() -> None:
    """Manage versioned model artifacts."""


@artifacts.command(name="list")
@click.option("--type", "artifact_type", default=None, help="Filter by artifact type")
@click.option("--name", default=None, help="Filter by artifact name")
@click.option("--session", default=None, help="Filter by session name")
def artifacts_list(artifact_type: str | None, name: str | None, session: str | None) -> None:
    """List artifacts in the store."""
    from gnomepy_research.artifacts import ArtifactStore
    store = ArtifactStore()
    try:
        refs = store.list(artifact_type=artifact_type, name=name, session_name=session)
    except Exception as e:
        raise click.ClickException(str(e))

    if not refs:
        click.echo("no artifacts found")
        return

    refs.sort(key=lambda r: (r.artifact_type, r.name, r.version))
    header = f"{'TYPE':<24} {'NAME':<30} {'VER':>4} {'SESSION':<24} {'S3 URI'}"
    click.echo(header)
    click.echo("-" * len(header))
    for r in refs:
        click.echo(f"{r.artifact_type:<24} {r.name:<30} {r.version:>4} {r.session_name:<24} {r.s3_uri}")


@artifacts.command(name="publish")
@click.argument("local_path", type=click.Path(exists=True))
@click.option("--type", "artifact_type", required=True, help="Artifact type (e.g. xgboost_model)")
@click.option("--name", required=True, help="Artifact name")
@click.option("--session", default="__global__", show_default=True, help="Owning session name")
@click.option("--description", default="", help="Human-readable description")
@click.option("--params", default="{}", help="JSON dict of hyperparameters or config")
def artifacts_publish(
    local_path: str,
    artifact_type: str,
    name: str,
    session: str,
    description: str,
    params: str,
) -> None:
    """Upload a local file as a new artifact version."""
    import json
    from gnomepy_research.artifacts import ArtifactStore
    try:
        params_dict = json.loads(params)
    except json.JSONDecodeError as e:
        raise click.UsageError(f"--params is not valid JSON: {e}")

    store = ArtifactStore()
    try:
        ref = store.publish(
            local_path,
            artifact_type=artifact_type,
            name=name,
            session_name=session,
            description=description,
            params=params_dict or None,
        )
    except Exception as e:
        raise click.ClickException(str(e))

    click.echo(f"published {ref}")


@artifacts.command(name="get")
@click.argument("ref")
@click.option("--output", "-o", default=None, type=click.Path(), help="Destination path (default: current dir)")
def artifacts_get(ref: str, output: str | None) -> None:
    """Download an artifact to a local file.

    REF format: type/name[:version]  or  artifact://type/name[:version]
    """
    import shutil
    from gnomepy_research.artifacts import ArtifactStore

    if not ref.startswith("artifact://"):
        ref = f"artifact://{ref}"

    store = ArtifactStore()
    try:
        cached = store.resolve(ref)
    except Exception as e:
        raise click.ClickException(str(e))

    if output:
        shutil.copy2(cached, output)
        click.echo(f"saved to {output}")
    else:
        import os
        dest = os.path.join(".", os.path.basename(cached))
        shutil.copy2(cached, dest)
        click.echo(f"saved to {dest}")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

@main.group()
def datasets() -> None:
    """Manage shared training datasets."""


@datasets.command(name="list")
@click.option("--name", default=None, help="Filter by dataset name")
def datasets_list(name: str | None) -> None:
    """List datasets in the store."""
    from gnomepy_research.artifacts import DatasetStore
    store = DatasetStore()
    try:
        refs = store.list(name=name)
    except Exception as e:
        raise click.ClickException(str(e))

    if not refs:
        click.echo("no datasets found")
        return

    refs.sort(key=lambda r: (r.dataset_name, r.version))
    header = f"{'NAME':<36} {'VER':>4} {'ROWS':>8}  {'S3 URI'}"
    click.echo(header)
    click.echo("-" * len(header))
    for r in refs:
        click.echo(f"{r.dataset_name:<36} {r.version:>4} {'—':>8}  {r.s3_uri}")


@datasets.command(name="publish")
@click.argument("parquet_path", type=click.Path(exists=True))
@click.option("--name", required=True, help="Dataset name")
@click.option("--description", default="", help="Human-readable description")
@click.option("--session", default=None, help="Producing session name")
def datasets_publish(
    parquet_path: str,
    name: str,
    description: str,
    session: str | None,
) -> None:
    """Publish a Parquet file as a new dataset version."""
    import pandas as pd
    from gnomepy_research.artifacts import DatasetStore

    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e:
        raise click.ClickException(f"failed to read parquet: {e}")

    store = DatasetStore()
    try:
        ref = store.publish(df, name=name, description=description, producing_session=session)
    except Exception as e:
        raise click.ClickException(str(e))

    click.echo(f"published {ref}  ({len(df)} rows)")


@datasets.command(name="get")
@click.argument("ref")
@click.option("--output", "-o", default=None, type=click.Path(), help="Destination path (default: <name>_v<version>.parquet)")
def datasets_get(ref: str, output: str | None) -> None:
    """Download a dataset to a local Parquet file.

    REF format: name  or  name:version
    """
    from gnomepy_research.artifacts import DatasetStore

    store = DatasetStore()
    try:
        dataset_ref = store.latest(ref) if ":" not in ref else None
        if dataset_ref is None:
            df = store.load(ref)
            name, ver_str = ref.rsplit(":", 1)
            dest = output or f"{name}_v{ver_str}.parquet"
        else:
            df = store.load(dataset_ref)
            dest = output or f"{dataset_ref.dataset_name}_v{dataset_ref.version}.parquet"
    except Exception as e:
        raise click.ClickException(str(e))

    df.to_parquet(dest, index=False)
    click.echo(f"saved {len(df)} rows to {dest}")


if __name__ == "__main__":
    main()
