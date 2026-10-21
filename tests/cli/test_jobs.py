"""`research jobs`: commit resolution against a real origin, and what submit sends."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from gnomepy_research import cli


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def clone(tmp_path: Path) -> tuple[Path, str]:
    """A clone whose origin has branch research/x and tag v1 at one commit, and whose local branch is ahead."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "--quiet", "-b", "main")
    _git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "--allow-empty", "--quiet", "-m", "one")
    _git(origin, "branch", "research/x")
    _git(origin, "tag", "-a", "v1", "-m", "v1")
    pushed = _git(origin, "rev-parse", "HEAD")
    work = tmp_path / "work"
    _git(tmp_path, "clone", "--quiet", str(origin), str(work))
    _git(work, "checkout", "--quiet", "research/x")
    _git(work, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "--allow-empty", "--quiet", "-m", "unpushed")
    return work, pushed


def test_a_branch_resolves_to_what_origin_has_not_the_local_tip(clone):
    work, pushed = clone
    assert cli.resolve_commit("research/x", cwd=work) == pushed
    assert cli.resolve_commit("v1", cwd=work) == pushed
    assert cli.resolve_commit("a" * 40, cwd=work) == "a" * 40


def test_a_ref_origin_lacks_is_refused(clone):
    work, _ = clone
    with pytest.raises(click.ClickException, match="push it first"):
        cli.resolve_commit("research/unpushed", cwd=work)


def test_inputs_collect_paths_by_name(tmp_path):
    (tmp_path / "a").write_text("a")
    (tmp_path / "b").write_text("b")
    assert cli.parse_inputs((f"data={tmp_path / 'a'}", f"data={tmp_path / 'b'}", f"x={tmp_path}")) == {
        "data": [tmp_path / "a", tmp_path / "b"], "x": [tmp_path]}
    for bad in ("data", "=x", f"data={tmp_path / 'missing'}"):
        with pytest.raises(click.BadParameter):
            cli.parse_inputs((bad,))


class Recorder:
    def __init__(self):
        self.uploaded: list[list[Path]] = []
        self.submitted: dict = {}

    def upload(self, paths):
        self.uploaded.append(paths)
        return "d" * 64

    def submit(self, module, commit, **kwargs):
        self.submitted = {"module": module, "commit": commit, **kwargs}
        return {"run_id": "r1", "task_count": len(kwargs["array"] or [1]), "output_prefix": "s3://b/research-jobs/runs/r1/"}


@pytest.fixture
def recorder(monkeypatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(cli, "upload_research_input", rec.upload)
    monkeypatch.setattr(cli, "submit_research_job", rec.submit)
    monkeypatch.setattr(cli, "resolve_commit", lambda ref: "c" * 40)
    return rec


def test_submit_passes_args_after_the_module_through_untouched(recorder):
    result = CliRunner().invoke(cli.main, ["jobs", "submit", "pkg.mod", "--commit", "research/x", "--cpu", "8",
                                           "--", "fit", "--seed", "3"])
    assert result.exit_code == 0, result.output
    assert recorder.submitted["args"] == ["fit", "--seed", "3"]
    assert recorder.submitted["array"] is None
    assert recorder.submitted["research_ref"] == "research/x"
    assert recorder.submitted["cpu"] == 8
    assert "submitted r1" in result.output


def test_submit_uploads_each_input_set_and_runs_one_task_per_array_entry(recorder, tmp_path):
    (tmp_path / "data.parquet").write_text("x")
    array = tmp_path / "tasks.json"
    array.write_text(json.dumps([["fit", "2025-09"], ["fit", "2025-10"]]))
    result = CliRunner().invoke(cli.main, ["jobs", "submit", "pkg.mod", "--commit", "c" * 40, "--array-file", str(array),
                                           "--input", f"data={tmp_path / 'data.parquet'}"])
    assert result.exit_code == 0, result.output
    assert recorder.uploaded == [[tmp_path / "data.parquet"]]
    assert recorder.submitted["inputs"] == {"data": "d" * 64}
    assert recorder.submitted["array"] == [["fit", "2025-09"], ["fit", "2025-10"]]
    assert recorder.submitted["args"] is None
    assert recorder.submitted["research_ref"] is None


def test_submit_refuses_args_with_an_array_file_and_malformed_array_files(recorder, tmp_path):
    array = tmp_path / "tasks.json"
    array.write_text(json.dumps([["ok"]]))
    result = CliRunner().invoke(cli.main, ["jobs", "submit", "pkg.mod", "x", "--commit", "main", "--array-file", str(array)])
    assert result.exit_code != 0 and "not both" in result.output
    array.write_text(json.dumps([["ok"], [1]]))
    result = CliRunner().invoke(cli.main, ["jobs", "submit", "pkg.mod", "--commit", "main", "--array-file", str(array)])
    assert result.exit_code != 0 and "argument lists" in result.output
