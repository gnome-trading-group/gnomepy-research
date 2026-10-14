from __future__ import annotations

import json

import numpy as np
import pytest

from gnomepy_research.models.cfb.efficiency.data import UNITS_SPLIT
from gnomepy_research.models.cfb.efficiency.posterior_snapshots import draws_name, read_drop_features, snapshot_options
from gnomepy_research.models.cfb.features.ratings import FCS_TEAM
from gnomepy_research.sessions.cfb_win_probability.evaluation.backtest import MODEL_CONFIGS, resolve_config
from gnomepy_research.sessions.cfb_win_probability.evaluation.compare_models import PAIRS
from gnomepy_research.sessions.cfb_win_probability.evaluation.g_report import CUT_MODELS, CUT_PAIRS
from gnomepy_research.sessions.cfb_win_probability.models.win_probability import game_draws
from tests.sessions import test_cfb_win_probability_models as wpm

FEATURES = {"keep_terms": ["pass_edge", "venue"], "drop_features": ["talent_qb", "portal_in_stars"]}


def features_file(tmp_path):
    path = tmp_path / "J_features.json"
    path.write_text(json.dumps(FEATURES))
    return path


class TestSnapshotCli:
    def test_read_drop_features(self, tmp_path):
        assert read_drop_features(features_file(tmp_path)) == ("talent_qb", "portal_in_stars")

    def test_options_compose(self, tmp_path):
        opts = snapshot_options(["2020", "2024", "--likelihood", "normal", "--units", "split", "--form", "--conference",
                                 "--drives", "--drop-features-from", str(features_file(tmp_path))])
        assert opts["units"] == UNITS_SPLIT and opts["conference"] and opts["drives"] and opts["form"]
        assert opts["drop_features"] == ("talent_qb", "portal_in_stars")
        assert (opts["first"], opts["last"], opts["likelihood"]) == (2020, 2024, "normal")

    def test_options_default_has_no_drops(self):
        opts = snapshot_options(["2020", "2024"])
        assert opts["drop_features"] == () and not opts["drives"]


def test_game_draws_loads_drives_and_drop_name(tmp_path):
    drop = ("talent_qb", "portal_in_stars")
    wpm.TestSplitGameDraws.write(tmp_path, [1, 2, FCS_TEAM], conference=True, option=[0, 1, 0])
    base = tmp_path / f"{draws_name(2024, UNITS_SPLIT, True)}.npz"
    base.rename(tmp_path / f"{draws_name(2024, UNITS_SPLIT, True, drop_features=drop, drives=True)}.npz")
    g = wpm.TestSplitGameDraws.games()
    gd = game_draws(g, [2024], directory=tmp_path, units=UNITS_SPLIT, conference=True, option=True,
                    drop_features=drop, drives=True)
    assert gd.x.shape[0] == 2
    with pytest.raises(FileNotFoundError):
        game_draws(g, [2024], directory=tmp_path, units=UNITS_SPLIT, conference=True, option=True, drop_features=drop)


class TestJConfigs:
    def test_resolve_reads_features_at_runtime(self, tmp_path):
        path = features_file(tmp_path)
        j0, j = resolve_config("J0", path), resolve_config("J", path)
        g = MODEL_CONFIGS["G"]
        for cfg in (j0, j):
            assert all(cfg[k] == g[k] for k in g)
            assert cfg["keep_terms"] == FEATURES["keep_terms"]
            assert cfg["drop_features"] == tuple(FEATURES["drop_features"])
        assert not j0.get("drives") and j["drives"]

    def test_missing_features_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_config("J", tmp_path / "nope.json")

    def test_other_models_untouched(self, tmp_path):
        assert resolve_config("G", tmp_path / "nope.json") == MODEL_CONFIGS["G"]


def test_report_pairs():
    assert {("J0", "G"), ("J", "G"), ("J", "J0")} <= set(PAIRS)
    assert CUT_MODELS["J"] == ["G", "J0", "J"]
    assert CUT_PAIRS["J"] == [("J", "G"), ("J", "J0")]
