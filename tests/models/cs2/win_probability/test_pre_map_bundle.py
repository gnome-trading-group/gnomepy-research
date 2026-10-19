"""The pre-map bundle file: the same predictions after a round trip, and no loading on an older xgboost."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

import gnomepy_research.models.cs2.win_probability.pre_map_model as pre_map_model
from gnomepy_research.models.cs2.win_probability.pre_map_features import PREVETO_FEATURE_NAMES
from gnomepy_research.models.cs2.win_probability.pre_map_model import (
    BUNDLE_FORMAT,
    Calibrator,
    CS2PreMapModel,
    _matrix,
    fit_calibrator,
    fit_production,
    read_bundle,
    write_bundle,
)
from gnomepy_research.models.cs2.win_probability.symmetry import build_swap_plan, symmetric_predict_proba


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(0)
    n = 1200
    df = pd.DataFrame(rng.normal(size=(n, len(PREVETO_FEATURE_NAMES))), columns=PREVETO_FEATURE_NAMES)
    df["match_id"] = np.repeat(np.arange(n // 2), 2)
    df["match_date"] = pd.Timestamp("2026-01-01") + pd.to_timedelta(df.match_id // 4, unit="D")
    df["map_name"] = ""
    df["team_a_won"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-df.elo_diff))).astype(int)
    return df


@pytest.fixture(scope="module")
def bundle(frame):
    return fit_production(frame, PREVETO_FEATURE_NAMES, n_estimators=60, max_depth=3)


def _in_memory(bundle: dict, X: np.ndarray) -> np.ndarray:
    raw = symmetric_predict_proba(bundle["model"], X, build_swap_plan(bundle["feature_names"]))
    return bundle["calibrator"].predict(raw)


def test_round_trip_predicts_identically(bundle, frame, tmp_path):
    path = tmp_path / "model.json"
    write_bundle(bundle, path)
    X = _matrix(frame, PREVETO_FEATURE_NAMES)
    np.testing.assert_array_equal(CS2PreMapModel(path).predict_batch(X), _in_memory(bundle, X))

    loaded = read_bundle(path)
    assert loaded["trained_through"] == bundle["trained_through"]
    assert loaded["n_maps"] == bundle["n_maps"] and loaded["feature_names"] == list(PREVETO_FEATURE_NAMES)


def test_isotonic_calibrator_matches_scikit_learn():
    rng = np.random.default_rng(1)
    p, y = rng.uniform(size=400), rng.integers(0, 2, size=400)
    iso = IsotonicRegression(out_of_bounds="clip").fit(np.concatenate([p, 1 - p]), np.concatenate([y, 1 - y]))
    calibrator = Calibrator.from_dict(Calibrator("isotonic", 1.0, iso.X_thresholds_, iso.y_thresholds_).to_dict())
    q = rng.uniform(-0.2, 1.2, size=1000)
    np.testing.assert_array_equal(calibrator.predict(q), np.clip(iso.predict(q), 0.01, 0.99))


def test_fit_calibrator_keeps_isotonic_thresholds():
    rng = np.random.default_rng(2)
    p = rng.uniform(size=3000)
    calibrator = fit_calibrator(p, (rng.uniform(size=3000) < p ** 2).astype(int))
    assert len(calibrator.isotonic_x) == len(calibrator.isotonic_y) > 1


def test_refuses_a_model_from_a_newer_xgboost(bundle, tmp_path, monkeypatch):
    path = tmp_path / "model.json"
    write_bundle(bundle, path)
    monkeypatch.setattr(pre_map_model.xgboost, "__version__", "3.0.0")
    with pytest.raises(ValueError, match="install xgboost >="):
        CS2PreMapModel(path)


def test_refuses_another_format(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps({"format": "something_else"}))
    with pytest.raises(ValueError, match=BUNDLE_FORMAT):
        read_bundle(path)
