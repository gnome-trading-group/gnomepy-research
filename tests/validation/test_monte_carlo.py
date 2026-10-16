"""Tests for the sensitivity sweep generators.

These were unreachable for months and had drifted to the pre-`sweep:` config format,
emitting configs the engine silently refused to expand.
"""
from __future__ import annotations

import yaml
import pytest
from gnomepy.sweep import expand_sweep

from gnomepy_research.validation.monte_carlo import (
    generate_latency_sweep_config,
    generate_queue_sweep_config,
)

BASE = {
    "strategy": {"class_name": "x:Y", "args": {"gamma": 0.5}},
    "start_date": "2026-08-24T00:25:00",
    "end_date": "2026-08-24T03:11:00",
    "listings": [{"listing_id": 1}],
    "exchanges": {
        "KALSHI": {
            "fee_model": {"type": "static", "taker_fee": 0.0, "maker_fee": 0.0},
            "network_latency": {"type": "static", "latency_nanos": 5_000_000},
            "queue_model": {"type": "risk_averse"},
        },
        "POLYMARKET_INTL": {
            "fee_model": {"type": "static", "taker_fee": 0.0, "maker_fee": 0.0},
            "network_latency": {"type": "static", "latency_nanos": 20_000_000},
            "queue_model": {"type": "risk_averse"},
        },
    },
}


@pytest.fixture
def base_config(tmp_path):
    path = tmp_path / "iter_001.yaml"
    path.write_text(yaml.safe_dump(BASE))
    return path


def _load(path):
    return yaml.safe_load(path.read_text())


def test_latency_sweep_is_expandable(base_config, tmp_path):
    cfg = _load(generate_latency_sweep_config(base_config, tmp_path / "s.yaml", [1_000_000, 2_000_000]))
    assert "sweep" in cfg, "values must live under the top-level sweep section to be expanded"
    assert len(expand_sweep(cfg)) == 4  # 2 values x 2 exchanges


def test_latency_sweep_scoped_to_one_exchange(base_config, tmp_path):
    cfg = _load(generate_latency_sweep_config(
        base_config, tmp_path / "s.yaml", [1_000_000, 2_000_000, 3_000_000], exchange_codes=["KALSHI"],
    ))
    assert set(cfg["sweep"]["exchanges"]) == {"KALSHI"}
    assert len(expand_sweep(cfg)) == 3


def test_latency_sweep_leaves_a_scalar_default(base_config, tmp_path):
    """The config must stay runnable on its own, so exchanges keeps a scalar."""
    cfg = _load(generate_latency_sweep_config(base_config, tmp_path / "s.yaml", [7_000_000, 9_000_000]))
    for overrides in cfg["exchanges"].values():
        assert isinstance(overrides["network_latency"]["latency_nanos"], int)


def test_queue_sweep_switches_model_and_expands(base_config, tmp_path):
    cfg = _load(generate_queue_sweep_config(base_config, tmp_path / "q.yaml", [0.2, 0.8], exchange_codes=["KALSHI"]))
    assert cfg["exchanges"]["KALSHI"]["queue_model"]["type"] == "probabilistic"
    assert isinstance(cfg["exchanges"]["KALSHI"]["queue_model"]["cancel_ahead_probability"], float)
    assert len(expand_sweep(cfg)) == 2


def test_generators_do_not_mutate_the_base_config(base_config, tmp_path):
    before = base_config.read_text()
    generate_latency_sweep_config(base_config, tmp_path / "a.yaml")
    generate_queue_sweep_config(base_config, tmp_path / "b.yaml")
    assert base_config.read_text() == before


def test_an_exchange_the_config_does_not_override_yet_can_be_varied(tmp_path):
    path = tmp_path / "plain.yaml"
    path.write_text(yaml.safe_dump({**{k: v for k, v in BASE.items() if k != "exchanges"}}))
    with pytest.raises(ValueError, match="name the ones to vary"):
        generate_latency_sweep_config(path, tmp_path / "x.yaml")
    cfg = _load(generate_latency_sweep_config(path, tmp_path / "s.yaml", [1, 2], exchange_codes=["KALSHI"]))
    assert cfg["exchanges"] == {"KALSHI": {"network_latency": {"type": "static", "latency_nanos": 1}}}
    assert len(expand_sweep(cfg)) == 2
