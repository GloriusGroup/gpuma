"""Tests for choosing the autobatcher memory metric per backend."""

import pytest

from gpuma.config import Config
from gpuma.optimizer import _memory_scales_with


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [("fairchem", "n_atoms"), ("uma", "n_atoms"), ("orb", "n_edges"), ("sevennet", "n_edges")],
)
def test_auto_picks_metric_by_backend(model_type, expected):
    cfg = Config({"model": {"model_type": model_type}})
    assert _memory_scales_with(cfg) == expected


@pytest.mark.parametrize("explicit", ["n_atoms", "n_edges"])
def test_explicit_metric_overrides_backend(explicit):
    cfg = Config({
        "model": {"model_type": "fairchem"},
        "technical": {"memory_scales_with": explicit},
    })
    assert _memory_scales_with(cfg) == explicit


def test_value_set_after_construction_is_checked():
    cfg = Config({"model": {"model_type": "orb"}})
    cfg.technical.memory_scales_with = "N_ATOMS"
    assert _memory_scales_with(cfg) == "n_atoms"
    cfg.technical.memory_scales_with = "n_atoms_x_density"
    with pytest.raises(ValueError, match="memory_scales_with"):
        _memory_scales_with(cfg)
