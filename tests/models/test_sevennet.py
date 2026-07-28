"""Tests for SevenNet model loading.

Model-loading / inference tests require both a GPU and the optional ``sevenn``
package; they are skipped otherwise.  The registry and validation tests run
everywhere (no GPU or ``sevenn`` needed).
"""

import importlib.util

import pytest
from conftest import DEVICE, requires_gpu

from gpuma.config import Config, resolve_model_type
from gpuma.models import (
    AVAILABLE_SEVENNET_MODELS,
    load_calculator,
    load_torchsim_model,
)

from .conftest import METHANE

requires_sevenn = pytest.mark.skipif(
    importlib.util.find_spec("sevenn") is None,
    reason="sevenn not installed",
)

# A small single-modal checkpoint that needs no `modal` selector.
SEVENNET_TEST_MODEL = "7net-0"


class TestSevenNetModelType:
    """model_type alias resolution for SevenNet."""

    def test_aliases_resolve(self):
        """'sevennet' and '7net' both resolve to the canonical 'sevennet'."""
        assert resolve_model_type(Config({"model": {"model_type": "sevennet"}})) == "sevennet"
        assert resolve_model_type(Config({"model": {"model_type": "7net"}})) == "sevennet"
        assert resolve_model_type({"model": {"model_type": "SevenNet"}}) == "sevennet"


class TestSevenNetValidation:
    """Model-name validation runs before the optional sevenn import."""

    def test_invalid_model_name(self):
        """Unknown model name raises ValueError (no sevenn needed)."""
        config = Config({
            "model": {"model_type": "sevennet", "model_name": "nonexistent_model"},
            "technical": {"device": DEVICE},
        })
        with pytest.raises(ValueError, match="Unknown SevenNet model name"):
            load_calculator(config)

    def test_missing_model_name(self):
        """Empty model name raises ValueError (no sevenn needed)."""
        config = Config({
            "model": {"model_type": "sevennet", "model_name": ""},
            "technical": {"device": DEVICE},
        })
        with pytest.raises(ValueError, match="Model name must be specified"):
            load_calculator(config)


class TestSevenNetCalculator:
    """SevenNet ASE calculator loading and inference."""

    @requires_gpu
    @requires_sevenn
    def test_forward_pass(self):
        """SevenNet calculator produces a non-zero energy on methane."""
        config = Config({
            "model": {"model_type": "sevennet", "model_name": SEVENNET_TEST_MODEL},
            "technical": {"device": DEVICE},
        })
        calc = load_calculator(config)
        atoms = METHANE.copy()
        atoms.calc = calc
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert energy != 0.0


class TestSevenNetTorchsim:
    """SevenNet torch-sim model loading for batch optimization."""

    @requires_gpu
    @requires_sevenn
    def test_load(self):
        """SevenNet torch-sim model loads and reports float64 (Float64Wrapper)."""
        import torch

        config = Config({
            "model": {"model_type": "sevennet", "model_name": SEVENNET_TEST_MODEL},
            "technical": {"device": DEVICE},
        })
        model = load_torchsim_model(config)
        assert model is not None
        # gpuma's batch pipeline runs float64; SevenNet is wrapped accordingly.
        assert model.dtype == torch.float64


class TestModelRegistries:
    """SevenNet model name registry."""

    def test_sevennet_model_names_exist(self):
        """AVAILABLE_SEVENNET_MODELS contains the expected checkpoints."""
        assert "7net-0" in AVAILABLE_SEVENNET_MODELS
        assert "7net-omni" in AVAILABLE_SEVENNET_MODELS
        assert "7net-mf-ompa" in AVAILABLE_SEVENNET_MODELS
