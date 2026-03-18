"""Slow integration tests that load and instantiate every available model.

These tests are **not** run by default.  Execute them explicitly with::

    pytest -m slow tests/test_model_loading_slow.py

Each test downloads (if not cached) and instantiates the model on the
available device, then runs a minimal forward pass on a small molecule
(methane) to verify correctness.

GPU memory is explicitly freed between parametrized tests to avoid OOM
errors when loading many models sequentially.
"""

import gc
import os

import pytest
import torch
from ase import Atoms

from gpuma.config import Config
from gpuma.models import (
    AVAILABLE_FAIRCHEM_MODELS,
    AVAILABLE_ORB_MODELS,
    load_calculator,
    load_torchsim_model,
)

# All tests in this module require real model loading (no mocks) and are slow.
pytestmark = [pytest.mark.slow, pytest.mark.real_model]

# Use GPU if available, otherwise CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A small molecule for quick forward-pass validation.
METHANE = Atoms(
    symbols=["C", "H", "H", "H", "H"],
    positions=[
        [0.000, 0.000, 0.000],
        [0.629, 0.629, 0.629],
        [-0.629, -0.629, 0.629],
        [-0.629, 0.629, -0.629],
        [0.629, -0.629, -0.629],
    ],
)


def _has_hf_token() -> bool:
    """Check whether a HuggingFace token is available."""
    return bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN"))


@pytest.fixture(autouse=True)
def _cleanup_gpu_memory():
    """Free GPU memory after each test to avoid OOM when loading many models."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Fairchem — ASE calculator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", AVAILABLE_FAIRCHEM_MODELS)
def test_load_fairchem_calculator(model_name: str):
    """Load a Fairchem calculator and run a single-point energy evaluation."""
    if not _has_hf_token():
        pytest.skip("HF_TOKEN not set — required for Fairchem model download")

    config = Config({
        "optimization": {
            "model_type": "fairchem",
            "model_name": model_name,
            "device": DEVICE,
        }
    })

    calc = load_calculator(config)
    assert calc is not None

    atoms = METHANE.copy()
    atoms.calc = calc
    atoms.info = {"charge": 0, "spin": 1}

    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)
    assert energy != 0.0, f"Energy is exactly 0.0 for model {model_name}"


# ---------------------------------------------------------------------------
# Fairchem — torch-sim model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", AVAILABLE_FAIRCHEM_MODELS)
def test_load_fairchem_torchsim(model_name: str):
    """Load a Fairchem torch-sim model and verify it is not None."""
    if not _has_hf_token():
        pytest.skip("HF_TOKEN not set — required for Fairchem model download")

    config = Config({
        "optimization": {
            "model_type": "fairchem",
            "model_name": model_name,
            "device": DEVICE,
        }
    })

    model = load_torchsim_model(config)
    assert model is not None


# ---------------------------------------------------------------------------
# ORB-v3 — ASE calculator
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", AVAILABLE_ORB_MODELS)
def test_load_orb_calculator(model_name: str):
    """Load an ORB calculator and run a single-point energy evaluation."""
    config = Config({
        "optimization": {
            "model_type": "orb",
            "model_name": model_name,
            "device": DEVICE,
        }
    })

    calc = load_calculator(config)
    assert calc is not None

    atoms = METHANE.copy()
    atoms.calc = calc
    atoms.info = {"charge": 0, "spin": 1}

    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)
    assert energy != 0.0, f"Energy is exactly 0.0 for model {model_name}"


# ---------------------------------------------------------------------------
# ORB-v3 — torch-sim model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", AVAILABLE_ORB_MODELS)
def test_load_orb_torchsim(model_name: str):
    """Load an ORB torch-sim model and verify it is not None."""
    config = Config({
        "optimization": {
            "model_type": "orb",
            "model_name": model_name,
            "device": DEVICE,
        }
    })

    model = load_torchsim_model(config)
    assert model is not None


# ---------------------------------------------------------------------------
# ORB-v3 with D3 dispersion correction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", ["orb_v3_direct_omol", "orb_v3_conservative_omol"])
def test_load_orb_calculator_with_d3(model_name: str):
    """Load an ORB calculator with D3 correction and verify energy output."""
    config = Config({
        "optimization": {
            "model_type": "orb",
            "model_name": model_name,
            "device": DEVICE,
            "d3_correction": True,
            "d3_functional": "PBE",
            "d3_damping": "BJ",
        }
    })

    calc = load_calculator(config)
    assert calc is not None

    atoms = METHANE.copy()
    atoms.calc = calc
    atoms.info = {"charge": 0, "spin": 1}

    energy = atoms.get_potential_energy()
    assert isinstance(energy, float)
    assert energy != 0.0, f"Energy is exactly 0.0 for model {model_name} with D3"
