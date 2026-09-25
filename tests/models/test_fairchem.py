"""Tests for Fairchem model loading — requires HF_TOKEN."""

import pytest

from gpuma.config import Config
from gpuma.models import (
    AVAILABLE_FAIRCHEM_MODELS,
    load_calculator,
    load_torchsim_model,
)

from conftest import DEVICE, requires_gpu, requires_hf_token

from .conftest import METHANE


class TestFairchemCalculator:
    """Fairchem ASE calculator and torch-sim model loading."""

    @requires_hf_token
    @requires_gpu
    @pytest.mark.parametrize("model_name", AVAILABLE_FAIRCHEM_MODELS)
    def test_load_and_forward_pass(self, model_name):
        """Each Fairchem model loads and produces non-zero energy on methane."""
        config = Config({
            "model": {"model_type": "fairchem", "model_name": model_name},
            "technical": {"device": DEVICE},
        })
        calc = load_calculator(config)
        assert calc is not None

        atoms = METHANE.copy()
        atoms.calc = calc
        atoms.info = {"charge": 0, "spin": 1}
        energy = atoms.get_potential_energy()
        assert isinstance(energy, float)
        assert energy != 0.0

    @requires_hf_token
    @requires_gpu
    @pytest.mark.parametrize("model_name", AVAILABLE_FAIRCHEM_MODELS)
    def test_load_torchsim(self, model_name):
        """Each Fairchem torch-sim model loads successfully."""
        config = Config({
            "model": {"model_type": "fairchem", "model_name": model_name},
            "technical": {"device": DEVICE},
        })
        model = load_torchsim_model(config)
        assert model is not None


class TestFairchemD3Correction:
    """D3 dispersion correction for Fairchem/UMA models (torch-sim D3DispersionModel)."""

    @requires_hf_token
    @requires_gpu
    def test_d3_changes_energy(self):
        """Enabling D3 correction produces a different energy than without."""
        config_no_d3 = Config({
            "model": {
                "model_type": "fairchem",
                "model_name": "uma-s-1p1",
                "d3_correction": False,
            },
            "technical": {"device": DEVICE},
        })
        config_d3 = Config({
            "model": {
                "model_type": "fairchem",
                "model_name": "uma-s-1p1",
                "d3_correction": True,
                "d3_functional": "PBE",
                "d3_damping": "BJ",
            },
            "technical": {"device": DEVICE},
        })

        calc_no_d3 = load_calculator(config_no_d3)
        calc_d3 = load_calculator(config_d3)

        atoms1 = METHANE.copy()
        atoms1.calc = calc_no_d3
        atoms1.info = {"charge": 0, "spin": 1}
        e_no_d3 = atoms1.get_potential_energy()

        atoms2 = METHANE.copy()
        atoms2.calc = calc_d3
        atoms2.info = {"charge": 0, "spin": 1}
        e_d3 = atoms2.get_potential_energy()

        assert e_no_d3 != e_d3, "D3 correction should change energy"

    @requires_hf_token
    @requires_gpu
    def test_d3_torchsim_returns_sum_model(self):
        """D3-enabled UMA torch-sim model is a SumModel(uma, d3)."""
        from torch_sim.models.interface import SumModel  # type: ignore

        config = Config({
            "model": {
                "model_type": "fairchem",
                "model_name": "uma-s-1p1",
                "d3_correction": True,
                "d3_functional": "PBE",
                "d3_damping": "BJ",
            },
            "technical": {"device": DEVICE},
        })
        model = load_torchsim_model(config)
        assert isinstance(model, SumModel)
        assert len(list(model.models)) == 2

    @requires_hf_token
    @requires_gpu
    def test_d3_contribution_matches_orb_path(self):
        """The D3 energy contribution must agree across UMA and ORB paths.

        Both backends share the same ``nvalchemiops`` kernel, so the
        Δ(D3 - no D3) on a fixed geometry must be numerically equivalent
        within float-precision (the underlying neighbor-list layouts
        differ slightly between paths; we tolerate ~1e-3 eV).
        """
        d3_kwargs = {"d3_correction": True, "d3_functional": "PBE", "d3_damping": "BJ"}

        def _delta(model_type: str, model_name: str) -> float:
            cfg_off = Config({
                "model": {"model_type": model_type, "model_name": model_name},
                "technical": {"device": DEVICE},
            })
            cfg_on = Config({
                "model": {"model_type": model_type, "model_name": model_name, **d3_kwargs},
                "technical": {"device": DEVICE},
            })
            atoms_off = METHANE.copy()
            atoms_off.calc = load_calculator(cfg_off)
            atoms_off.info = {"charge": 0, "spin": 1}
            atoms_on = METHANE.copy()
            atoms_on.calc = load_calculator(cfg_on)
            atoms_on.info = {"charge": 0, "spin": 1}
            return atoms_on.get_potential_energy() - atoms_off.get_potential_energy()

        delta_uma = _delta("fairchem", "uma-s-1p1")
        delta_orb = _delta("orb", "orb_v3_direct_omol")

        assert abs(delta_uma - delta_orb) < 1e-3, (
            f"D3 contribution mismatch: UMA={delta_uma:.6f} ORB={delta_orb:.6f}"
        )
        assert delta_uma < 0, f"D3(BJ) should be attractive; got {delta_uma:+.6f}"


class TestFairchemModelRegistry:
    """Fairchem model name registry."""

    def test_fairchem_model_names_exist(self):
        """AVAILABLE_FAIRCHEM_MODELS contains all expected UMA models."""
        assert len(AVAILABLE_FAIRCHEM_MODELS) >= 3
        assert "uma-s-1p2" in AVAILABLE_FAIRCHEM_MODELS


class TestFairchemD3Caching:
    """_FairchemD3Calculator must evaluate each geometry once, not twice.

    Both the Fairchem calculator and the D3 model are stubs here: the property
    under test is how many times the wrapper calls through for a given
    geometry, which real models would only make slower to measure.
    """

    @staticmethod
    def _wrapper():
        import numpy as np
        import torch

        from gpuma.models.dispersion import _FairchemD3Calculator

        calls = {"ml": 0, "d3": 0}

        class FakeFairchem:
            def __init__(self):
                self.results = {}

            def calculate(self, atoms=None, properties=None, system_changes=None):
                calls["ml"] += 1
                self.results = {
                    "energy": float(len(atoms)),
                    "forces": np.zeros((len(atoms), 3)),
                }

        class FakeD3:
            dtype = torch.float64

            def forward(self, state):
                calls["d3"] += 1
                n = state.positions.shape[0]
                return {
                    "energy": torch.zeros(1, dtype=torch.float64),
                    "forces": torch.zeros((n, 3), dtype=torch.float64),
                }

        wrapper = _FairchemD3Calculator(FakeFairchem(), FakeD3(), torch.device("cpu"))
        return wrapper, calls

    def test_energy_then_forces_evaluates_once(self):
        """The ASE step order that used to cost two full evaluations."""
        wrapper, calls = self._wrapper()
        atoms = METHANE.copy()
        atoms.info = {"charge": 0, "spin": 1}

        wrapper.get_potential_energy(atoms)
        wrapper.get_forces(atoms)

        assert calls == {"ml": 1, "d3": 1}

    def test_moving_the_atoms_recomputes(self):
        """Caching must key on geometry, not on the Atoms object's identity.

        ASE optimizers mutate one Atoms in place, so an identity-keyed cache
        would return stale forces for every step after the first.
        """
        wrapper, calls = self._wrapper()
        atoms = METHANE.copy()
        atoms.info = {"charge": 0, "spin": 1}

        wrapper.get_potential_energy(atoms)
        atoms.positions[0][0] += 0.1
        wrapper.get_potential_energy(atoms)

        assert calls == {"ml": 2, "d3": 2}

    def test_changing_charge_recomputes(self):
        """Charge and spin feed the UMA prediction, so they invalidate too."""
        wrapper, calls = self._wrapper()
        atoms = METHANE.copy()
        atoms.info = {"charge": 0, "spin": 1}

        wrapper.get_potential_energy(atoms)
        atoms.info["charge"] = -1
        wrapper.get_potential_energy(atoms)

        assert calls == {"ml": 2, "d3": 2}

    def test_calculation_required_tracks_the_cache(self):
        """ASE consults this before deciding to call calculate()."""
        wrapper, _ = self._wrapper()
        atoms = METHANE.copy()
        atoms.info = {"charge": 0, "spin": 1}

        assert wrapper.calculation_required(atoms, ("energy",)) is True
        wrapper.get_potential_energy(atoms)
        assert wrapper.calculation_required(atoms, ("energy",)) is False
        atoms.positions[0][0] += 0.1
        assert wrapper.calculation_required(atoms, ("energy",)) is True
