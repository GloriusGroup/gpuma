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


class TestFairchemLocalCheckpoint:
    """A config supplying only ``model_path`` must work on both paths.

    These stub out ``FairChemModel``/``pretrained_mlip`` rather than downloading
    a checkpoint: the behaviour under test is purely which config fields each
    loader insists on *before* it constructs anything, so the real model would
    add a token requirement without adding coverage.
    """

    @staticmethod
    def _checkpoint_only_config(tmp_path):
        ckpt = tmp_path / "local.pt"
        ckpt.write_bytes(b"")
        return Config({
            "model": {"model_type": "fairchem", "model_name": "", "model_path": str(ckpt)},
            "technical": {"device": "cpu"},
        })

    def test_torchsim_accepts_checkpoint_without_model_name(self, tmp_path, monkeypatch):
        """Batch mode used to demand model_name even with a checkpoint given.

        _verify_model_name_and_cache_dir raises on an empty model_name, and it
        was called unconditionally, so this config failed in batch mode while
        working in sequential mode.
        """
        import torch_sim.models.fairchem as ts_fairchem

        sentinel = object()
        seen = {}

        def fake_model(**kwargs):
            seen.update(kwargs)
            return sentinel

        monkeypatch.setattr(ts_fairchem, "FairChemModel", fake_model)

        from gpuma.models.fairchem import _load_fairchem_torchsim

        assert _load_fairchem_torchsim(self._checkpoint_only_config(tmp_path)) is sentinel
        assert seen["model"].name == "local.pt"

    def test_calculator_accepts_checkpoint_without_model_name(self, tmp_path, monkeypatch):
        """The sequential path already worked; pin it so the two stay in step."""
        import fairchem.core as fc

        sentinel = object()
        monkeypatch.setattr(
            fc.pretrained_mlip, "load_predict_unit", lambda **kw: sentinel
        )
        monkeypatch.setattr(fc, "FAIRChemCalculator", lambda **kw: sentinel)

        from gpuma.models.fairchem import _load_fairchem_calculator

        assert _load_fairchem_calculator(self._checkpoint_only_config(tmp_path)) is sentinel

    def test_model_name_still_required_without_a_checkpoint(self):
        """Dropping the check entirely would defer the error to a confusing place."""
        from gpuma.models.fairchem import _load_fairchem_torchsim

        config = Config({
            "model": {"model_type": "fairchem", "model_name": ""},
            "technical": {"device": "cpu"},
        })
        with pytest.raises(ValueError, match="Model name must be specified"):
            _load_fairchem_torchsim(config)
