"""Tests for AIMNet2 model loading and its input checks.

Model-loading / inference tests require a GPU; the registry, validation and
input-check tests run everywhere.
"""

import os

import numpy as np
import pytest
import torch
import torch_sim
from ase import Atoms
from conftest import DEVICE, requires_gpu
from torch_sim.typing import SystemExtras

from gpuma.config import Config, resolve_model_type
from gpuma.models import (
    AVAILABLE_AIMNET_MODELS,
    check_structure_supported,
    load_calculator,
    load_torchsim_model,
)
from gpuma.models.base import AIMNET_ELEMENTS

from .conftest import METHANE


def _config(model_name="aimnet2", **model):
    return Config(
        {
            "model": {"model_type": "aimnet", "model_name": model_name, **model},
            "technical": {"device": DEVICE},
        }
    )


def _methyl():
    """Planar CH3 radical."""
    return Atoms("CH3", positions=[[0, 0, 0], [1.08, 0, 0], [-0.54, 0.935, 0], [-0.54, -0.935, 0]])


class TestAIMNetModelType:
    def test_aliases_resolve(self):
        assert resolve_model_type(Config({"model": {"model_type": "aimnet"}})) == "aimnet"
        assert resolve_model_type({"model": {"model_type": "AIMNet2"}}) == "aimnet"


class TestAIMNetValidation:
    def test_invalid_model_name(self):
        with pytest.raises(ValueError, match="Unknown AIMNet model name"):
            load_calculator(_config("aimnet2-rxn"))

    def test_missing_model_name(self):
        with pytest.raises(ValueError, match="Model name must be specified"):
            load_torchsim_model(_config(""))

    def test_d3_rejected(self):
        """D3 is part of every AIMNet2 model; adding it counts dispersion twice."""
        with pytest.raises(ValueError, match="already includes dispersion"):
            _config(d3_correction=True)

    def test_model_path_rejected(self, tmp_path):
        ckpt = tmp_path / "model.pt"
        ckpt.write_bytes(b"")
        with pytest.raises(ValueError, match="model_path is not supported"):
            _config(model_path=str(ckpt))


def _structure(symbols, coordinates, multiplicity=1):
    return {"symbols": list(symbols), "coordinates": coordinates, "multiplicity": multiplicity}


def _check(config, structure):
    check_structure_supported(config, **structure)


_CH4 = _structure(METHANE.get_chemical_symbols(), METHANE.positions.tolist())
_CH3 = _structure(["C", "H", "H", "H"], _methyl().positions.tolist(), multiplicity=2)
# CH2 group with its hydrogens 0.7 Å apart: AIMNet2's spurious collapse region.
_SQUEEZED_CH4 = _structure(
    ["C", "H", "H", "H", "H"],
    [[0, 0, 0], [0.35, 1.0, 0], [-0.35, 1.0, 0], [0, -0.63, 0.9], [0, -0.63, -0.9]],
)
_H2 = _structure(["H", "H"], [[0, 0, 0], [0, 0, 0.74]])


class TestStructureSupported:
    def test_other_backends_unchecked(self):
        orb = Config({"model": {"model_type": "orb", "model_name": "orb_v3_direct_omol"}})
        _check(orb, _structure(["Fe"], [[0, 0, 0]], multiplicity=5))

    def test_unknown_model_name_first(self):
        """A non-AIMNet name (e.g. the UMA default) is named as the problem, not the input."""
        with pytest.raises(ValueError, match="Unknown AIMNet model name 'uma-s-1p2p1-ft-d4rev'"):
            _check(_config("uma-s-1p2p1-ft-d4rev"), _CH3)

    def test_supported_structure_passes(self):
        _check(_config(), _CH4)

    def test_unsupported_element(self):
        fe = _structure(["Fe", "C", "O"], [[0, 0, 0], [1.8, 0, 0], [2.95, 0, 0]])
        with pytest.raises(ValueError, match="Fe.*not supported by aimnet2"):
            _check(_config(), fe)

    @pytest.mark.parametrize("model_name", ["aimnet2", "aimnet2-2025"])
    def test_closed_shell_models_reject_open_shell(self, model_name):
        with pytest.raises(ValueError, match="closed-shell.*aimnet2-nse"):
            _check(_config(model_name), _CH3)

    def test_nse_accepts_open_shell(self):
        _check(_config("aimnet2-nse"), _CH3)

    @pytest.mark.parametrize("model_name", ["aimnet2", "aimnet2-2025"])
    def test_short_hh_contact_rejected(self, model_name):
        with pytest.raises(ValueError, match="H–H distance 0.70 Å"):
            _check(_config(model_name), _SQUEEZED_CH4)

    def test_nse_has_no_hh_defect(self):
        _check(_config("aimnet2-nse"), _SQUEEZED_CH4)

    def test_hh_check_needs_coordinates(self):
        """SMILES inputs are checked before embedding, without geometry."""
        check_structure_supported(_config(), ["C", "H", "H", "H", "H"], 1)

    def test_free_h2_accepted(self):
        """Upstream describes free H2 correctly; only hydrogens on heavy atoms collapse."""
        _check(_config(), _H2)


@requires_gpu
class TestAIMNetModels:
    @pytest.mark.parametrize("model_name", AVAILABLE_AIMNET_MODELS)
    def test_elements_match_model(self, model_name):
        """gpuma's element list (checked before loading) matches each model's own."""
        from ase.data import chemical_symbols

        calc = load_calculator(_config(model_name))
        species = calc.base_calc.metadata["implemented_species"]
        assert {chemical_symbols[z] for z in species} == set(AIMNET_ELEMENTS)
        assert calc.base_calc.is_nse == (model_name == "aimnet2-nse")

    def test_forward_pass(self):
        atoms = METHANE.copy()
        atoms.calc = load_calculator(_config())
        energy = atoms.get_potential_energy()
        assert np.isfinite(energy) and energy < 0.0
        assert atoms.get_forces().shape == (5, 3)

    def test_nse_uses_multiplicity(self):
        """The NSE model reads the multiplicity from atoms.info like gpuma sets it."""
        calc = load_calculator(_config("aimnet2-nse"))
        energies = []
        for mult in (2, 4):
            atoms = _methyl()
            atoms.info = {"charge": 0, "spin": mult}
            atoms.calc = calc
            energies.append(atoms.get_potential_energy())
        assert energies[0] != pytest.approx(energies[1], abs=1e-3)

    def test_torchsim_float64_matches_ase(self):
        """The batch model reports float64 and agrees with the ASE calculator."""
        model = load_torchsim_model(_config())
        assert model.dtype == torch.float64
        atoms = METHANE.copy()
        atoms.info = {"charge": 0, "spin": 1}
        state = torch_sim.io.atoms_to_state(
            [atoms],
            device=torch.device(DEVICE),
            dtype=torch.float64,
            system_extras_map={SystemExtras.CHARGE: "charge", SystemExtras.SPIN: "spin"},
        )
        out = model(state)
        assert out["energy"].dtype == out["forces"].dtype == torch.float64
        atoms.calc = load_calculator(_config())
        assert out["energy"].item() == pytest.approx(atoms.get_potential_energy(), abs=1e-4)

    def test_torchsim_per_system_charge(self):
        """Each system in one batch keeps its own charge."""
        model = load_torchsim_model(_config())
        nh3 = Atoms(
            "NH3",
            positions=[[0, 0, 0], [0.94, 0, -0.38], [-0.47, 0.81, -0.38], [-0.47, -0.81, -0.38]],
        )
        nh4 = Atoms(
            "NH4",
            positions=[
                [0, 0, 0],
                [0.59, 0.59, 0.59],
                [-0.59, -0.59, 0.59],
                [-0.59, 0.59, -0.59],
                [0.59, -0.59, -0.59],
            ],
        )
        nh3.info, nh4.info = {"charge": 0, "spin": 1}, {"charge": 1, "spin": 1}

        def energies(systems):
            state = torch_sim.io.atoms_to_state(
                systems,
                device=torch.device(DEVICE),
                dtype=torch.float64,
                system_extras_map={SystemExtras.CHARGE: "charge", SystemExtras.SPIN: "spin"},
            )
            return model(state)["energy"].tolist()

        both = energies([nh3, nh4])
        assert both == pytest.approx(energies([nh3]) + energies([nh4]), abs=1e-4)

    def test_model_cache_dir(self, tmp_path, monkeypatch):
        """model_cache_dir is where AIMNet downloads its weights."""
        monkeypatch.setenv("AIMNET_CACHE_DIR", str(tmp_path / "unused"))
        load_calculator(_config(model_cache_dir=str(tmp_path)))
        assert os.environ["AIMNET_CACHE_DIR"] == str(tmp_path)
        assert any(tmp_path.iterdir())


@requires_gpu
def test_batch_and_sequential_agree_on_mixed_charges():
    """torch-sim batches and the cached ASE calculator give the same minima per charge."""
    from gpuma.optimizer import optimize_structure_batch
    from gpuma.structure import Structure

    def structures():
        return [
            Structure(
                ["N", "H", "H", "H"],
                [(0, 0, 0), (0.94, 0, -0.38), (-0.47, 0.81, -0.38), (-0.47, -0.81, -0.38)],
                charge=0,
                multiplicity=1,
            ),
            Structure(
                ["N", "H", "H", "H", "H"],
                [
                    (0, 0, 0),
                    (0.59, 0.59, 0.59),
                    (-0.59, -0.59, 0.59),
                    (-0.59, 0.59, -0.59),
                    (0.59, -0.59, -0.59),
                ],
                charge=1,
                multiplicity=1,
            ),
        ]

    energies = {}
    for mode in ("batch", "sequential"):
        config = _config()
        config.optimization.batch_optimization_mode = mode
        config.optimization.force_convergence_criterion = 1e-3
        energies[mode] = [s.energy for s in optimize_structure_batch(structures(), config)]
    assert energies["batch"] == pytest.approx(energies["sequential"], abs=1e-3)
