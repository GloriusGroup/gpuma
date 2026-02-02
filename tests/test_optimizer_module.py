import pytest
import sys
import numpy as np
from unittest.mock import MagicMock, patch

from gpuma.optimizer import optimize_single_structure, optimize_structure_batch
from gpuma.structure import Structure
from gpuma.config import Config


def test_optimize_structure_batch_empty_returns_empty():
    assert optimize_structure_batch([]) == []


def test_optimize_structure_batch_raises_on_mismatch():
    s = Structure(["H", "H"], [(0.0, 0.0, 0.0)], charge=0, multiplicity=1)
    with pytest.raises(ValueError):
        optimize_structure_batch([s])


def test_optimize_structure_batch_raises_on_empty_structure():
    s = Structure([], [], charge=0, multiplicity=1)
    with pytest.raises(ValueError):
        optimize_structure_batch([s])


def test_optimize_single_structure_runs_bfgs():
    s = Structure(["H"], [(0.0, 0.0, 0.0)], charge=0, multiplicity=1)

    # Mock Calculator
    calc = MagicMock()

    # Mock Atoms instance
    mock_atoms_instance = MagicMock()
    mock_atoms_instance.get_positions.return_value = np.array([[1.0, 0.0, 0.0]])
    mock_atoms_instance.get_potential_energy.return_value = -13.6

    with patch("gpuma.optimizer.Atoms", return_value=mock_atoms_instance) as MockAtoms, \
         patch("gpuma.optimizer.BFGS") as MockBFGS:

         mock_bfgs_instance = MockBFGS.return_value

         res = optimize_single_structure(s, calculator=calc)

         MockAtoms.assert_called()
         MockBFGS.assert_called_with(mock_atoms_instance, logfile=None)
         mock_bfgs_instance.run.assert_called_once()

         assert res.coordinates == [[1.0, 0.0, 0.0]]
         assert res.energy == -13.6


def test_optimize_structure_batch_sequential():
    s1 = Structure(["H"], [(0,0,0)], 0, 1)
    s2 = Structure(["He"], [(1,0,0)], 0, 1)

    cfg = Config()
    cfg.optimization.batch_optimization_mode = "sequential"
    cfg.optimization.device = "cpu"

    with patch("gpuma.optimizer.optimize_single_structure") as mock_opt:
        mock_opt.side_effect = lambda s, c, calc: s.with_energy(-5.0)

        results = optimize_structure_batch([s1, s2], config=cfg)

        assert len(results) == 2
        assert results[0].energy == -5.0
        assert results[1].energy == -5.0
        assert mock_opt.call_count == 2


def test_optimize_structure_batch_batch_mode():
    s1 = Structure(["H"], [(0,0,0)], 0, 1)

    cfg = Config()
    cfg.optimization.batch_optimization_mode = "batch"
    cfg.optimization.device = "cuda"

    mock_ts = MagicMock()
    mock_ts_autobatching = MagicMock()

    # We patch both torch_sim and torch_sim.autobatching to support the import
    modules_to_patch = {
        "torch_sim": mock_ts,
        "torch_sim.autobatching": mock_ts_autobatching
    }

    with patch("gpuma.optimizer._parse_device_string", return_value="cuda"), \
         patch("gpuma.optimizer._device_for_torch") as mock_dev_fn, \
         patch("gpuma.optimizer._get_cached_torchsim_model") as mock_model_fn, \
         patch.dict(sys.modules, modules_to_patch):

         mock_dev_fn.return_value = MagicMock()
         mock_model_fn.return_value = MagicMock()

         # Mock batched state
         mock_batched_state = MagicMock()
         mock_batched_state.n_atoms = 1
         mock_ts.io.atoms_to_state.return_value = mock_batched_state

         # Mock final state
         mock_final_state = MagicMock()
         mock_ts.optimize.return_value = mock_final_state

         # Mock final atoms
         mock_final_atom = MagicMock()
         mock_final_atom.get_chemical_symbols.return_value = ["H"]
         mock_final_atom.get_positions.return_value = np.array([[0.5, 0.0, 0.0]])

         mock_final_state.to_atoms.return_value = [mock_final_atom]

         # energy/charge/spin tensors
         mock_final_state.energy = [MagicMock(item=lambda: -10.0)]
         mock_final_state.charge = [MagicMock(item=lambda: 0)]
         mock_final_state.spin = [MagicMock(item=lambda: 1)]

         results = optimize_structure_batch([s1], config=cfg)

         assert len(results) == 1
         assert results[0].energy == -10.0
         assert results[0].coordinates == [[0.5, 0.0, 0.0]]

         mock_ts.optimize.assert_called_once()
