"""Tests for the chunked n_edges memory scalers. Runs on CPU."""

import pytest
import torch
import torch_sim
import torch_sim.autobatching as ts_autobatching
from ase.build import molecule

from gpuma.autobatching import (
    _chunk_bounds,
    _torch_sim_n_edges_scalers,
    install_chunked_edge_scalers,
    n_edges_scalers_chunked,
)

CUTOFF = 6.0


@pytest.fixture
def mixed_state():
    """Seven molecules of unequal size, so chunk edges never fall on equal counts."""
    names = ["H2O", "CH4", "C6H6", "C2H6", "NH3", "CH3CH2OH", "C60"]
    atoms = [molecule(n) for n in names]
    for a in atoms:
        a.info["charge"] = 0
        a.info["spin"] = 1
    return torch_sim.io.atoms_to_state(atoms, device=torch.device("cpu"), dtype=torch.float64)


class TestChunkBounds:
    def test_single_chunk_when_under_budget(self):
        assert _chunk_bounds([3, 4, 5], 100) == [(0, 3, 0, 12)]

    def test_splits_at_system_boundaries(self):
        assert _chunk_bounds([3, 4, 5, 2], 7) == [(0, 2, 0, 7), (2, 4, 7, 14)]
        assert _chunk_bounds([3, 4, 5, 2], 6) == [
            (0, 1, 0, 3),
            (1, 2, 3, 7),
            (2, 3, 7, 12),
            (3, 4, 12, 14),
        ]

    def test_oversized_system_runs_alone(self):
        assert _chunk_bounds([2, 50, 2], 10) == [(0, 1, 0, 2), (1, 2, 2, 52), (2, 3, 52, 54)]


class TestChunkedScalers:
    def test_matches_single_pass(self, mixed_state):
        reference = _torch_sim_n_edges_scalers(mixed_state, CUTOFF)
        assert len(reference) == mixed_state.n_systems
        for max_atoms in (1, 10, 25, 1000):
            chunked = n_edges_scalers_chunked(mixed_state, CUTOFF, max_atoms=max_atoms)
            assert chunked == reference, f"max_atoms={max_atoms}"

    def test_no_budget_on_cpu_uses_single_pass(self, mixed_state):
        reference = _torch_sim_n_edges_scalers(mixed_state, CUTOFF)
        assert n_edges_scalers_chunked(mixed_state, CUTOFF) == reference


class TestInstall:
    def test_routes_torch_sim_scalers(self, mixed_state):
        install_chunked_edge_scalers()
        install_chunked_edge_scalers()
        assert ts_autobatching._n_edges_scalers is n_edges_scalers_chunked
        via_torch_sim = ts_autobatching.calculate_memory_scalers(mixed_state, "n_edges", CUTOFF)
        assert via_torch_sim == _torch_sim_n_edges_scalers(mixed_state, CUTOFF)
