"""Chunked ``n_edges`` memory scalers for torch-sim's autobatchers.

torch-sim sizes every system for bin-packing by running the neighbor list over
the whole batched state in one pass. That pass holds fixed-width buffers of
roughly ten kilobytes per atom, so a few million atoms fill a 44 GB card before
the first optimization step. Per-system edge counts do not depend on other
systems, so the same numbers come out of passes over atom-bounded chunks.
"""

from __future__ import annotations

import logging

import torch
import torch_sim.autobatching as _autobatching
from torch_sim.neighbors import torchsim_nl
from torch_sim.state import SimState

logger = logging.getLogger(__name__)

# Worst-case bytes per neighbor slot: int32 matrix, int32x3 shifts, bool mask,
# and the int64/int32 pair-list temporaries built from them.
_BYTES_PER_NEIGHBOR_SLOT = 53
_DEVICE_FRACTION = 0.5
_FALLBACK_MAX_NEIGHBORS = 192

_torch_sim_n_edges_scalers = _autobatching._n_edges_scalers


def _max_neighbors(cutoff: float) -> int:
    try:
        from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
    except ImportError:
        return _FALLBACK_MAX_NEIGHBORS
    return int(estimate_max_neighbors(cutoff))


def chunk_atom_budget(device: torch.device, cutoff: float) -> int | None:
    """Atoms one neighbor-list pass may hold on ``device``.

    Parameters
    ----------
    device : torch.device
        Device the state lives on; only CUDA devices are budgeted.
    cutoff : float
        Neighbor-list cutoff in Å, which sets the matrix width.

    Returns
    -------
    int or None
        Atom budget, or ``None`` when the device is not budgeted.
    """
    if device.type != "cuda":
        return None
    free, _total = torch.cuda.mem_get_info(device)
    per_atom = _BYTES_PER_NEIGHBOR_SLOT * _max_neighbors(cutoff)
    return max(1, int(free * _DEVICE_FRACTION / per_atom))


def _chunk_bounds(n_atoms_per_system: list[int], max_atoms: int) -> list[tuple[int, int, int, int]]:
    """Split systems into runs of at most ``max_atoms`` atoms; an oversized system runs alone."""
    bounds = []
    s0 = a0 = 0
    atoms = 0
    for s, n in enumerate(n_atoms_per_system):
        if atoms and atoms + n > max_atoms:
            bounds.append((s0, s, a0, a0 + atoms))
            s0, a0, atoms = s, a0 + atoms, 0
        atoms += n
    bounds.append((s0, len(n_atoms_per_system), a0, a0 + atoms))
    return bounds


def n_edges_scalers_chunked(
    state: SimState, cutoff: float, max_atoms: int | None = None
) -> list[float]:
    """Per-system neighbor-list edge counts, computed in atom-bounded chunks.

    Parameters
    ----------
    state : SimState
        Batched state whose atoms are grouped contiguously by system.
    cutoff : float
        Neighbor-list cutoff in Å.
    max_atoms : int, optional
        Atoms per pass; defaults to :func:`chunk_atom_budget` for the state's device.

    Returns
    -------
    list of float
        One edge count per system, identical to torch-sim's single-pass result.
    """
    if max_atoms is None:
        max_atoms = chunk_atom_budget(state.device, cutoff)
    if max_atoms is None or state.n_atoms <= max_atoms:
        return _torch_sim_n_edges_scalers(state, cutoff)

    bounds = _chunk_bounds(state.n_atoms_per_system.tolist(), max_atoms)
    logger.debug(
        "n_edges scalers: %d atoms in %d chunks of <= %d atoms",
        state.n_atoms,
        len(bounds),
        max_atoms,
    )
    scalers: list[float] = []
    for s0, s1, a0, a1 in bounds:
        _, system_mapping, _ = torchsim_nl(
            positions=state.positions[a0:a1],
            cell=state.cell[s0:s1],
            pbc=state.pbc,
            cutoff=cutoff,
            system_idx=state.system_idx[a0:a1] - s0,
        )
        scalers.extend(system_mapping.bincount(minlength=s1 - s0).float().tolist())
    return scalers


def install_chunked_edge_scalers() -> None:
    """Route torch-sim's ``n_edges`` scalers through the chunked pass. Idempotent."""
    if _autobatching._n_edges_scalers is not n_edges_scalers_chunked:
        _autobatching._n_edges_scalers = n_edges_scalers_chunked
