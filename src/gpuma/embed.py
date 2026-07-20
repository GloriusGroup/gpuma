"""Batched SMILES -> 3D structure generation, GPU-accelerated where available.

This module replaces the per-molecule :func:`gpuma.mol_utils.smiles_to_structure`
path for bulk workloads. Two things make it faster:

1. **Conformer budget.** The morfeus path generates 50-300 conformers per
   molecule (chosen by rotatable-bond count), MMFF-minimizes every one, then
   discards all but the lowest-energy one. This module generates a much smaller
   budget, since only one conformer is ever consumed downstream.
2. **Batching.** Conformer generation is submitted as a batch across molecules,
   which is what the GPU backend (nvMolKit) needs to be worth using at all.
   A per-molecule GPU call loses to CPU on launch overhead.

The GPU backend is optional. If nvMolKit is missing, broken, or no CUDA device
is present, everything falls back to RDKit on the CPU and results stay valid.

Notes
-----
The GPU and CPU backends are not bit-identical. nvMolKit requires
``useRandomCoords=True``, so the GPU path starts ETKDG from random coordinates
rather than a distance-geometry guess. Both are valid ETKDGv3 embeddings, but
a given molecule may land in a different conformer basin depending on backend.
Pin ``backend`` explicitly if you need run-to-run comparability.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from .structure import Structure

logger = logging.getLogger(__name__)

#: Conformers generated per molecule, by rotatable-bond count.
#: Only the lowest-energy conformer survives, so these are deliberately far
#: below the morfeus defaults (50/200/300). Raising the upper tiers buys a
#: better-converged minimum for flexible molecules at linear cost.
CONF_BUDGET: tuple[tuple[int, int], ...] = ((7, 4), (12, 16), (10**9, 32))

#: Default ETKDG seed. Fixed so runs are reproducible; the morfeus path used -1.
DEFAULT_SEED = 0xF00D

#: RMSD threshold (Angstrom) for discarding duplicate conformers during embedding.
DEFAULT_PRUNE_RMS = 0.35

#: Max MMFF/UFF minimization iterations per conformer.
DEFAULT_MAX_ITERS = 200


@dataclass(frozen=True)
class BackendStatus:
    """Result of probing for a usable GPU backend."""

    available: bool
    reason: str

    def __bool__(self) -> bool:
        return self.available


_gpu_probe: BackendStatus | None = None


def gpu_backend_status(refresh: bool = False) -> BackendStatus:
    """Report whether the nvMolKit GPU backend can actually be used.

    The probe is cached: it imports a native extension and queries CUDA, which
    is not free, and the answer cannot change within a process.

    Parameters
    ----------
    refresh:
        Re-run the probe instead of returning the cached result.

    Returns
    -------
    BackendStatus
        Truthy if GPU embedding is usable, with a human-readable ``reason``
        explaining the verdict either way.
    """
    global _gpu_probe
    if _gpu_probe is not None and not refresh:
        return _gpu_probe

    try:
        import torch
    except ImportError:
        _gpu_probe = BackendStatus(False, "torch not installed")
        return _gpu_probe

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        _gpu_probe = BackendStatus(False, "no CUDA device visible")
        return _gpu_probe

    try:
        import nvmolkit.embedMolecules  # noqa: F401
        import nvmolkit.mmffOptimization  # noqa: F401
        from nvmolkit.types import HardwareOptions  # noqa: F401
    except ImportError as exc:
        # The common failure is an RDKit/boost ABI mismatch: nvMolKit pins an
        # exact rdkit build and loads boost .so files by content hash, so a
        # different rdkit version leaves the symlink dangling.
        _gpu_probe = BackendStatus(False, f"nvmolkit import failed ({exc})")
        return _gpu_probe

    n = torch.cuda.device_count()
    _gpu_probe = BackendStatus(True, f"nvmolkit ready, {n} CUDA device(s)")
    return _gpu_probe


def _conf_budget(mol, override: int | None) -> int:
    """Return how many conformers to generate for ``mol``."""
    if override is not None:
        return max(1, override)
    from rdkit.Chem import rdMolDescriptors

    nrb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    for threshold, budget in CONF_BUDGET:
        if nrb <= threshold:
            return budget
    return CONF_BUDGET[-1][1]  # pragma: no cover - final tier is unbounded


def _prepare(smiles: str):
    """Parse a SMILES into an H-added mol plus its formal charge.

    Charge is read before ``AddHs`` to match the existing gpuma behaviour.

    Returns
    -------
    tuple[Mol, int] | None
        ``None`` if the SMILES cannot be parsed.
    """
    from rdkit import Chem

    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    charge = Chem.GetFormalCharge(mol)
    return Chem.AddHs(mol), charge


def _etkdg_params(seed: int, prune_rms: float, random_coords: bool, n_threads: int = 1):
    """Build ETKDGv3 parameters.

    The morfeus path built these from a bare ``EmbedParameters()``, which left
    ``useExpTorsionAnglePrefs``/``useBasicKnowledge`` off -- so it was running
    plain distance geometry, not ETKDG. ``ETKDGv3()`` turns them on.
    """
    from rdkit.Chem import rdDistGeom

    p = rdDistGeom.ETKDGv3()
    p.randomSeed = seed
    p.pruneRmsThresh = prune_rms
    p.numThreads = n_threads
    p.useRandomCoords = random_coords  # required True by nvMolKit
    return p


def _to_structure(mol, conf_id: int, charge: int, multiplicity: int, smiles: str) -> Structure:
    """Extract one conformer into a :class:`Structure`."""
    conf = mol.GetConformer(conf_id)
    positions = conf.GetPositions()
    return Structure(
        symbols=[atom.GetSymbol() for atom in mol.GetAtoms()],
        coordinates=[(float(x), float(y), float(z)) for x, y, z in positions],
        charge=charge,
        multiplicity=multiplicity,
        comment=f"Generated from SMILES: {smiles}",
    )


def _minimize_cpu(mol, max_iters: int, n_threads: int) -> list[float]:
    """Minimize every conformer of ``mol`` in place, returning per-conformer energies.

    Falls back to UFF when MMFF94 lacks parameters for the molecule (roughly 2%
    of a typical library -- e.g. some phosphine ligands). Returns an empty list
    if neither force field applies, in which case the caller keeps conformer 0
    unminimized rather than dropping the molecule.
    """
    from rdkit.Chem import AllChem

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=max_iters, numThreads=n_threads)
        elif AllChem.UFFHasAllMoleculeParams(mol):
            res = AllChem.UFFOptimizeMoleculeConfs(mol, maxIters=max_iters, numThreads=n_threads)
        else:
            return []
    except Exception as exc:  # noqa: BLE001 - a bad force field must not kill the batch
        logger.debug("minimization failed, keeping unminimized conformer: %s", exc)
        return []
    return [energy for _converged, energy in res]


def _embed_cpu(prepared, n_confs_override, seed, prune_rms, max_iters, n_threads):
    """CPU backend: embed and minimize one molecule at a time.

    ``prepared`` is a list of ``(index, smiles, mol, charge)``; results are
    written into ``out`` by original index.
    """
    from rdkit.Chem import rdDistGeom

    out: dict[int, tuple] = {}
    for idx, smiles, mol, charge in prepared:
        n_confs = _conf_budget(mol, n_confs_override)
        params = _etkdg_params(seed, prune_rms, random_coords=False, n_threads=n_threads)
        cids = list(rdDistGeom.EmbedMultipleConfs(mol, n_confs, params))
        if not cids:
            # Retry from random coordinates -- rescues most cage/macrocycle failures.
            params = _etkdg_params(seed, prune_rms, random_coords=True, n_threads=n_threads)
            cids = list(rdDistGeom.EmbedMultipleConfs(mol, n_confs, params))
        if not cids:
            logger.warning("embedding failed for %s", smiles)
            continue
        energies = _minimize_cpu(mol, max_iters, n_threads)
        best = cids[min(range(len(energies)), key=energies.__getitem__)] if energies else cids[0]
        out[idx] = (mol, best, charge, smiles)
    return out


def _embed_gpu(prepared, n_confs_override, seed, prune_rms, max_iters, gpu_ids, batch_size):
    """GPU backend: embed and minimize the whole batch via nvMolKit.

    Molecules are grouped by conformer budget because nvMolKit takes a single
    ``confsPerMolecule`` per call. Molecules without MMFF parameters are
    minimized on the CPU afterwards rather than maintaining a second GPU path
    for a ~2% minority.
    """
    from nvmolkit.embedMolecules import EmbedMolecules
    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
    from nvmolkit.types import HardwareOptions
    from rdkit.Chem import AllChem

    hardware = HardwareOptions(
        preprocessingThreads=min(8, os.cpu_count() or 1),
        batchSize=batch_size,
        batchesPerGpu=4,
        gpuIds=list(gpu_ids) if gpu_ids else [],
    )

    groups: dict[int, list] = {}
    for idx, smiles, mol, charge in prepared:
        groups.setdefault(_conf_budget(mol, n_confs_override), []).append(
            (idx, smiles, mol, charge)
        )

    out: dict[int, tuple] = {}
    for n_confs, members in sorted(groups.items()):
        mols = [m for _, _, m, _ in members]
        # nvMolKit mandates useRandomCoords=True.
        params = _etkdg_params(seed, prune_rms, random_coords=True)
        EmbedMolecules(mols, params, confsPerMolecule=n_confs, hardwareOptions=hardware)

        mmff_ok = [m for m in mols if m.GetNumConformers() and AllChem.MMFFHasAllMoleculeParams(m)]
        energies_by_mol: dict[int, list[float]] = {}
        if mmff_ok:
            nested = MMFFOptimizeMoleculesConfs(
                mmff_ok, maxIters=max_iters, hardwareOptions=hardware
            )
            for mol, energies in zip(mmff_ok, nested, strict=True):
                energies_by_mol[id(mol)] = list(energies)

        for idx, smiles, mol, charge in members:
            if not mol.GetNumConformers():
                logger.warning("GPU embedding produced no conformer for %s", smiles)
                continue
            cids = [c.GetId() for c in mol.GetConformers()]
            energies = energies_by_mol.get(id(mol))
            if energies is None:
                # No MMFF parameters -- minimize this one on the CPU.
                energies = _minimize_cpu(mol, max_iters, n_threads=1)
            best = (
                cids[min(range(len(energies)), key=energies.__getitem__)] if energies else cids[0]
            )
            out[idx] = (mol, best, charge, smiles)
    return out


def generate_structures(
    smiles_list: list[str],
    multiplicity: int = 1,
    n_confs: int | None = None,
    backend: str = "auto",
    seed: int = DEFAULT_SEED,
    prune_rms_thresh: float = DEFAULT_PRUNE_RMS,
    max_iters: int = DEFAULT_MAX_ITERS,
    gpu_ids: list[int] | None = None,
    batch_size: int = 500,
    n_threads: int = 1,
) -> list[Structure | None]:
    """Convert a list of SMILES to 3D structures as a single batch.

    For each molecule a small conformer ensemble is embedded with ETKDGv3,
    every conformer is force-field minimized, and the lowest-energy one is
    returned. Molecules that cannot be parsed or embedded yield ``None`` rather
    than raising, so one bad SMILES cannot abort a library.

    Parameters
    ----------
    smiles_list:
        SMILES strings to convert.
    multiplicity:
        Spin multiplicity applied to every returned structure.
    n_confs:
        Conformers per molecule. ``None`` selects a budget from rotatable-bond
        count via :data:`CONF_BUDGET`.
    backend:
        ``"auto"`` uses the GPU when usable and silently falls back to CPU,
        ``"gpu"`` requires it (raises if unavailable), ``"cpu"`` forces RDKit.
    seed:
        ETKDG random seed.
    prune_rms_thresh:
        RMSD threshold for discarding duplicate conformers during embedding.
    max_iters:
        Maximum force-field minimization iterations per conformer.
    gpu_ids:
        CUDA devices for the GPU backend. ``None``/empty uses all visible GPUs.
    batch_size:
        Molecules per nvMolKit batch. GPU backend only.
    n_threads:
        RDKit thread count. CPU backend only.

    Returns
    -------
    list[Structure | None]
        One entry per input, in input order. ``None`` marks a failed molecule.

    Raises
    ------
    ValueError
        If ``backend`` is not one of ``"auto"``, ``"gpu"``, ``"cpu"``, or if
        ``backend="gpu"`` was requested but no usable GPU backend exists.
    """
    if backend not in ("auto", "gpu", "cpu"):
        raise ValueError(f"backend must be 'auto', 'gpu' or 'cpu', got {backend!r}")
    if not smiles_list:
        return []

    prepared = []
    results: list[Structure | None] = [None] * len(smiles_list)
    for idx, smiles in enumerate(smiles_list):
        got = _prepare(smiles)
        if got is None:
            logger.warning("could not parse SMILES at index %d: %r", idx, smiles)
            continue
        mol, charge = got
        prepared.append((idx, smiles, mol, charge))

    if not prepared:
        return results

    status = gpu_backend_status()
    if backend == "gpu" and not status:
        raise ValueError(f"backend='gpu' requested but unavailable: {status.reason}")
    use_gpu = backend == "gpu" or (backend == "auto" and bool(status))

    if use_gpu:
        try:
            out = _embed_gpu(
                prepared, n_confs, seed, prune_rms_thresh, max_iters, gpu_ids, batch_size
            )
        except Exception as exc:  # noqa: BLE001 - a GPU fault must not lose the run
            if backend == "gpu":
                raise
            logger.warning("GPU embedding failed (%s); falling back to CPU", exc)
            out = _embed_cpu(prepared, n_confs, seed, prune_rms_thresh, max_iters, n_threads)
    else:
        if backend == "auto":
            logger.info("using CPU embedding backend: %s", status.reason)
        out = _embed_cpu(prepared, n_confs, seed, prune_rms_thresh, max_iters, n_threads)

    for idx, (mol, conf_id, charge, smiles) in out.items():
        results[idx] = _to_structure(mol, conf_id, charge, multiplicity, smiles)
    return results
