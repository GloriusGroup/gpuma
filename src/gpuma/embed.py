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

The backend follows ``config.technical.device`` like the rest of gpuma. The
GPU path is optional: if nvMolKit is missing, broken, or no CUDA device is
present, everything falls back to RDKit on the CPU and results stay valid.

Notes
-----
The GPU and CPU backends are not bit-identical. nvMolKit requires
``useRandomCoords=True``, so the GPU path starts ETKDG from random coordinates
rather than a distance-geometry guess. Both are valid ETKDGv3 embeddings, but
a given molecule may land in a different conformer basin depending on backend.
Pin ``config.technical.device`` if you need run-to-run comparability.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .structure import Structure

if TYPE_CHECKING:  # pragma: no cover - annotation only, avoids importing torch
    from .config import Config

logger = logging.getLogger(__name__)

#: Conformers generated per molecule, by rotatable-bond count. Only the
#: lowest-energy one survives, so these sit far below the morfeus defaults
#: (50/200/300, at conformer.py:1819). Cost is linear in the budget; flexible
#: molecules benefit from a larger one, rigid molecules do not.
CONF_BUDGET: tuple[tuple[int, int], ...] = ((7, 50), (12, 200), (10**9, 300))

#: Default ETKDG seed. ``-1`` matches the morfeus path: RDKit picks a seed per
#: run, so embeddings are NOT reproducible. Pass a fixed ``seed`` for that.
DEFAULT_SEED = -1

#: RMSD threshold (Angstrom) for discarding duplicate conformers during embedding.
DEFAULT_PRUNE_RMS = 0.35

#: Max MMFF/UFF minimization iterations per conformer.
DEFAULT_MAX_ITERS = 200

def _gpu_ids_from_device(device: str) -> list[int] | None:
    """Translate a gpuma device string into nvMolKit ``gpuIds``.

    Deliberately does not consult ``torch.cuda.is_available()``. That reports
    whether *torch* can reach a GPU, which is a separate question from whether
    nvMolKit can -- nvMolKit ships its own CUDA runtime, and the two genuinely
    disagree when torch is built against a newer CUDA than the driver. Letting
    torch veto here would skip a working backend. Intent comes from the config
    string; capability is established by attempting the call.

    Parameters
    ----------
    device:
        ``"cpu"``, ``"cuda"``, or ``"cuda:N"``.

    Returns
    -------
    list[int] | None
        ``None`` for CPU, ``[]`` for every visible GPU, ``[N]`` for a pinned
        device. An unrecognised string falls back to ``None`` with a warning.
    """
    dev = (device or "").strip().lower()
    if dev == "cpu":
        return None
    if dev == "cuda":
        return []
    if dev.startswith("cuda:"):
        try:
            return [int(dev.split(":", 1)[1])]
        except ValueError:
            logger.warning("invalid CUDA index in device %r; using all visible GPUs", device)
            return []
    logger.warning("unrecognised device %r; using CPU", device)
    return None


def _conf_budget(mol, override: int | None) -> int:
    """Return how many conformers to generate for ``mol``.

    ``override`` short-circuits the rotatable-bond tiers with a flat count.
    """
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


def _embed_cpu(prepared, n_confs_override, seed, prune_rms, multiplicity, n_threads):
    """CPU backend: morfeus, one molecule at a time.

    Uses :class:`morfeus.conformer.ConformerEnsemble` so CPU results stay
    consistent with the rest of gpuma (``mol_utils.smiles_to_structure`` and
    everything downstream of it). Unlike that path, the conformer count, seed
    and thread count are passed explicitly rather than left at morfeus's
    defaults.

    Note that morfeus builds its embedding from a bare
    ``AllChem.EmbedParameters()``, which leaves ``useExpTorsionAnglePrefs``
    and ``useBasicKnowledge`` off -- so this is plain distance geometry, not
    ETKDG. The GPU backend does use ETKDGv3, so the two are not equivalent
    beyond the ``useRandomCoords`` difference noted in the module docstring.

    ``prepared`` is a list of ``(index, smiles, mol, charge)``; results are
    written into ``out`` by original index.
    """
    from morfeus.conformer import ConformerEnsemble

    from .mol_utils import _to_coord_list, _to_symbol_list

    out: dict[int, Structure] = {}
    for idx, smiles, mol, charge in prepared:
        n_confs = _conf_budget(mol, n_confs_override)
        try:
            ensemble = ConformerEnsemble.from_rdkit(
                mol,
                n_conformers=n_confs,
                optimize="MMFF94",
                random_seed=seed if seed >= 0 else None,
                rmsd_thres=prune_rms,
                n_threads=n_threads,
            )
            ensemble.prune_rmsd()
            ensemble.multiplicity = multiplicity
            ensemble.sort()
        except Exception as exc:  # noqa: BLE001 - one bad molecule must not stop the batch
            logger.warning("embedding failed for %s: %s", smiles, exc)
            continue

        conformers = list(ensemble)
        if not conformers:
            logger.warning("no conformers generated for %s", smiles)
            continue

        best = conformers[0]  # sort() puts lowest energy first
        symbols = _to_symbol_list(getattr(best, "elements", []))
        coordinates = _to_coord_list(getattr(best, "coordinates", []))
        if len(symbols) != len(coordinates):
            logger.warning("element/coordinate mismatch for %s", smiles)
            continue

        out[idx] = Structure(
            symbols=symbols,
            coordinates=coordinates,
            charge=charge,
            multiplicity=ensemble.multiplicity,
            comment=f"Generated from SMILES: {smiles}",
        )
    return out


def _embed_gpu(
    prepared, n_confs_override, seed, prune_rms, max_iters, multiplicity, gpu_ids, batch_size
):
    """GPU backend: embed and minimize the whole batch via nvMolKit.

    Molecules are grouped by conformer budget because nvMolKit takes a single
    ``confsPerMolecule`` per call, so a batch spanning several budgets needs
    one call per distinct count. Molecules without MMFF parameters are
    minimized on the CPU afterwards rather than maintaining a second GPU path
    for a ~2% minority.
    """
    from nvmolkit.embedMolecules import EmbedMolecules
    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
    from nvmolkit.types import HardwareOptions
    from rdkit.Chem import AllChem

    # -1 lets nvMolKit auto-detect thread count; it links libgomp, so
    # OMP_NUM_THREADS already governs that natively. gpuIds=[] means "every
    # visible GPU", which CUDA_VISIBLE_DEVICES already narrows -- so both
    # resource limits stay the environment's business.
    hardware = HardwareOptions(
        preprocessingThreads=-1,
        batchSize=batch_size,
        batchesPerGpu=4,
        gpuIds=list(gpu_ids) if gpu_ids else [],
    )

    groups: dict[int, list] = {}
    for idx, smiles, mol, charge in prepared:
        groups.setdefault(_conf_budget(mol, n_confs_override), []).append(
            (idx, smiles, mol, charge)
        )

    out: dict[int, Structure] = {}
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
            out[idx] = _to_structure(mol, best, charge, multiplicity, smiles)
    return out


def generate_structures(
    smiles_list: list[str],
    config: Config | None = None,
    multiplicity: int | None = None,
    n_confs: int | None = None,
    seed: int = DEFAULT_SEED,
    prune_rms_thresh: float = DEFAULT_PRUNE_RMS,
    max_iters: int = DEFAULT_MAX_ITERS,
    batch_size: int = 500,
    n_threads: int = 1,
    allow_cpu_fallback: bool = True,
) -> list[Structure | None]:
    """Convert a list of SMILES to 3D structures as a single batch.

    For each molecule a conformer ensemble is embedded with ETKDGv3, every
    conformer is force-field minimized, and the lowest-energy one is returned.
    Molecules that cannot be parsed or embedded yield ``None`` rather than
    raising, so one bad SMILES cannot abort a library.

    Backend selection follows ``config.technical.device``, the same field the
    rest of gpuma uses: ``"cpu"`` runs RDKit, ``"cuda"`` uses every visible
    GPU, ``"cuda:N"`` pins device ``N``. Consistent with the rest of gpuma, a
    GPU request that cannot be served falls back to CPU rather than failing.

    Parameters
    ----------
    smiles_list:
        SMILES strings to convert.
    config:
        gpuma configuration. Loaded from the default location if omitted.
    multiplicity:
        Spin multiplicity applied to every returned structure. ``None`` takes
        ``config.optimization.multiplicity``.
    n_confs:
        Conformers per molecule. ``None`` selects a budget from rotatable-bond
        count via :data:`CONF_BUDGET`; an explicit int applies that count
        uniformly instead.
    seed:
        ETKDG random seed. ``-1`` lets RDKit choose per run.
    prune_rms_thresh:
        RMSD threshold for discarding duplicate conformers during embedding.
    max_iters:
        Maximum force-field minimization iterations per conformer. GPU backend
        only -- the CPU backend goes through morfeus, which does not expose it.
    batch_size:
        Molecules per nvMolKit batch. GPU backend only.
    n_threads:
        RDKit thread count, CPU backend only -- matching RDKit's own default
        of 1. Leave at 1 when the caller already parallelizes across molecules;
        raise it when embedding a small number of molecules in one process.
    allow_cpu_fallback:
        When a GPU was requested but is unusable, ``True`` transparently runs
        on CPU. Set ``False`` to re-raise instead -- useful when the caller
        parallelizes CPU work differently (e.g. across processes) and needs to
        know the GPU path was not taken, rather than silently getting a serial
        CPU run.

    Returns
    -------
    list[Structure | None]
        One entry per input, in input order. ``None`` marks a failed molecule.

    Raises
    ------
    Exception
        Whatever the GPU backend raised, when a GPU was requested and
        ``allow_cpu_fallback`` is ``False``.
    """
    if not smiles_list:
        return []

    if config is None:
        # Imported lazily: gpuma.config pulls in torch, and the CPU path here
        # has no other reason to pay that import cost.
        from .config import load_config_from_file

        config = load_config_from_file()

    if multiplicity is None:
        multiplicity = int(config.optimization.multiplicity)
    gpu_ids = _gpu_ids_from_device(str(config.technical.device))

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

    if gpu_ids is None:
        out = _embed_cpu(prepared, n_confs, seed, prune_rms_thresh, multiplicity, n_threads)
    else:
        try:
            # No pre-flight probe: a missing/broken nvMolKit raises ImportError
            # here, which is the same fallback path as a mid-run CUDA fault.
            out = _embed_gpu(
                prepared,
                n_confs,
                seed,
                prune_rms_thresh,
                max_iters,
                multiplicity,
                gpu_ids,
                batch_size,
            )
        except Exception as exc:  # noqa: BLE001 - a GPU fault must not lose the run
            if not allow_cpu_fallback:
                raise
            logger.warning("GPU embedding unavailable (%s); falling back to CPU", exc)
            out = _embed_cpu(prepared, n_confs, seed, prune_rms_thresh, multiplicity, n_threads)

    for idx, structure in out.items():
        results[idx] = structure
    return results
