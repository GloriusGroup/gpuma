"""Batched SMILES -> 3D structure generation, GPU-accelerated where available.

The single SMILES-to-geometry path in gpuma: the per-molecule helpers in
:mod:`gpuma.conformer_generation.mol_utils`, and therefore the API and CLI,
delegate here.
:func:`generate_structures` keeps the lowest-energy conformer per molecule,
:func:`generate_ensembles` keeps several. Both take whole lists, because the
GPU backend parallelizes across molecules rather than within one.

The backend follows ``config.technical.device`` like the rest of gpuma, and
falls back to CPU whenever the GPU is unusable.

The two backends do not produce identical geometries: morfeus (CPU) runs plain
distance geometry, while nvMolKit (GPU) runs ETKDGv3 from random coordinates.
Pin the device if you need comparable runs.

Both backends do, however, prune identically. Duplicate removal happens once,
in :func:`_prune_by_rmsd`, *after* force-field minimization and *after* the
energy sort -- never during embedding. Pruning raw embeddings is what the two
backends used to disagree about (they generate different raw geometries), and
it silently discarded hydrogen-bonded rotamers: an O-H torsion barely moves any
heavy atom, so before minimization the rotamers look like duplicates and there
is no energy yet to tell them apart. Salicylic acid collapsed to a single
conformer that way. After minimization the hydrogen-bonded rotamer is a
distinct minimum -- usually the lowest one -- so an energy-ordered prune keeps
it.
"""

from __future__ import annotations

import logging
from math import isfinite
from typing import TYPE_CHECKING

from ..structure import Structure

if TYPE_CHECKING:  # pragma: no cover - annotation only, avoids importing torch
    from ..config import Config

logger = logging.getLogger(__name__)

#: ``(max_rotatable_bonds, n_conformers)`` tiers, first match wins. Mirrors
#: morfeus's own defaults (conformer.py:1819), so the CPU path generates the
#: same number of conformers gpuma generated before this module existed. Cost
#: is roughly linear in the budget: lowering the tiers is the single biggest
#: speedup available, at the price of a less converged conformer search --
#: which matters for flexible molecules and barely at all for rigid ones.
CONF_BUDGET: tuple[tuple[int, int], ...] = ((7, 50), (12, 200), (10**9, 300))

#: Default ETKDG seed. ``-1`` matches the morfeus path: RDKit picks a seed per
#: run, so embeddings are NOT reproducible. Pass a fixed ``seed`` for that.
DEFAULT_SEED = -1

#: Heavy-atom RMSD threshold (Angstrom) for discarding duplicate conformers.
#: Applied by :func:`_prune_by_rmsd` after minimization, not during embedding.
#: The value is morfeus's own default, kept so ensembles stay comparable with
#: what gpuma produced before the prune moved.
DEFAULT_PRUNE_RMS = 0.35

#: Max MMFF94 minimization iterations per conformer. GPU backend only --
#: morfeus does not expose an iteration limit, so the CPU path uses RDKit's
#: default of 200.
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


def _etkdg_params(seed: int, random_coords: bool, n_threads: int = 1):
    """Build ETKDGv3 parameters, with embedding-time pruning switched off.

    The morfeus path builds these from a bare ``EmbedParameters()``, which
    leaves ``useExpTorsionAnglePrefs``/``useBasicKnowledge`` off -- so it runs
    plain distance geometry, not ETKDG. ``ETKDGv3()`` turns them on.

    ``useSmallRingTorsions`` is enabled to match morfeus, which sets it while
    ``ETKDGv3()`` does not. It applies torsion preferences to rings of 8 atoms
    or fewer, so it mostly affects saturated heterocycles.

    ``pruneRmsThresh`` is pinned to ``-1`` (RDKit's own default, meaning no
    pruning). Deduplication is deferred to :func:`_prune_by_rmsd`, which runs
    on minimized, energy-sorted geometries -- see the module docstring for why
    pruning raw embeddings loses conformers that matter.
    """
    from rdkit.Chem import rdDistGeom

    p = rdDistGeom.ETKDGv3()
    p.randomSeed = seed
    p.pruneRmsThresh = -1.0
    p.numThreads = n_threads
    p.useRandomCoords = random_coords  # required True by nvMolKit
    p.useSmallRingTorsions = True
    return p


def _prune_by_rmsd(symbols, coordinates, thresh: float) -> list[int]:
    """Greedy heavy-atom RMSD prune over an *energy-sorted* conformer list.

    Walks the list from lowest energy downwards and keeps a conformer only when
    it is further than ``thresh`` from every conformer already kept, so the
    survivor of each duplicate cluster is its lowest-energy member. Callers
    must therefore sort before calling; pruning an unsorted list keeps whichever
    conformer happened to come first.

    RMSD is index-matched over heavy atoms after optimal superposition (Kabsch),
    with no graph-automorphism search. That deliberately matches morfeus's
    ``AlignMolConformers`` path so the two backends prune identically; the price
    is that symmetry-equivalent orientations are not recognised as duplicates.

    Parameters
    ----------
    symbols:
        Atomic symbols, shared by every conformer.
    coordinates:
        One ``(N, 3)`` coordinate set per conformer, lowest energy first.
    thresh:
        Heavy-atom RMSD in Angstrom below which two conformers are the same.
        Non-positive disables pruning.

    Returns
    -------
    list[int]
        Indices into ``coordinates`` to keep, in input (energy) order.
    """
    import numpy as np

    n_confs = len(coordinates)
    if thresh is None or thresh <= 0 or n_confs < 2:
        return list(range(n_confs))

    heavy = [i for i, symbol in enumerate(symbols) if symbol != "H"]
    if len(heavy) < 2:
        # Nothing to superimpose on -- e.g. methane. Any two conformers of such
        # a molecule are the same up to rotation, but saying so here would drop
        # conformers the caller cannot get back, so leave them alone.
        return list(range(n_confs))

    centred = []
    for xyz in coordinates:
        block = np.asarray(xyz, dtype=float)[heavy]
        centred.append(block - block.mean(axis=0))

    n_heavy = len(heavy)
    kept: list[int] = []
    kept_coords: list = []
    for i, probe in enumerate(centred):
        for reference in kept_coords:
            u, _, vt = np.linalg.svd(probe.T @ reference)
            flip = 1.0 if np.linalg.det(u @ vt) > 0 else -1.0
            rotation = u @ np.diag([1.0, 1.0, flip]) @ vt
            delta = probe @ rotation - reference
            if float(np.sqrt((delta * delta).sum() / n_heavy)) <= thresh:
                break
        else:
            kept.append(i)
            kept_coords.append(probe)
    return kept


def _rank_by_energy(cids: list[int], energies, smiles: str) -> list[int]:
    """Order conformer ids lowest-energy first, matching ``ensemble.sort()``.

    ``energies`` is nvMolKit's per-conformer output for one molecule, in
    conformer order, or ``None`` when no force field applied. Anything the
    ranking cannot be trusted on falls back to embedding order rather than
    guessing, because the caller treats position 0 as the winner.

    Parameters
    ----------
    cids:
        Conformer ids, in ``mol.GetConformers()`` order.
    energies:
        One energy per conformer, or ``None`` if the molecule was not
        minimized.
    smiles:
        Only used to name the molecule in warnings.

    Returns
    -------
    list[int]
        ``cids`` reordered, lowest energy first. Non-finite energies sort
        last. Returns ``cids`` unchanged when there is nothing to rank on.
    """
    if energies is None:
        # Neither MMFF94 nor UFF applies, so there is nothing to rank by and
        # the conformers stay as embedded. The CPU backend does the same,
        # silently -- hence the warning here.
        logger.warning(
            "no MMFF94 or UFF parameters for %s; keeping unminimized conformer", smiles
        )
        return list(cids)
    if len(energies) != len(cids):
        # nvMolKit returns one energy per conformer, in conformer order. A
        # mismatch means the pairing is unknowable, and guessing it would
        # silently promote the wrong geometry to lowest-energy -- so fall back
        # to embedding order, as in the no-force-field case above.
        logger.warning(
            "%s: %d energies for %d conformers; keeping embedding order",
            smiles,
            len(energies),
            len(cids),
        )
        return list(cids)
    # A diverged minimization yields NaN, which compares false against
    # everything: sorting on the raw value leaves such a conformer wherever the
    # comparisons happen to drop it, which can be position 0. Order on
    # (is-not-finite, energy) instead, so NaN and +/-inf sort last and never
    # reach a comparison that decides the winner.
    return [
        cids[i]
        for i in sorted(
            range(len(energies)),
            key=lambda i: (not isfinite(energies[i]), energies[i]),
        )
    ]


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


def _force_field_for(mol) -> str | None:
    """Pick the force field to minimize ``mol`` with.

    Both backends follow the same ladder -- MMFF94, then UFF, then nothing --
    so a given molecule is minimized the same way regardless of device.
    MMFF94 is preferred where it applies; UFF covers most of what MMFF94 does
    not (boronic esters, some phosphines), and roughly 0.1% of a typical
    library has neither.

    Because the ladder mixes force fields, energies are only comparable
    *within* a molecule -- which is all this module uses them for, to rank
    conformers. Do not compare energies across molecules without checking
    which field produced each.

    Returns
    -------
    str | None
        ``"MMFF94"``, ``"UFF"``, or ``None`` when no force field applies.
    """
    from rdkit.Chem import AllChem

    if AllChem.MMFFHasAllMoleculeParams(mol):
        return "MMFF94"
    if AllChem.UFFHasAllMoleculeParams(mol):
        return "UFF"
    return None


def _embed_cpu(prepared, n_confs_override, seed, prune_rms, multiplicity, n_threads, n_keep=1):
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
    written into ``out`` by original index as lists of at most ``n_keep``
    structures, lowest energy first (``n_keep=None`` keeps every survivor).

    morfeus's own pruning is switched off (``rmsd_thres=None``) and the ensemble
    is sorted before :func:`_prune_by_rmsd` runs, so the prune sees minimized,
    energy-ordered geometries -- identical treatment to the GPU backend. The
    previous code pruned twice, once during embedding and once via
    ``ensemble.prune_rmsd()`` *before* ``sort()``; that second call is greedy
    over the list as it stands, so it kept whichever conformer was embedded
    first rather than the lower-energy one.
    """
    from morfeus.conformer import ConformerEnsemble

    from .mol_utils import _to_coord_list, _to_symbol_list

    out: dict[int, list[Structure]] = {}
    for idx, smiles, mol, charge in prepared:
        n_confs = _conf_budget(mol, n_confs_override)
        force_field = _force_field_for(mol)
        if force_field is None:
            logger.warning(
                "no MMFF94 or UFF parameters for %s; keeping unminimized conformer", smiles
            )
        try:
            ensemble = ConformerEnsemble.from_rdkit(
                mol,
                n_conformers=n_confs,
                optimize=force_field,
                random_seed=seed if seed >= 0 else None,
                rmsd_thres=None,  # pruning is deferred to _prune_by_rmsd
                n_threads=n_threads,
            )
            ensemble.multiplicity = multiplicity
            ensemble.sort()
        except Exception as exc:  # noqa: BLE001 - one bad molecule must not stop the batch
            logger.warning("embedding failed for %s: %s", smiles, exc)
            continue

        conformers = list(ensemble)
        if not conformers:
            logger.warning("no conformers generated for %s", smiles)
            continue

        # sort() has put lowest energy first, which is what the prune needs.
        records: list[tuple[list[str], list[tuple[float, float, float]]]] = []
        for conformer in conformers:
            symbols = _to_symbol_list(getattr(conformer, "elements", []))
            coordinates = _to_coord_list(getattr(conformer, "coordinates", []))
            if len(symbols) != len(coordinates):
                logger.warning("element/coordinate mismatch for %s", smiles)
                continue
            records.append((symbols, coordinates))
        if not records:
            logger.warning("no usable conformers for %s", smiles)
            continue

        survivors = _prune_by_rmsd(records[0][0], [coords for _, coords in records], prune_rms)
        kept = [
            Structure(
                symbols=records[i][0],
                coordinates=records[i][1],
                charge=charge,
                multiplicity=ensemble.multiplicity,
                comment=f"Generated from SMILES: {smiles}",
            )
            for i in survivors[:n_keep]
        ]
        if kept:
            out[idx] = kept
    return out


def _embed_gpu(
    prepared,
    n_confs_override,
    seed,
    prune_rms,
    max_iters,
    multiplicity,
    gpu_ids,
    batch_size,
    n_keep=1,
):
    """GPU backend: embed and minimize the whole batch via nvMolKit.

    Molecules are grouped by conformer budget because nvMolKit takes a single
    ``confsPerMolecule`` per call, so a batch spanning several budgets needs
    one call per distinct count. Within a group they are partitioned again by
    force field -- see :func:`_force_field_for` -- and minimized in one batched
    call per field, so the UFF minority stays on the GPU rather than falling
    back to per-molecule CPU work.

    nvMolKit is handed ETKDG parameters with pruning disabled; duplicates are
    removed afterwards by :func:`_prune_by_rmsd`, on the energy-ranked minimized
    conformers, exactly as on the CPU path.
    """
    from nvmolkit.embedMolecules import EmbedMolecules
    from nvmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs
    from nvmolkit.types import HardwareOptions
    from nvmolkit.uffOptimization import UFFOptimizeMoleculesConfs

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

    out: dict[int, list[Structure]] = {}
    for n_confs, members in sorted(groups.items()):
        mols = [m for _, _, m, _ in members]
        # nvMolKit mandates useRandomCoords=True.
        params = _etkdg_params(seed, random_coords=True)
        EmbedMolecules(mols, params, confsPerMolecule=n_confs, hardwareOptions=hardware)

        by_field: dict[str, list] = {"MMFF94": [], "UFF": []}
        for mol in mols:
            if not mol.GetNumConformers():
                continue
            field = _force_field_for(mol)
            if field is not None:
                by_field[field].append(mol)

        # maxIters is passed to both so the two groups get the same convergence
        # budget; nvMolKit's UFF default is 1000 against MMFF's 200.
        energies_by_mol: dict[int, list[float]] = {}
        for field, minimize in (
            ("MMFF94", MMFFOptimizeMoleculesConfs),
            ("UFF", UFFOptimizeMoleculesConfs),
        ):
            group = by_field[field]
            if not group:
                continue
            nested = minimize(group, maxIters=max_iters, hardwareOptions=hardware)
            for mol, energies in zip(group, nested, strict=True):
                energies_by_mol[id(mol)] = list(energies)

        for idx, smiles, mol, charge in members:
            if not mol.GetNumConformers():
                logger.warning("GPU embedding produced no conformer for %s", smiles)
                continue
            cids = [c.GetId() for c in mol.GetConformers()]
            ranked = _rank_by_energy(cids, energies_by_mol.get(id(mol)), smiles)
            # Prune here, on minimized and energy-ordered geometries, using the
            # same helper as the CPU backend.
            symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
            ranked_coords = [mol.GetConformer(cid).GetPositions() for cid in ranked]
            survivors = _prune_by_rmsd(symbols, ranked_coords, prune_rms)
            kept = [
                _to_structure(mol, ranked[i], charge, multiplicity, smiles)
                for i in survivors[:n_keep]
            ]
            # Guarded like the CPU backend: a molecule that yields nothing gets
            # no entry, so _generate reports it as None. Assigning [] here would
            # make the two backends disagree about what "failed" looks like.
            if kept:
                out[idx] = kept
    return out


def _generate(
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
    n_keep: int | None = 1,
) -> list[list[Structure] | None]:
    """Convert a list of SMILES to 3D structures as a single batch.

    Shared implementation behind :func:`generate_structures` and
    :func:`generate_ensembles`, which differ only in ``n_keep``. For each
    molecule a conformer ensemble is embedded, every conformer is minimized
    with the force field :func:`_force_field_for` selects, and the ``n_keep``
    lowest-energy ones are kept. Molecules that cannot be parsed or embedded
    yield ``None`` rather than raising, so one bad SMILES cannot abort a
    library.

    Backend selection follows ``config.technical.device``, the same field the
    rest of gpuma uses: ``"cpu"`` runs morfeus, ``"cuda"`` uses every visible
    GPU via nvMolKit, ``"cuda:N"`` pins device ``N``. Consistent with the rest
    of gpuma, a GPU request that cannot be served falls back to CPU rather
    than failing.

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
        Conformers *generated* per molecule. ``None`` selects a budget from
        rotatable-bond count via :data:`CONF_BUDGET`; an explicit int applies
        that count uniformly instead. Distinct from ``n_keep``, which is how
        many are returned.
    seed:
        Random seed for the embedding. ``-1`` lets RDKit choose one per run,
        making geometries non-reproducible.
    prune_rms_thresh:
        Heavy-atom RMSD threshold for discarding duplicate conformers. Applied
        after force-field minimization and after the energy sort, so the
        survivor of each duplicate cluster is its lowest-energy member; the
        embedding itself does not prune. Fewer conformers than ``n_confs`` may
        survive it.
    max_iters:
        Maximum force-field minimization iterations per conformer. GPU backend
        only -- the CPU backend goes through morfeus, which does not expose it.
    batch_size:
        Molecules per nvMolKit batch. GPU backend only.
    n_threads:
        Thread count handed to RDKit for embedding and minimization, CPU
        backend only. Leave at 1 when the caller already parallelizes across
        molecules; raise it when embedding a few molecules in one process.
    allow_cpu_fallback:
        When a GPU was requested but is unusable, ``True`` transparently runs
        on CPU. Set ``False`` to re-raise instead -- useful when the caller
        parallelizes CPU work differently (e.g. across processes) and needs to
        know the GPU path was not taken, rather than silently getting a serial
        CPU run.
    n_keep:
        Maximum conformers *returned* per molecule, lowest energy first.
        ``None`` keeps every conformer surviving the post-minimization RMSD prune.

    Returns
    -------
    list[list[Structure] | None]
        One entry per input, in input order. Each entry is a list of at most
        ``n_keep`` structures, or ``None`` for a molecule that could not be
        parsed or embedded.

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
        from ..config import load_config_from_file

        config = load_config_from_file()

    if multiplicity is None:
        multiplicity = int(config.optimization.multiplicity)
    gpu_ids = _gpu_ids_from_device(str(config.technical.device))

    # Mirrors the optimizer's "Optimization device" line. Logged before the
    # attempt, so a later fallback warning tells you the GPU was tried and
    # lost rather than never selected.
    logger.info("Embedding device: %s", "CPU" if gpu_ids is None else "GPU")

    prepared = []
    results: list[list[Structure] | None] = [None] * len(smiles_list)
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
        out = _embed_cpu(
            prepared, n_confs, seed, prune_rms_thresh, multiplicity, n_threads, n_keep
        )
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
                n_keep,
            )
        except Exception as exc:  # noqa: BLE001 - a GPU fault must not lose the run
            if not allow_cpu_fallback:
                raise
            logger.warning("GPU embedding unavailable (%s); falling back to CPU", exc)
            out = _embed_cpu(
                prepared, n_confs, seed, prune_rms_thresh, multiplicity, n_threads, n_keep
            )

    for idx, structures in out.items():
        results[idx] = structures
    return results


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
    """Convert SMILES to one 3D structure each, as a single batch.

    Embeds a conformer ensemble per molecule, minimizes every conformer, and
    keeps the lowest-energy one. Use :func:`generate_ensembles` to keep more
    than one.

    Parameters
    ----------
    smiles_list:
        SMILES strings to convert.
    config:
        gpuma configuration; ``technical.device`` selects the backend. Loaded
        from the default location if omitted.
    multiplicity:
        Spin multiplicity applied to every returned structure. ``None`` takes
        ``config.optimization.multiplicity``.
    n_confs:
        Conformers generated per molecule. ``None`` selects a budget from
        rotatable-bond count via :data:`CONF_BUDGET`; an explicit int applies
        that count uniformly instead.
    seed:
        Random seed for the embedding. ``-1`` lets RDKit choose one per run,
        making geometries non-reproducible.
    prune_rms_thresh:
        Heavy-atom RMSD threshold for discarding duplicate conformers. Applied
        after force-field minimization and after the energy sort, so the
        survivor of each duplicate cluster is its lowest-energy member; the
        embedding itself does not prune. Fewer conformers than ``n_confs`` may
        survive it.
    max_iters:
        Maximum force-field minimization iterations per conformer. GPU backend
        only -- the CPU backend goes through morfeus, which does not expose it.
    batch_size:
        Molecules per nvMolKit batch. GPU backend only.
    n_threads:
        Thread count handed to RDKit, CPU backend only. Leave at 1 when the
        caller already parallelizes across molecules.
    allow_cpu_fallback:
        When a GPU was requested but is unusable, ``True`` transparently runs
        on CPU. Set ``False`` to re-raise instead, so a caller that
        parallelizes CPU work itself can choose its own strategy rather than
        silently getting a serial CPU run.

    Returns
    -------
    list[Structure | None]
        One entry per input, in input order. ``None`` marks a molecule that
        could not be parsed or embedded, so one bad SMILES cannot abort a
        library.

    Raises
    ------
    Exception
        Whatever the GPU backend raised, when a GPU was requested and
        ``allow_cpu_fallback`` is ``False``.
    """
    batches = _generate(
        smiles_list,
        config,
        multiplicity,
        n_confs,
        seed,
        prune_rms_thresh,
        max_iters,
        batch_size,
        n_threads,
        allow_cpu_fallback,
        n_keep=1,
    )
    return [structures[0] if structures else None for structures in batches]


def generate_ensembles(
    smiles_list: list[str],
    max_num_confs: int | None,
    config: Config | None = None,
    multiplicity: int | None = None,
    n_confs: int | None = None,
    seed: int = DEFAULT_SEED,
    prune_rms_thresh: float = DEFAULT_PRUNE_RMS,
    max_iters: int = DEFAULT_MAX_ITERS,
    batch_size: int = 500,
    n_threads: int = 1,
    allow_cpu_fallback: bool = True,
) -> list[list[Structure] | None]:
    """Convert SMILES to conformer ensembles, as a single batch.

    As :func:`generate_structures`, but keeps several conformers per molecule
    instead of one. All other parameters carry the same meaning.

    Parameters
    ----------
    max_num_confs:
        Maximum conformers *returned* per molecule, lowest energy first. This
        is distinct from ``n_confs``, which controls how many are *generated* --
        generating fewer than you keep simply wastes the budget. Fewer than
        requested may come back either way, since RMSD pruning removes
        duplicates. ``None`` keeps every conformer that survives the
        post-minimization RMSD prune, lowest energy first.

    Returns
    -------
    list[list[Structure] | None]
        One entry per input, in input order. Each entry is a list of at most
        ``max_num_confs`` structures, or ``None`` for a molecule that could not
        be parsed or embedded.

    Raises
    ------
    ValueError
        If ``max_num_confs`` is an int that is not positive.
    Exception
        Whatever the GPU backend raised, when a GPU was requested and
        ``allow_cpu_fallback`` is ``False``.
    """
    if max_num_confs is not None and max_num_confs <= 0:
        raise ValueError(f"max_num_confs must be positive, got {max_num_confs}")
    return _generate(
        smiles_list,
        config,
        multiplicity,
        n_confs,
        seed,
        prune_rms_thresh,
        max_iters,
        batch_size,
        n_threads,
        allow_cpu_fallback,
        n_keep=max_num_confs,
    )
