"""Pairwise conformer RMSD, symmetry-blind and symmetry-aware.

Each builder returns an ``rmsd(i, j)`` callable over one ensemble's coordinate
list, or ``None`` when it cannot serve that ensemble -- too few heavy atoms, an
atom order it cannot trust, or a symmetry group too large to afford. ``None``
is the caller's signal to fall back, never a reason to drop a conformer.
"""

from __future__ import annotations

import logging
from typing import Callable, Sequence

logger = logging.getLogger(__name__)

#: Measured ``GetBestRMS`` cost, per conformer pair per graph automorphism.
SECONDS_PER_AUT_PAIR = 3.1e-6

#: Estimated symmetric-RMSD seconds above which a molecule falls back to Kabsch.
SYMMETRIC_BUDGET_SECONDS = 300.0

#: Cap on the automorphism *count*. Reaching it puts the molecule over budget.
#: Never passed to ``GetBestRMS``, where capping misses real duplicates.
AUTOMORPHISM_CAP = 2_000_000


def _strip_nonpolar_hydrogens(mol):
    """``mol`` without its C-H hydrogens, conformers and all.

    Hydrogens on N, O and S stay, because a polar hydrogen is often the only
    thing separating two minima; the rule drops what is bonded to carbon, so an
    unusual element errs toward being kept.
    """
    from rdkit import Chem

    doomed = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 1
        and atom.GetDegree() == 1
        and atom.GetNeighbors()[0].GetAtomicNum() == 6
    ]
    editable = Chem.RWMol(mol)
    # Descending, so a removal cannot shift an index not yet removed.
    for index in sorted(doomed, reverse=True):
        editable.RemoveAtom(index)
    stripped = editable.GetMol()
    # Both sides of every comparison are this same mol, so the graph only has
    # to be self-consistent; no valence is read.
    Chem.SanitizeMol(stripped)
    return stripped


def kabsch_rmsd_fn(
    symbols: Sequence[str], coordinates: Sequence
) -> Callable[[int, int], float] | None:
    """Heavy-atom RMSD after optimal superposition, index-matched.

    Parameters
    ----------
    symbols:
        Atomic symbols, shared by every conformer.
    coordinates:
        One ``(N, 3)`` coordinate set per conformer.

    Returns
    -------
    Callable[[int, int], float] | None
        ``rmsd(i, j)``, or ``None`` when fewer than two heavy atoms leave
        nothing to superpose on.
    """
    import numpy as np

    heavy = [i for i, symbol in enumerate(symbols) if symbol != "H"]
    if len(heavy) < 2:
        return None

    centred = []
    for xyz in coordinates:
        block = np.asarray(xyz, dtype=float)[heavy]
        centred.append(block - block.mean(axis=0))
    n_heavy = len(heavy)

    def rmsd(i: int, j: int) -> float:
        probe, reference = centred[i], centred[j]
        u, _, vt = np.linalg.svd(probe.T @ reference)
        # Keep the rotation proper, or a chiral pair superposes onto its mirror.
        flip = 1.0 if np.linalg.det(u @ vt) > 0 else -1.0
        rotation = u @ np.diag([1.0, 1.0, flip]) @ vt
        delta = probe @ rotation - reference
        return float(np.sqrt((delta * delta).sum() / n_heavy))

    return rmsd


def symmetric_rmsd_fn(
    mol, symbols: Sequence[str], coordinates: Sequence, smiles: str = ""
) -> Callable[[int, int], float] | None:
    """RMSD over the molecular graph's automorphisms, via ``GetBestRMS``.

    Compares heavy atoms and polar hydrogens, so a rotated methyl or a swapped
    pair of equivalent phenyls reads as identical while carboxyl rotamers do
    not.

    Parameters
    ----------
    mol:
        The ``AddHs`` mol the ensemble was embedded from, in its atom order.
    symbols:
        Atomic symbols, checked against ``mol`` before it is trusted.
    coordinates:
        One ``(N, 3)`` coordinate set per conformer.
    smiles:
        Named in the warning when this path is declined.

    Returns
    -------
    Callable[[int, int], float] | None
        ``rmsd(i, j)``, or ``None`` when the atom order disagrees or the
        symmetry group makes the sweep unaffordable.
    """
    if mol is None:
        return None

    from rdkit import Chem
    from rdkit.Chem.rdMolAlign import GetBestRMS
    from rdkit.Geometry import Point3D

    if [atom.GetSymbol() for atom in mol.GetAtoms()] != list(symbols):
        logger.warning(
            "%s: atom order does not match the embedding; using Kabsch RMSD", smiles
        )
        return None

    template = Chem.Mol(mol)
    template.RemoveAllConformers()
    for xyz in coordinates:
        conformer = Chem.Conformer(template.GetNumAtoms())
        for index, position in enumerate(xyz):
            x, y, z = (float(value) for value in position)
            conformer.SetAtomPosition(index, Point3D(x, y, z))
        # assignId numbers conformers in insertion order, so ids are indices.
        template.AddConformer(conformer, assignId=True)
    compared = _strip_nonpolar_hydrogens(template)

    n_confs = len(coordinates)
    automorphisms = len(
        compared.GetSubstructMatches(
            compared, uniquify=False, useChirality=False, maxMatches=AUTOMORPHISM_CAP
        )
    )
    estimate = n_confs * (n_confs - 1) / 2 * automorphisms * SECONDS_PER_AUT_PAIR
    if automorphisms >= AUTOMORPHISM_CAP or estimate > SYMMETRIC_BUDGET_SECONDS:
        logger.warning(
            "%s: symmetric RMSD estimated at %.0f s over %d automorphisms; "
            "using Kabsch RMSD",
            smiles,
            estimate,
            automorphisms,
        )
        return None

    def rmsd(i: int, j: int) -> float:
        # GetBestRMS aligns the probe in place; harmless, since it realigns
        # optimally on every call and ``compared`` is a private copy.
        return float(GetBestRMS(compared, compared, prbId=int(i), refId=int(j)))

    return rmsd
