"""Tests for the pairwise conformer RMSD calculators.

No mocking — real RDKit throughout, matching the rest of the suite. Ensembles
are embedded with a fixed seed and a small conformer count so they stay fast.
"""

import logging

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolAlign import GetBestRMS

from gpuma.conformer_generation import rmsd as rmsd_module
from gpuma.conformer_generation.rmsd import (
    _strip_nonpolar_hydrogens,
    kabsch_rmsd_fn,
    symmetric_rmsd_fn,
)

#: Two t-butyls and a methyl: many rotor orientations that are the same
#: conformer once the graph's automorphisms are allowed.
ROTOR_RICH = "Cc1ccccc1C(C)(C)C"

#: Its two carboxyl rotamers are a real pair of minima, but GetBestRMS may
#: swap the carboxyl oxygens, so heavy atoms alone make them look identical.
THIAZOLE_ACID = "O=C(O)c1cscn1"


def _ensemble(smiles: str, n_confs: int = 12, seed: int = 42):
    """An H-added mol with minimized conformers, plus its symbols and coordinates."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=400)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coordinates = [conf.GetPositions() for conf in mol.GetConformers()]
    return mol, symbols, coordinates


# ---------------------------------------------------------------------------
# Polar hydrogens
# ---------------------------------------------------------------------------


def test_strip_drops_carbon_hydrogens_and_keeps_polar_ones():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))

    stripped = _strip_nonpolar_hydrogens(mol)

    kept = [atom.GetSymbol() for atom in stripped.GetAtoms()]
    assert kept.count("H") == 1, "only the hydroxyl hydrogen survives"
    assert stripped.GetNumAtoms() == 4


@pytest.mark.parametrize(
    ("smiles", "polar_hydrogens"),
    [("CCO", 1), ("CCN", 2), ("CCS", 1), ("CC", 0)],
)
def test_strip_keeps_every_heteroatom_hydrogen(smiles, polar_hydrogens):
    stripped = _strip_nonpolar_hydrogens(Chem.AddHs(Chem.MolFromSmiles(smiles)))

    hydrogens = [a for a in stripped.GetAtoms() if a.GetSymbol() == "H"]
    assert len(hydrogens) == polar_hydrogens


def test_strip_removes_conformer_coordinates_too():
    """A stripped mol must stay indexable, or GetBestRMS would read stale positions."""
    mol, _, _ = _ensemble("CCO", n_confs=2)

    stripped = _strip_nonpolar_hydrogens(mol)

    assert stripped.GetNumConformers() == mol.GetNumConformers()
    assert stripped.GetConformer(0).GetNumAtoms() == stripped.GetNumAtoms()


# ---------------------------------------------------------------------------
# The two metrics
# ---------------------------------------------------------------------------


def test_symmetric_collapses_a_pair_the_blind_metric_keeps():
    """Rotor-equivalent conformers are one basin; only the symmetric metric says so."""
    mol, symbols, coordinates = _ensemble(ROTOR_RICH)
    blind = kabsch_rmsd_fn(symbols, coordinates)
    symmetric = symmetric_rmsd_fn(mol, symbols, coordinates, ROTOR_RICH)

    pairs = [
        (blind(i, j), symmetric(i, j))
        for i in range(len(coordinates))
        for j in range(i)
    ]
    assert any(b > 0.5 > s for b, s in pairs), "no rotor duplicate was recognised"


def test_symmetric_never_exceeds_the_blind_metric_without_polar_hydrogens():
    """With no polar hydrogen the atom sets match, so minimising over automorphisms
    can only lower the RMSD."""
    mol, symbols, coordinates = _ensemble("CCCCCCCC")
    blind = kabsch_rmsd_fn(symbols, coordinates)
    symmetric = symmetric_rmsd_fn(mol, symbols, coordinates, "CCCCCCCC")

    for i in range(len(coordinates)):
        for j in range(i):
            assert symmetric(i, j) <= blind(i, j) + 1e-6


def test_symmetric_is_order_independent():
    """GetBestRMS aligns the probe in place, so repeated and reordered calls must agree.

    A prior rigid transform of the probe cannot change an RMSD taken after
    optimal superposition, but the sweep asks pairs in one order only, so the
    invariance is worth pinning.
    """
    mol, symbols, coordinates = _ensemble(ROTOR_RICH, n_confs=6)
    symmetric = symmetric_rmsd_fn(mol, symbols, coordinates, ROTOR_RICH)

    forward = {(i, j): symmetric(i, j) for i in range(6) for j in range(i)}
    backward = {
        (i, j): symmetric(i, j) for i in reversed(range(6)) for j in reversed(range(i))
    }

    for pair, value in forward.items():
        assert backward[pair] == pytest.approx(value, abs=1e-9)
        assert symmetric(*pair) == pytest.approx(value, abs=1e-9)


def test_carboxyl_rotamers_survive_the_symmetric_metric():
    """The regression the polar-hydrogen convention exists for.

    Over heavy atoms alone GetBestRMS swaps the carboxyl oxygens and maps a
    180-degree rotamer back onto the conformer it was turned from.
    """
    mol, symbols, coordinates = _ensemble(THIAZOLE_ACID)
    symmetric = symmetric_rmsd_fn(mol, symbols, coordinates, THIAZOLE_ACID)
    heavy_only = Chem.RemoveHs(Chem.Mol(mol))

    rotamers = [
        (i, j)
        for i in range(len(coordinates))
        for j in range(i)
        if GetBestRMS(heavy_only, heavy_only, prbId=i, refId=j) < 0.5
        and symmetric(i, j) > 0.5
    ]
    assert rotamers, "heavy-atom-only RMSD collapsed nothing, so nothing was tested"
    for i, j in rotamers:
        assert symmetric(i, j) > 1.0


# ---------------------------------------------------------------------------
# Declining to serve an ensemble
# ---------------------------------------------------------------------------


def test_kabsch_declines_when_there_is_nothing_to_superpose_on():
    assert kabsch_rmsd_fn(["C", "H", "H", "H", "H"], [[(0.0, 0.0, 0.0)] * 5]) is None


def test_symmetric_declines_without_a_mol():
    assert symmetric_rmsd_fn(None, ["C", "C"], [[(0.0, 0.0, 0.0)] * 2]) is None


def test_symmetric_declines_on_an_atom_order_mismatch(caplog):
    """A structure that did not come from this embedding must not be trusted."""
    mol, symbols, coordinates = _ensemble("CCO", n_confs=2)
    scrambled = list(reversed(symbols))

    with caplog.at_level(logging.WARNING):
        assert symmetric_rmsd_fn(mol, scrambled, coordinates, "CCO") is None
    assert "atom order" in caplog.text


def test_symmetric_declines_when_over_budget(caplog, monkeypatch):
    """Cost is estimated up front, so an expensive molecule falls back rather than hangs."""
    mol, symbols, coordinates = _ensemble(ROTOR_RICH, n_confs=4)
    monkeypatch.setattr(rmsd_module, "SYMMETRIC_BUDGET_SECONDS", 0.0)

    with caplog.at_level(logging.WARNING):
        assert symmetric_rmsd_fn(mol, symbols, coordinates, ROTOR_RICH) is None
    assert "automorphisms" in caplog.text


def test_symmetric_serves_the_same_ensemble_within_budget():
    """The budget is the only thing that declined it above."""
    mol, symbols, coordinates = _ensemble(ROTOR_RICH, n_confs=4)

    assert symmetric_rmsd_fn(mol, symbols, coordinates, ROTOR_RICH) is not None
