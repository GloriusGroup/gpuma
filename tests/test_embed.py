"""Tests for batched SMILES -> 3D structure generation.

No mocking — real RDKit and morfeus throughout, matching the rest of the suite.
Tests pin ``device="cpu"`` and pass a small ``n_confs`` so they stay fast: the
default CONF_BUDGET generates 50-300 conformers per molecule, which is far more
work than any of these assertions need.
"""

import copy

import pytest

from gpuma.config import Config, load_config_from_file
from gpuma.embed import (
    CONF_BUDGET,
    _conf_budget,
    _force_field_for,
    _gpu_ids_from_device,
    _prepare,
    generate_ensembles,
    generate_structures,
)
from gpuma.structure import Structure

# ---------------------------------------------------------------------------
# Constants and fixtures
# ---------------------------------------------------------------------------

#: Small and rigid, so embedding is quick and the atom counts are known.
ETHANOL = "CCO"
METHANE = "C"
BENZOIC_ACID = "c1ccccc1C(=O)O"

#: Pinacol boronate: no MMFF94 parameters, but UFF covers it. Around 0.6% of
#: a typical library falls in this gap, so the UFF rung matters.
UFF_ONLY = "CC1(C)OB(c2ccc(C(=O)O)cc2)OC1(C)C"

#: Parses as a molecule but has neither MMFF94 nor UFF parameters.
NO_FORCE_FIELD_PARAMS = "[Fe](Cl)(Cl)Cl"

#: Rejected by RDKit's SMILES parser.
INVALID_SMILES = "not_a_smiles"

#: Keep conformer counts low; these tests check plumbing, not search quality.
FEW_CONFS = 3


@pytest.fixture
def cpu_config() -> Config:
    """Config pinned to the CPU backend, so tests never depend on a GPU."""
    config = copy.deepcopy(load_config_from_file())
    config.technical.device = "cpu"
    return config


# ---------------------------------------------------------------------------
# Device string -> gpuIds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cpu", None),
        ("CPU", None),
        ("  cpu  ", None),
        ("cuda", []),
        ("cuda:0", [0]),
        ("cuda:3", [3]),
    ],
)
def test_gpu_ids_from_device(device, expected):
    """Recognised device strings map to nvMolKit gpuIds; None means CPU."""
    assert _gpu_ids_from_device(device) == expected


@pytest.mark.parametrize("device", ["", "weird", "gpu", None])
def test_gpu_ids_from_device_unrecognised_falls_back_to_cpu(device):
    """An unusable device string degrades to CPU rather than raising."""
    assert _gpu_ids_from_device(device) is None


def test_gpu_ids_from_device_bad_index_uses_all_gpus():
    """A malformed CUDA index still selects the GPU backend, not the CPU."""
    assert _gpu_ids_from_device("cuda:x") == []


# ---------------------------------------------------------------------------
# Conformer budget
# ---------------------------------------------------------------------------


def test_conf_budget_follows_rotatable_bond_tiers():
    """Budget increases with flexibility, per CONF_BUDGET."""
    rigid, _ = _prepare(BENZOIC_ACID)
    flexible, _ = _prepare("CCCCCCCCCC(=O)O")
    very_flexible, _ = _prepare("CCCCCCCC/C=C\\CCCCCCCC(=O)O")

    assert _conf_budget(rigid, None) == CONF_BUDGET[0][1]
    assert _conf_budget(flexible, None) == CONF_BUDGET[1][1]
    assert _conf_budget(very_flexible, None) == CONF_BUDGET[2][1]


def test_conf_budget_override_bypasses_tiers():
    """An explicit count applies uniformly regardless of rotatable bonds."""
    rigid, _ = _prepare(BENZOIC_ACID)
    very_flexible, _ = _prepare("CCCCCCCC/C=C\\CCCCCCCC(=O)O")

    assert _conf_budget(rigid, 7) == 7
    assert _conf_budget(very_flexible, 7) == 7


def test_conf_budget_override_is_at_least_one():
    """A nonsensical override is clamped rather than producing zero conformers."""
    mol, _ = _prepare(ETHANOL)
    assert _conf_budget(mol, 0) == 1
    assert _conf_budget(mol, -5) == 1


# ---------------------------------------------------------------------------
# Force field selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (ETHANOL, "MMFF94"),
        (METHANE, "MMFF94"),
        (BENZOIC_ACID, "MMFF94"),
        (UFF_ONLY, "UFF"),
        (NO_FORCE_FIELD_PARAMS, None),
    ],
)
def test_force_field_ladder(smiles, expected):
    """MMFF94 where it applies, UFF next, None when neither does."""
    mol, _ = _prepare(smiles)
    assert _force_field_for(mol) == expected


def test_uff_only_molecule_is_minimized(cpu_config):
    """A molecule without MMFF94 parameters still gets a real geometry via UFF."""
    (structure,) = generate_structures([UFF_ONLY], cpu_config, n_confs=FEW_CONFS)

    assert isinstance(structure, Structure)
    assert "B" in structure.symbols
    assert len({tuple(xyz) for xyz in structure.coordinates}) > 1


# ---------------------------------------------------------------------------
# SMILES preparation
# ---------------------------------------------------------------------------


def test_prepare_adds_hydrogens_and_reads_charge():
    """Hydrogens are explicit and formal charge is read before AddHs."""
    mol, charge = _prepare(ETHANOL)
    assert mol.GetNumAtoms() == 9
    assert charge == 0


def test_prepare_reads_nonzero_formal_charge():
    """Charged species keep their formal charge."""
    _, anion = _prepare("CC(=O)[O-]")
    _, cation = _prepare("C[NH3+]")
    assert anion == -1
    assert cation == 1


@pytest.mark.parametrize("smiles", [INVALID_SMILES, "", "   ", None])
def test_prepare_returns_none_for_unparseable(smiles):
    """Unparseable input yields None rather than raising."""
    assert _prepare(smiles) is None


# ---------------------------------------------------------------------------
# generate_structures
# ---------------------------------------------------------------------------


def test_generate_structures_returns_one_per_input(cpu_config):
    """Every input gets exactly one slot in the result."""
    smiles = [ETHANOL, METHANE, BENZOIC_ACID]
    results = generate_structures(smiles, cpu_config, n_confs=FEW_CONFS)

    assert len(results) == len(smiles)
    assert all(isinstance(s, Structure) for s in results)


def test_generate_structures_geometry_is_valid(cpu_config):
    """Structures have matching symbols/coordinates and are not degenerate."""
    (structure,) = generate_structures([ETHANOL], cpu_config, n_confs=FEW_CONFS)

    assert structure.n_atoms == 9
    assert len(structure.symbols) == len(structure.coordinates)
    assert {"C", "O", "H"} == set(structure.symbols)
    assert all(len(xyz) == 3 for xyz in structure.coordinates)
    # A collapsed embedding would put every atom at the origin.
    assert len({tuple(xyz) for xyz in structure.coordinates}) > 1


def test_generate_structures_preserves_order_around_failures(cpu_config):
    """Failed molecules become None in place, leaving neighbours aligned."""
    smiles = [ETHANOL, INVALID_SMILES, METHANE, "", BENZOIC_ACID]
    results = generate_structures(smiles, cpu_config, n_confs=FEW_CONFS)

    assert len(results) == len(smiles)
    assert results[1] is None
    assert results[3] is None
    assert results[0].n_atoms == 9
    assert results[2].n_atoms == 5
    assert results[4].n_atoms == 15


def test_generate_structures_empty_input(cpu_config):
    """An empty list is returned unchanged without touching a backend."""
    assert generate_structures([], cpu_config) == []


def test_generate_structures_all_invalid(cpu_config):
    """A batch of only bad SMILES yields all None rather than raising."""
    results = generate_structures([INVALID_SMILES, ""], cpu_config)
    assert results == [None, None]


def test_generate_structures_applies_multiplicity(cpu_config):
    """An explicit multiplicity reaches the returned structures."""
    (structure,) = generate_structures(
        [ETHANOL], cpu_config, multiplicity=3, n_confs=FEW_CONFS
    )
    assert structure.multiplicity == 3


def test_generate_structures_multiplicity_defaults_to_config(cpu_config):
    """Omitting multiplicity takes config.optimization.multiplicity."""
    cpu_config.optimization.multiplicity = 2
    (structure,) = generate_structures([ETHANOL], cpu_config, n_confs=FEW_CONFS)
    assert structure.multiplicity == 2


def test_generate_structures_without_force_field_params(cpu_config):
    """A molecule with no MMFF94/UFF parameters still yields a geometry."""
    (structure,) = generate_structures(
        [NO_FORCE_FIELD_PARAMS], cpu_config, n_confs=FEW_CONFS
    )
    assert isinstance(structure, Structure)
    assert "Fe" in structure.symbols


def test_mixed_force_fields_in_one_batch(cpu_config):
    """MMFF94, UFF and unminimizable molecules survive the same batch."""
    smiles = [ETHANOL, UFF_ONLY, NO_FORCE_FIELD_PARAMS]
    results = generate_structures(smiles, cpu_config, n_confs=FEW_CONFS)

    assert all(isinstance(s, Structure) for s in results)
    assert "B" in results[1].symbols
    assert "Fe" in results[2].symbols


def test_generate_structures_loads_config_when_omitted():
    """Config is optional; the default is loaded on demand."""
    (structure,) = generate_structures([METHANE], n_confs=FEW_CONFS)
    assert isinstance(structure, Structure)


# ---------------------------------------------------------------------------
# generate_ensembles
# ---------------------------------------------------------------------------


def test_generate_ensembles_returns_lists(cpu_config):
    """Each molecule gets a list of conformers, capped at max_num_confs."""
    smiles = [ETHANOL, BENZOIC_ACID]
    results = generate_ensembles(smiles, 3, cpu_config, n_confs=8)

    assert len(results) == len(smiles)
    for conformers in results:
        assert 1 <= len(conformers) <= 3
        assert all(isinstance(s, Structure) for s in conformers)


def test_generate_ensembles_conformers_share_composition(cpu_config):
    """Conformers of one molecule differ in geometry, not in atoms."""
    (conformers,) = generate_ensembles([BENZOIC_ACID], 3, cpu_config, n_confs=8)

    first = conformers[0]
    for other in conformers[1:]:
        assert other.symbols == first.symbols
        assert other.n_atoms == first.n_atoms


def test_generate_ensembles_agrees_with_generate_structures(cpu_config):
    """The single-structure entry point returns the ensemble's first conformer."""
    (single,) = generate_structures([ETHANOL], cpu_config, n_confs=FEW_CONFS)
    (ensemble,) = generate_ensembles([ETHANOL], 3, cpu_config, n_confs=FEW_CONFS)

    assert single.n_atoms == ensemble[0].n_atoms
    assert single.symbols == ensemble[0].symbols


def test_generate_ensembles_preserves_order_around_failures(cpu_config):
    """Failures are None in place, as for generate_structures."""
    results = generate_ensembles(
        [ETHANOL, INVALID_SMILES, METHANE], 2, cpu_config, n_confs=FEW_CONFS
    )
    assert len(results) == 3
    assert results[1] is None
    assert isinstance(results[0], list)
    assert isinstance(results[2], list)


@pytest.mark.parametrize("bad", [0, -1])
def test_generate_ensembles_rejects_nonpositive_count(cpu_config, bad):
    """max_num_confs must be positive."""
    with pytest.raises(ValueError, match="max_num_confs"):
        generate_ensembles([ETHANOL], bad, cpu_config)


# ---------------------------------------------------------------------------
# Backend selection
# ---------------------------------------------------------------------------


def test_cuda_request_falls_back_to_cpu_when_unavailable(cpu_config):
    """A GPU request degrades to CPU rather than failing the run.

    nvMolKit is an optional dependency, so on a machine without a working
    install this exercises the fallback; where it does work, the GPU path runs
    and the same assertions hold.
    """
    cpu_config.technical.device = "cuda"
    (structure,) = generate_structures([ETHANOL], cpu_config, n_confs=FEW_CONFS)

    assert isinstance(structure, Structure)
    assert structure.n_atoms == 9


def test_allow_cpu_fallback_false_surfaces_gpu_failure(cpu_config):
    """With fallback disabled, an unusable GPU raises instead of degrading.

    Skipped when nvMolKit works, since then there is no failure to surface.
    """
    try:
        import nvmolkit.embedMolecules  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("nvMolKit is usable, so the GPU path does not fail")

    cpu_config.technical.device = "cuda"
    with pytest.raises(Exception):  # noqa: B017 - backend decides the type
        generate_structures(
            [ETHANOL], cpu_config, n_confs=FEW_CONFS, allow_cpu_fallback=False
        )


def test_explicit_cpu_never_raises_on_fallback_flag(cpu_config):
    """allow_cpu_fallback is irrelevant when the CPU was requested outright."""
    results = generate_structures(
        [ETHANOL], cpu_config, n_confs=FEW_CONFS, allow_cpu_fallback=False
    )
    assert isinstance(results[0], Structure)


# ---------------------------------------------------------------------------
# Delegation from mol_utils
# ---------------------------------------------------------------------------


def test_mol_utils_smiles_to_structure_delegates(cpu_config):
    """The per-molecule helper returns the same geometry shape as the batch API."""
    from gpuma.mol_utils import smiles_to_structure

    structure = smiles_to_structure(ETHANOL, config=cpu_config)
    assert isinstance(structure, Structure)
    assert structure.n_atoms == 9


def test_mol_utils_ensemble_delegates(cpu_config):
    """The per-molecule ensemble helper caps at max_num_confs."""
    from gpuma.mol_utils import smiles_to_conformer_ensemble

    structures = smiles_to_conformer_ensemble(
        BENZOIC_ACID, max_num_confs=2, config=cpu_config
    )
    assert 1 <= len(structures) <= 2
    assert all(isinstance(s, Structure) for s in structures)


def test_mol_utils_raises_on_invalid_smiles(cpu_config):
    """The per-molecule helpers keep raising, unlike the batch API's None."""
    from gpuma.mol_utils import smiles_to_structure

    with pytest.raises(ValueError):
        smiles_to_structure(INVALID_SMILES, config=cpu_config)
