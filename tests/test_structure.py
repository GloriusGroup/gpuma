"""Tests for the Structure dataclass."""

import pytest

from gpuma.structure import Structure, check_spin_state


def test_structure_initialization(methane):
    """Structure stores symbols, coordinates, charge, multiplicity, and comment."""
    s = methane
    assert s.symbols == ["C", "H", "H", "H", "H"]
    assert len(s.coordinates) == 5
    assert s.charge == 0
    assert s.multiplicity == 1
    assert s.comment == "Methane"
    assert s.energy is None
    assert s.metadata == {}


def test_structure_n_atoms(methane):
    """n_atoms property returns the number of symbols."""
    assert methane.n_atoms == 5


def test_structure_with_energy(methane):
    """with_energy() sets energy and supports chaining."""
    assert methane.energy is None
    methane.with_energy(-123.456)
    assert methane.energy == -123.456

    s2 = methane.with_energy(-100.0)
    assert s2 is methane
    assert methane.energy == -100.0


def test_structure_metadata():
    """Structure accepts and stores arbitrary metadata."""
    s = Structure(
        symbols=["H"],
        coordinates=[(0.0, 0.0, 0.0)],
        charge=0,
        multiplicity=2,
        metadata={"source": "test"},
    )
    assert s.metadata["source"] == "test"


@pytest.mark.parametrize(
    ("symbols", "charge", "multiplicity"),
    [
        (["C", "H", "H", "H"], 0, 2),  # methyl radical
        (["O", "H", "H"], 0, 1),
        (["O", "H", "H"], 0, 3),
        (["Na"], 1, 1),
        (["H"], 1, 1),  # a bare proton has no electrons
        (["O", "O"], 0, 3),
    ],
)
def test_check_spin_state_accepts_valid_states(symbols, charge, multiplicity):
    check_spin_state(symbols, charge, multiplicity)


@pytest.mark.parametrize(
    ("symbols", "charge", "multiplicity"),
    [
        (["C", "H", "H", "H"], 0, 1),  # odd electron count as a singlet
        (["O", "H", "H"], 0, 2),
        (["H"], 0, 4),  # more unpaired electrons than electrons
        (["H"], 2, 1),  # negative electron count
    ],
)
def test_check_spin_state_rejects_impossible_states(symbols, charge, multiplicity):
    with pytest.raises(ValueError, match="multiplicity"):
        check_spin_state(symbols, charge, multiplicity)


def test_check_spin_state_rejects_unknown_element():
    with pytest.raises(ValueError, match="Xx"):
        check_spin_state(["Xx"], 0, 1)
