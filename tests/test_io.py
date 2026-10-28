"""Tests for I/O functions — uses real test data files and real RDKit/morfeus."""

import pytest

from gpuma.utils.io_handler import (
    read_multi_xyz,
    read_xyz,
    read_xyz_directory,
    save_as_single_xyz_files,
    save_multi_xyz,
    save_xyz_file,
    smiles_to_ensemble,
    smiles_to_xyz,
)
from gpuma.config import Config
from gpuma.structure import Structure

from conftest import (
    BUTENE_SINGLET_XYZ,
    BUTENE_TRIPLET_MULTI_XYZ,
    MULTI_XYZ,
    MULTI_XYZ_DIR,
    SINGLE_XYZ,
    SMALL_BATCH_XYZ,
)


# ---------------------------------------------------------------------------
# read_xyz
# ---------------------------------------------------------------------------


def test_read_xyz_single_file():
    """Read a single XYZ file and verify structure properties."""
    s = read_xyz(str(SINGLE_XYZ))
    assert isinstance(s, Structure)
    assert s.n_atoms > 0
    assert len(s.coordinates) == s.n_atoms


def test_read_xyz_with_charge_override():
    """Charge and multiplicity overrides are applied to the structure."""
    s = read_xyz(str(SINGLE_XYZ), charge=-1, multiplicity=2)
    assert s.charge == -1
    assert s.multiplicity == 2


def test_read_xyz_butene_singlet():
    """Butene singlet XYZ file loads correctly."""
    s = read_xyz(str(BUTENE_SINGLET_XYZ))
    assert s.n_atoms > 0


def test_read_xyz_not_found():
    """Missing XYZ file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_xyz("/nonexistent/path.xyz")


def test_read_xyz_empty_file(tmp_path):
    """Empty XYZ file raises ValueError."""
    empty = tmp_path / "empty.xyz"
    empty.write_text("")
    with pytest.raises(ValueError):
        read_xyz(str(empty))


def test_read_xyz_malformed(tmp_path):
    """Malformed atom count raises ValueError."""
    bad = tmp_path / "bad.xyz"
    bad.write_text("not_a_number\ncomment\nC 0 0 0\n")
    with pytest.raises(ValueError):
        read_xyz(str(bad))


# ---------------------------------------------------------------------------
# read_multi_xyz
# ---------------------------------------------------------------------------


def test_read_multi_xyz():
    """Multi-XYZ file loads multiple structures."""
    structures = read_multi_xyz(str(MULTI_XYZ))
    assert len(structures) >= 2
    for s in structures:
        assert isinstance(s, Structure)
        assert s.n_atoms > 0


def test_read_multi_xyz_small_batch():
    """500-structure batch file loads completely with expected atom ranges."""
    structures = read_multi_xyz(str(SMALL_BATCH_XYZ))
    assert len(structures) == 50
    atoms = [s.n_atoms for s in structures]
    assert min(atoms) >= 1
    assert max(atoms) <= 200


def test_read_multi_xyz_butene_triplet():
    """Butene triplet multi-XYZ loads at least one structure."""
    structures = read_multi_xyz(str(BUTENE_TRIPLET_MULTI_XYZ))
    assert len(structures) >= 1


@pytest.mark.parametrize(
    ("content", "message"),
    [
        # malformed atom line
        ("2\nfirst\nO 0 0 0\nH 1 0 0\n3\nbad\nO 0 0 0\nH oops\nH 0 1 0\n", "structure 2"),
        # count too high: swallows the next block's header
        ("3\nfirst\nO 0 0 0\nH 1 0 0\n2\nsecond\nN 0 0 0\nN 1 0 0\n", "structure 1"),
        # count too low: an atom line where the next count should be
        ("1\nfirst\nO 0 0 0\nH 1 0 0\n", "atom count"),
        # truncated at end of file
        ("2\nfirst\nO 0 0 0\nH 1 0 0\n5\ntruncated\nO 0 0 0\n", "structure 2"),
        # header without a comment line
        ("2\nfirst\nO 0 0 0\nH 1 0 0\n999\n", "structure 2"),
    ],
)
def test_read_multi_xyz_rejects_inconsistent_blocks(tmp_path, content, message):
    """A block that doesn't match its atom count is an error, never skipped or merged."""
    f = tmp_path / "bad.xyz"
    f.write_text(content)
    with pytest.raises(ValueError, match=message):
        read_multi_xyz(str(f))


def test_read_multi_xyz_allows_blank_lines_between_blocks(tmp_path):
    f = tmp_path / "blank.xyz"
    f.write_text("2\nfirst\nO 0 0 0\nH 1 0 0\n\n2\nsecond\nN 0 0 0\nN 1 0 0\n\n")
    assert [s.comment for s in read_multi_xyz(str(f))] == ["first", "second"]


def test_read_xyz_takes_charge_and_multiplicity_from_gpuma_comment(tmp_path, caplog):
    """gpuma's own output comment carries the state; it wins over the passed default."""
    f = tmp_path / "radical.xyz"
    f.write_text(
        "4\nOptimized | Energy: -1.0 eV | Charge: 0 | Multiplicity: 2\n"
        "C 0 0 0\nH 1 0 0\nH 0 1 0\nH 0 0 1\n"
    )
    s = read_xyz(str(f))
    assert (s.charge, s.multiplicity) == (0, 2)
    assert "Multiplicity" in caplog.text
    multi = read_multi_xyz(str(f), charge=0, multiplicity=1)
    assert (multi[0].charge, multi[0].multiplicity) == (0, 2)


# ---------------------------------------------------------------------------
# read_xyz_directory
# ---------------------------------------------------------------------------


def test_read_xyz_directory():
    """All 28 XYZ files in the test directory are loaded."""
    structures = read_xyz_directory(str(MULTI_XYZ_DIR))
    assert len(structures) == 28
    for s in structures:
        assert s.n_atoms > 0


def test_read_xyz_directory_not_found():
    """Missing directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_xyz_directory("/nonexistent/dir")


def test_read_xyz_directory_raises_on_unreadable_file(tmp_path):
    """One bad file is an error naming that file, not a warning and a shorter list."""
    (tmp_path / "a.xyz").write_text("2\nok\nH 0 0 0\nH 0.7 0 0\n")
    (tmp_path / "b.xyz").write_text("2\nbroken\nH 0 0 0\n")
    with pytest.raises(ValueError, match="b.xyz"):
        read_xyz_directory(str(tmp_path))


def test_read_xyz_directory_records_source_file(tmp_path):
    (tmp_path / "mol_2.xyz").write_text("2\nok\nH 0 0 0\nH 0.7 0 0\n")
    (tmp_path / "mol_10.xyz").write_text("2\nok\nH 0 0 0\nH 0.7 0 0\n")
    structures = read_xyz_directory(str(tmp_path))
    assert [s.metadata["source_file"] for s in structures] == [
        str(tmp_path / "mol_2.xyz"),
        str(tmp_path / "mol_10.xyz"),
    ]


# ---------------------------------------------------------------------------
# save_xyz_file
# ---------------------------------------------------------------------------


def test_save_xyz_file(tmp_path, methane):
    """Saved XYZ file contains atom count, energy, and charge."""
    out = tmp_path / "out.xyz"
    methane.energy = -42.5
    save_xyz_file(methane, str(out))

    assert out.exists()
    content = out.read_text()
    assert "5" in content.split("\n")[0]
    assert "-42.5" in content
    assert "Charge: 0" in content


def test_save_xyz_permission_error(methane):
    """Writing to a read-only path raises PermissionError."""
    with pytest.raises(PermissionError):
        save_xyz_file(methane, "/root/no_permission.xyz")


# ---------------------------------------------------------------------------
# save_multi_xyz
# ---------------------------------------------------------------------------


def test_save_multi_xyz(tmp_path, methane, ethanol):
    """Multi-XYZ output contains one block per structure."""
    out = tmp_path / "multi.xyz"
    save_multi_xyz([methane, ethanol], str(out))

    content = out.read_text()
    blocks = [b for b in content.strip().split("\n") if b.strip().isdigit()]
    assert len(blocks) == 2


# ---------------------------------------------------------------------------
# save_as_single_xyz_files
# ---------------------------------------------------------------------------


def test_save_as_single_xyz_files(tmp_path, methane, ethanol):
    """Each structure is saved as a separate XYZ file."""
    out_dir = tmp_path / "singles"
    save_as_single_xyz_files([methane, ethanol], str(out_dir))
    files = list(out_dir.glob("*.xyz"))
    assert len(files) == 2


def test_save_as_single_xyz_files_creates_dir(tmp_path, methane):
    """Nested output directories are created automatically."""
    nested = tmp_path / "a" / "b" / "c"
    save_as_single_xyz_files([methane], str(nested))
    assert nested.exists()
    assert len(list(nested.glob("*.xyz"))) == 1


# ---------------------------------------------------------------------------
# SMILES conversion (uses real RDKit + morfeus)
# ---------------------------------------------------------------------------


def test_smiles_to_xyz_methane():
    """Methane SMILES produces a 5-atom CH4 structure."""
    s = smiles_to_xyz("C")
    assert isinstance(s, Structure)
    assert "C" in s.symbols
    assert "H" in s.symbols
    assert s.n_atoms == 5


def test_smiles_to_xyz_ethanol():
    """Ethanol SMILES produces a 9-atom C2H5OH structure."""
    s = smiles_to_xyz("CCO")
    assert isinstance(s, Structure)
    assert "O" in s.symbols
    assert s.n_atoms == 9


def test_smiles_to_xyz_as_string():
    """return_full_xyz_str=True returns an XYZ-formatted string."""
    result = smiles_to_xyz("C", return_full_xyz_str=True)
    assert isinstance(result, str)
    assert "C" in result


def test_smiles_to_ensemble():
    """Conformer ensemble generation produces multiple butane structures."""
    structures = smiles_to_ensemble("CCCC", max_num_confs=3)
    assert isinstance(structures, list)
    assert len(structures) >= 1
    for s in structures:
        assert isinstance(s, Structure)
        assert s.n_atoms == 14


def test_smiles_to_xyz_with_multiplicity():
    """Multiplicity parameter is passed through to the structure."""
    s = smiles_to_xyz("C", multiplicity=3)
    assert s.multiplicity == 3


CPU = Config({"technical": {"device": "cpu"}})


def test_smiles_to_xyz_rejects_radical_as_singlet():
    """[CH3] has an odd electron count, so the default singlet is impossible."""
    with pytest.raises(ValueError, match="multiplicity"):
        smiles_to_xyz("[CH3]", config=CPU)
    assert smiles_to_xyz("[CH3]", multiplicity=2, config=CPU).multiplicity == 2


def test_smiles_to_xyz_takes_multiplicity_from_config():
    config = Config({"optimization": {"multiplicity": 3}, "technical": {"device": "cpu"}})
    assert smiles_to_xyz("C", config=config).multiplicity == 3


def test_smiles_to_xyz_uses_config_seed():
    """Same config seed, same geometry; the seed used to be dropped (RDKit picked one)."""
    config = Config({"conformer_generation": {"conformer_seed": 7}, "technical": {"device": "cpu"}})
    first = smiles_to_xyz("CCCCO", config=config)
    second = smiles_to_xyz("CCCCO", config=config)
    assert first.coordinates == second.coordinates


def test_smiles_to_ensemble_uses_config_seed():
    config = Config({"conformer_generation": {"conformer_seed": 7}, "technical": {"device": "cpu"}})
    first = smiles_to_ensemble("CCCCO", max_num_confs=3, config=config)
    second = smiles_to_ensemble("CCCCO", max_num_confs=3, config=config)
    assert [s.coordinates for s in first] == [s.coordinates for s in second]


def test_save_marks_unconverged_structures(tmp_path, methane):
    methane.converged = False
    out = tmp_path / "u.xyz"
    save_multi_xyz([methane], str(out))
    assert "Not converged" in out.read_text().splitlines()[1]


# ---------------------------------------------------------------------------
# Round-trip: write -> read -> verify
# ---------------------------------------------------------------------------


def test_single_xyz_round_trip(tmp_path, ethanol):
    """Write a structure, read it back, verify data integrity."""
    ethanol.energy = -123.456
    out = tmp_path / "round_trip.xyz"
    save_xyz_file(ethanol, str(out))

    loaded = read_xyz(str(out))
    assert loaded.n_atoms == ethanol.n_atoms
    assert loaded.symbols == ethanol.symbols
    for orig, loaded_c in zip(ethanol.coordinates, loaded.coordinates):
        for a, b in zip(orig, loaded_c):
            assert abs(a - b) < 1e-4


def test_multi_xyz_round_trip(tmp_path, methane, ethanol):
    """Write multiple structures, read back, verify count and atoms."""
    methane.energy = -10.0
    ethanol.energy = -20.0
    out = tmp_path / "multi_rt.xyz"
    save_multi_xyz([methane, ethanol], str(out))

    loaded = read_multi_xyz(str(out))
    assert len(loaded) == 2
    assert loaded[0].n_atoms == 5
    assert loaded[1].n_atoms == 9
