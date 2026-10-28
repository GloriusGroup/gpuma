"""Input/Output handler for molecular structures in GPUMA.

This module provides functions for reading molecular structures from various
formats and converting between different representations.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..conformer_generation import (
    smiles_to_conformer_ensemble as _smiles_to_ensemble_util,
)
from ..conformer_generation import (
    smiles_to_structure as _smiles_to_structure_util,
)
from ..structure import Structure, check_spin_state

if TYPE_CHECKING:  # pragma: no cover - annotation only, avoids importing torch
    from ..config import Config

logger = logging.getLogger(__name__)

# The state gpuma writes into every XYZ comment line (see save_xyz_file).
_STATE_RE = re.compile(r"\| Charge: (-?\d+) \| Multiplicity: (\d+)")


def _state_from_comment(comment: str, charge: int, multiplicity: int) -> tuple[int, int]:
    """Return the charge and multiplicity gpuma wrote into ``comment``, else the given ones."""
    found = _STATE_RE.findall(comment)
    if not found:
        return charge, multiplicity
    return int(found[-1][0]), int(found[-1][1])


def _warn_state_override(where: str, found: tuple[int, int], given: tuple[int, int]) -> None:
    if found != given:
        logger.warning(
            "%s: using Charge %d and Multiplicity %d from the comment line instead of %d and %d",
            where,
            *found,
            *given,
        )


def read_xyz(file_path: str, charge: int = 0, multiplicity: int = 1) -> Structure:
    """Read an XYZ file and return a :class:`Structure` instance.

    Parameters
    ----------
    file_path:
        Path to the XYZ file to read.
    charge:
        Optional total charge to set on the structure (default: ``0``).
    multiplicity:
        Optional spin multiplicity to set (default: ``1``).

    Returns
    -------
    Structure
        Object with symbols, coordinates, and an optional comment.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is invalid.

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    symbols: list[str] = []
    coordinates: list[tuple[float, float, float]] = []

    try:
        with open(file_path, encoding="utf-8") as infile:
            # Read first line: number of atoms
            line = infile.readline()
            try:
                num_atoms = int(line.strip())
            except ValueError as exc:
                raise ValueError(
                    "First line must contain the number of atoms as an integer"
                ) from exc

            # Read second line: comment
            comment_line = infile.readline()
            # If EOF is reached, comment_line is "" (which is falsy)
            # But a blank line "\n" is truthy.
            if not comment_line and num_atoms >= 0:
                # We expected a comment line
                # Note: Original code calculated found as max(0, len(lines) - 2)
                # If we have 1 line, found = 0.
                raise ValueError(f"Expected {num_atoms} atom lines, but found 0")

            comment = comment_line.rstrip("\n")

            for i in range(num_atoms):
                line = infile.readline()
                if not line:
                    raise ValueError(f"Expected {num_atoms} atom lines, but found {i}")

                parts = line.split()
                if len(parts) < 4:
                    raise ValueError(f"Line {i + 3} must contain at least 4 elements: symbol x y z")
                symbol = parts[0]
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError as exc:
                    raise ValueError(f"Invalid coordinates in line {i + 3}: {parts[1:4]}") from exc
                symbols.append(symbol)
                coordinates.append((x, y, z))

    except Exception as exc:
        if isinstance(exc, (FileNotFoundError, ValueError)):
            raise
        raise ValueError(f"Error reading XYZ file: {exc}") from exc

    state = _state_from_comment(comment, charge, multiplicity)
    _warn_state_override(file_path, state, (charge, multiplicity))
    return Structure(
        symbols=symbols,
        coordinates=coordinates,
        comment=comment,
        charge=state[0],
        multiplicity=state[1],
    )


def read_multi_xyz(file_path: str, charge: int = 0, multiplicity: int = 1) -> list[Structure]:
    """Read an XYZ file containing multiple structures.

    Parameters
    ----------
    file_path:
        Path to the multi-structure XYZ file.
    charge:
        Optional total charge to set on all returned structures (default: ``0``).
    multiplicity:
        Optional spin multiplicity to set (default: ``1``).

    Returns
    -------
    list[Structure]
        List of structures read from the file.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If the file format is invalid.

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    try:
        with open(file_path, encoding="utf-8") as infile:
            lines = infile.read().splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Error reading multi-XYZ file: {exc}") from exc

    structures: list[Structure] = []
    overridden: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        where = f"{file_path}: structure {len(structures) + 1} (line {i + 1})"
        try:
            num_atoms = int(lines[i].strip())
        except ValueError:
            raise ValueError(f"{where}: expected an atom count, got {lines[i]!r}") from None
        if num_atoms < 0:
            raise ValueError(f"{where}: negative atom count {num_atoms}")
        if i + 1 >= len(lines):
            raise ValueError(f"{where}: missing comment line")
        comment = lines[i + 1]
        atom_lines = lines[i + 2 : i + 2 + num_atoms]
        if len(atom_lines) < num_atoms:
            raise ValueError(
                f"{where}: expected {num_atoms} atom lines, file ends after {len(atom_lines)}"
            )

        symbols: list[str] = []
        coordinates: list[tuple[float, float, float]] = []
        for offset, line in enumerate(atom_lines):
            parts = line.split()
            try:
                if len(parts) < 4:
                    raise ValueError
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                raise ValueError(
                    f"{where}: line {i + 3 + offset} is not 'symbol x y z': {line!r}"
                ) from None
            symbols.append(parts[0])
            coordinates.append((x, y, z))

        state = _state_from_comment(comment, charge, multiplicity)
        if state != (charge, multiplicity):
            overridden.append(state)
        structures.append(
            Structure(
                symbols=symbols,
                coordinates=coordinates,
                comment=comment,
                charge=state[0],
                multiplicity=state[1],
            )
        )
        i += 2 + num_atoms

    if overridden:
        _warn_state_override(
            f"{file_path} ({len(overridden)} structures)", overridden[0], (charge, multiplicity)
        )
    return structures


def natural_sort_key(filepath: str) -> list:
    """
    Generate a sort key that orders strings with embedded numbers naturally.

    E.g. 'mol_2.xyz' < 'mol_10.xyz' (unlike plain string sort).
    """
    filename = os.path.basename(filepath)
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r'(\d+)', filename)
    ]

def read_xyz_directory(
    directory_path: str,
    charge: int = 0,
    multiplicity: int = 1,
    sort: bool | Callable[[str], list] = True,
) -> list[Structure]:
    """Read all XYZ files from a directory.

    Parameters
    ----------
    directory_path:
        Path to directory containing XYZ files.
    charge:
        Optional total charge to set on all returned structures (default: ``0``).
    multiplicity:
        Optional spin multiplicity to set (default: ``1``).
    sort:
        Controls the order in which files are read. If ``True`` (default),
        files are sorted in natural numerical order (e.g. ``mol_2.xyz``
        before ``mol_10.xyz``). If ``False``, files are read in
        OS-dependent order, which is not guaranteed and may vary between
        runs or platforms. If a callable is provided, it is used as the
        sort key instead (see :func:`sorted`).

    Returns
    -------
    list[Structure]
        List of structures from all XYZ files in the directory, in the
        order determined by ``sort``.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    ValueError
        If no XYZ files are found or one cannot be read.

    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} not found")

    xyz_files = glob.iglob(os.path.join(directory_path, "*.xyz"))

    if sort is True:
        xyz_files = sorted(xyz_files, key=natural_sort_key)
    elif callable(sort):
        xyz_files = sorted(xyz_files, key=sort)
    else:
        xyz_files = list(xyz_files)

    structures: list[Structure] = []

    for xyz_file in xyz_files:
        try:
            structure = read_xyz(xyz_file, charge=charge, multiplicity=multiplicity)
        except ValueError as exc:
            raise ValueError(f"Failed to read {xyz_file}: {exc}") from exc
        structure.metadata["source_file"] = xyz_file
        structures.append(structure)

    if not structures:
        raise ValueError(f"No XYZ files found in directory {directory_path}")

    return structures


def smiles_to_xyz(
    smiles_string: str,
    return_full_xyz_str: bool = False,
    multiplicity: int | None = None,
    config: Config | None = None,
) -> Structure | str:
    """Convert a SMILES string to a :class:`Structure` or an XYZ string.

    Parameters
    ----------
    smiles_string:
        Valid SMILES string representing the molecular structure.
    return_full_xyz_str:
        If ``True``, return an XYZ-format string instead of a
        :class:`Structure` instance.
    multiplicity:
        Spin multiplicity of the structure; ``None`` takes it from ``config``.

    Returns
    -------
    Structure | str
        Either a :class:`Structure` or an XYZ string depending on
        ``return_full_xyz_str``.

    """
    if not smiles_string or not smiles_string.strip():
        raise ValueError("SMILES string cannot be empty or None")

    struct = _smiles_to_structure_util(smiles_string.strip(), config=config)
    if multiplicity is not None:
        struct.multiplicity = int(multiplicity)
    check_spin_state(struct.symbols, struct.charge, struct.multiplicity)

    if return_full_xyz_str:
        xyz_lines = [str(struct.n_atoms)]
        xyz_lines.append(
            "Generated from SMILES | "
            f"Charge: {struct.charge} | "
            f"Multiplicity: {struct.multiplicity}"
        )
        for atom, coord in zip(struct.symbols, struct.coordinates, strict=True):
            xyz_lines.append(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")
        return "\n".join(xyz_lines)

    struct.comment = (
        f"Generated from SMILES: {smiles_string} | "
        f"Charge: {struct.charge} | "
        f"Multiplicity: {struct.multiplicity}"
    )
    return struct


def smiles_to_ensemble(
    smiles_string: str,
    max_num_confs: int,
    multiplicity: int | None = None,
    seed: int | None = None,
    config: Config | None = None,
) -> list[Structure]:
    """Generate conformer ensemble from SMILES.

    Parameters
    ----------
    smiles_string:
        Valid SMILES string representing the molecular structure.
    max_num_confs:
        Maximum number of conformers to generate.
    multiplicity:
        Spin multiplicity of the structures; ``None`` takes it from ``config``.
    seed:
        Random seed for the embedding; ``None`` takes ``conformer_seed`` from ``config``.

    Returns
    -------
    list[Structure]
        A list of :class:`Structure` instances representing the conformers.

    """
    if not smiles_string or not smiles_string.strip():
        raise ValueError("SMILES string cannot be empty or None")

    structs = _smiles_to_ensemble_util(
        smiles_string.strip(),
        max_num_confs,
        multiplicity=multiplicity,
        seed=seed,
        config=config,
    )
    check_spin_state(structs[0].symbols, structs[0].charge, structs[0].multiplicity)
    return structs


def save_xyz_file(structure: Structure, file_path: str) -> None:
    """Save a single :class:`Structure` to an XYZ file.

    The comment line includes the energy (if set), charge, and multiplicity.

    Parameters
    ----------
    structure:
        Structure to write.
    file_path:
        Destination file path.

    """
    lines: list[str] = [str(structure.n_atoms)]
    # include existing comment and ensure energy/charge/multiplicity are visible
    base_comment = structure.comment or ""
    energy_part = ""
    if structure.energy is not None:
        energy_part = f" | Energy: {structure.energy:.6f} eV"
    if structure.converged is False:
        energy_part += " | Not converged"
    state_part = f" | Charge: {structure.charge} | Multiplicity: {structure.multiplicity}"
    comment = (base_comment + energy_part + state_part).strip() or "Structure"
    lines.append(comment)
    for symbol, coord in zip(structure.symbols, structure.coordinates, strict=True):
        lines.append(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")

    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")


def save_multi_xyz(
    structures: list[Structure], file_path: str, comments: list[str] | None = None
) -> None:
    """Save multiple structures to a single multi-structure XYZ file.

    Each structure block includes the energy (if set), charge, and
    multiplicity in the comment line.

    Parameters
    ----------
    structures:
        List of structures to write.
    file_path:
        Destination file path.
    comments:
        Optional per-structure comment strings. Falls back to each
        structure's own comment if not provided.

    """
    lines: list[str] = []
    for idx, struct in enumerate(structures):
        lines.append(str(struct.n_atoms))
        base_comment = ""
        if comments and idx < len(comments):
            base_comment = comments[idx]
        elif struct.comment:
            base_comment = struct.comment
        energy_part = ""
        if struct.energy is not None:
            energy_part = f" | Energy: {struct.energy:.6f} eV"
        if struct.converged is False:
            energy_part += " | Not converged"
        state_part = f" | Charge: {struct.charge} | Multiplicity: {struct.multiplicity}"
        comment = (base_comment + energy_part + state_part).strip() or f"Structure {idx + 1}"
        lines.append(comment)
        for symbol, coord in zip(struct.symbols, struct.coordinates, strict=True):
            lines.append(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")

    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
        fh.write("\n")


def save_as_single_xyz_files(
    structures: list[Structure], output_dir: str, comments: list[str] | None = None
) -> None:
    """Save each structure to its own XYZ file in a directory.

    Files are zero-padded to sort naturally, e.g. ``structure_01.xyz`` for
    up to 99 structures, ``structure_0001.xyz`` for up to 9999, etc.

    Parameters
    ----------
    structures:
        List of structures to save.
    output_dir:
        Directory where files will be written. Created if it does not exist.
    comments:
        Optional per-structure comment strings.

    """
    os.makedirs(output_dir, exist_ok=True)
    width = len(str(len(structures)))
    for idx, struct in enumerate(structures):
        if comments and idx < len(comments):
            struct = Structure(
                symbols=struct.symbols,
                coordinates=struct.coordinates,
                energy=struct.energy,
                charge=struct.charge,
                multiplicity=struct.multiplicity,
                comment=comments[idx],
            )
        file_path = os.path.join(output_dir, f"structure_{idx + 1:0{width}d}.xyz")
        save_xyz_file(struct, file_path)
