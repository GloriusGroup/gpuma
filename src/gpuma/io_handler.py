"""Input/Output handler for molecular structures in GPUMA.

This module provides functions for reading molecular structures from various
formats and converting between different representations.
"""

import glob
import logging
import os

from .mol_utils import (
    smiles_to_conformer_ensemble as _smiles_to_ensemble_util,
)
from .mol_utils import (
    smiles_to_structure as _smiles_to_structure_util,
)
from .structure import Structure

logger = logging.getLogger(__name__)


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
    comment = ""

    try:
        with open(file_path, encoding="utf-8") as infile:
            lines = [line.rstrip("\n") for line in infile.readlines()]

        try:
            num_atoms = int(lines[0])
        except Exception as exc:
            raise ValueError("First line must contain the number of atoms as an integer") from exc

        if len(lines) < 2 + num_atoms:
            raise ValueError(f"Expected {num_atoms} atom lines, but found {max(0, len(lines) - 2)}")
        comment = lines[1] if len(lines) > 1 else ""

        for i in range(num_atoms):
            parts = lines[2 + i].split()
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

    return Structure(
        symbols=symbols,
        coordinates=coordinates,
        comment=comment,
        charge=charge,
        multiplicity=multiplicity,
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

    structures: list[Structure] = []

    try:
        with open(file_path, encoding="utf-8") as infile:
            lines = [line.rstrip("\n") for line in infile.readlines()]

        i = 0
        while i < len(lines):
            if lines[i].strip() == "":
                i += 1
                continue

            try:
                num_atoms = int(lines[i].strip())
            except ValueError:
                i += 1
                continue

            if i + 1 + num_atoms >= len(lines):
                break

            comment = lines[i + 1] if (i + 1) < len(lines) else ""

            symbols: list[str] = []
            coordinates: list[tuple[float, float, float]] = []

            valid = True
            for j in range(num_atoms):
                parts = lines[i + 2 + j].split()
                if len(parts) < 4:
                    valid = False
                    break
                symbol = parts[0]
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                except ValueError:
                    valid = False
                    break
                symbols.append(symbol)
                coordinates.append((x, y, z))

            if valid and len(symbols) == num_atoms:
                structures.append(
                    Structure(
                        symbols=symbols,
                        coordinates=coordinates,
                        comment=comment,
                        charge=charge,
                        multiplicity=multiplicity,
                    )
                )

            i = i + 2 + num_atoms

    except Exception as exc:
        raise ValueError(f"Error reading multi-XYZ file: {exc}") from exc

    return structures


def read_xyz_directory(
    directory_path: str, charge: int = 0, multiplicity: int = 1
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

    Returns
    -------
    list[Structure]
        List of structures from all XYZ files in the directory.

    Raises
    ------
    FileNotFoundError
        If the directory does not exist.
    ValueError
        If no valid XYZ files are found.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} not found")

    xyz_files = glob.glob(os.path.join(directory_path, "*.xyz"))

    if not xyz_files:
        raise ValueError(f"No XYZ files found in directory {directory_path}")

    structures: list[Structure] = []

    for xyz_file in xyz_files:
        try:
            structures.append(read_xyz(xyz_file, charge=charge, multiplicity=multiplicity))
        except Exception as exc:  # pragma: no cover - logged and skipped
            logger.warning("Failed to read %s: %s", xyz_file, exc)

    if not structures:
        raise ValueError("No valid structures could be read from any XYZ files")

    return structures


def smiles_to_xyz(smiles_string: str, return_full_xyz_str: bool = False) -> Structure | str:
    """Convert a SMILES string to a :class:`Structure` or an XYZ string.

    Parameters
    ----------
    smiles_string:
        Valid SMILES string representing the molecular structure.
    return_full_xyz_str:
        If ``True``, return an XYZ-format string instead of a
        :class:`Structure` instance.

    Returns
    -------
    Structure | str
        Either a :class:`Structure` or an XYZ string depending on
        ``return_full_xyz_str``.
    """
    if not smiles_string or not smiles_string.strip():
        raise ValueError("SMILES string cannot be empty or None")

    struct = _smiles_to_structure_util(smiles_string.strip())

    if return_full_xyz_str:
        xyz_lines = [str(struct.n_atoms)]
        xyz_lines.append("Generated from SMILES using MORFEUS")
        for atom, coord in zip(struct.symbols, struct.coordinates, strict=True):
            xyz_lines.append(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")
        return "\n".join(xyz_lines)

    struct.comment = f"Generated from SMILES: {smiles_string}"
    return struct


def smiles_to_ensemble(
    smiles_string: str,
    num_conformers: int,
    return_full_xyz_str: bool = False,
):
    """Generate a conformational ensemble from a SMILES string.

    Parameters
    ----------
    smiles_string:
        Valid SMILES string representing the molecular structure.
    num_conformers:
        Maximum number of conformers to generate.
    return_full_xyz_str:
        If ``True``, returns a list of XYZ strings for each conformer.

    Returns
    -------
    list[str] | list[Structure]
        A list of XYZ strings or a list of :class:`Structure` instances.
    """
    if not smiles_string or not smiles_string.strip():
        raise ValueError("SMILES string cannot be empty or None")
    if not isinstance(num_conformers, int) or num_conformers <= 0:
        raise ValueError("num_conformers must be a positive integer")

    conformers: list[Structure] = _smiles_to_ensemble_util(smiles_string.strip(), num_conformers)
    if not return_full_xyz_str:
        for i, struct in enumerate(conformers):
            struct.comment = f"Conformer {i + 1} generated from SMILES using MORFEUS"
        return conformers

    xyz_conformers: list[str] = []
    for i, struct in enumerate(conformers):
        xyz_lines = [str(struct.n_atoms)]
        xyz_lines.append(f"Conformer {i + 1} from SMILES using MORFEUS")
        for atom, coord in zip(struct.symbols, struct.coordinates, strict=True):
            xyz_lines.append(f"{atom} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}")
        xyz_conformers.append("\n".join(xyz_lines))

    return xyz_conformers


def save_xyz_file(structure: Structure, filepath: str) -> None:
    """Save a single :class:`Structure` to an XYZ file.

    Parameters
    ----------
    structure:
        Structure to save. Uses ``structure.energy`` in the comment if set.
    filepath:
        Output file path for the XYZ file.
    """
    if not structure or structure.n_atoms == 0:
        raise ValueError("Cannot save empty structure")

    for i, coord in enumerate(structure.coordinates):
        if len(coord) != 3:
            raise ValueError(f"Coordinate {i} must have exactly 3 components (x, y, z)")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as outfile:
        outfile.write(f"{structure.n_atoms}\n")
        if structure.energy is not None:
            outfile.write(f"{structure.comment} | Energy: {structure.energy:.6f}\n")
        else:
            outfile.write(f"{structure.comment}\n")
        for symbol, coord in zip(structure.symbols, structure.coordinates, strict=True):
            outfile.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")


def save_multi_xyz(
    structures: list[Structure],
    filepath: str,
    comments: list[str] | None = None,
) -> None:
    """Save multiple structures to a single multi-XYZ file.

    Parameters
    ----------
    structures:
        List of :class:`Structure` objects. Energy is optional.
    filepath:
        Output file path for the multi-XYZ file.
    comments:
        Optional list of comments, one for each structure.
    """
    if not structures:
        raise ValueError("Cannot save empty structure list")

    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as outfile:
        for i, struct in enumerate(structures):
            if struct.n_atoms != len(struct.coordinates):
                raise ValueError(f"Structure {i}: symbols/coordinates length mismatch")
            comment = (
                comments[i]
                if comments and i < len(comments)
                else (struct.comment or f"Structure {i + 1}")
            )
            outfile.write(f"{struct.n_atoms}\n")
            if struct.energy is not None:
                outfile.write(f"{comment} | Energy: {struct.energy:.6f} eV\n")
            else:
                outfile.write(f"{comment}\n")
            for symbol, coord in zip(struct.symbols, struct.coordinates, strict=True):
                outfile.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
