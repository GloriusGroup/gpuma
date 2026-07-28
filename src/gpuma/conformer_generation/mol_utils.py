"""Molecular utilities for SMILES processing and structure generation in GPUMA.

This module provides functions for converting SMILES strings to 3D molecular
structures and generating conformer ensembles. The per-molecule helpers here
delegate to :mod:`gpuma.conformer_generation.embed`, which embeds on the GPU via nvMolKit when
``config.technical.device`` requests a CUDA device and falls back to the CPU
:mod:`morfeus` backend (with RDKit) otherwise.
"""

from numbers import Integral
from typing import TYPE_CHECKING

from ase.data import chemical_symbols

from ..decorators import time_it
from ..structure import Structure

if TYPE_CHECKING:  # pragma: no cover - annotation only, avoids importing torch
    from ..config import Config


def _to_symbol_list(elements) -> list[str]:
    """Convert a sequence of element descriptors to a list of atomic symbols.

    The function accepts element symbols as strings or atomic numbers
    (:class:`int` or other :class:`numbers.Integral` types) and converts them
    to a list of string symbols. Numpy arrays are supported transparently.
    """
    try:
        if hasattr(elements, "tolist"):
            elements = elements.tolist()
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        pass

    symbols: list[str] = []
    for elem in elements:
        if isinstance(elem, str):
            symbols.append(elem)
        elif isinstance(elem, Integral):
            try:
                symbols.append(chemical_symbols[int(elem)])
            except Exception as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid atomic number: {elem}") from exc
        else:
            symbols.append(str(elem))
    return symbols


def _to_coord_list(coords) -> list[tuple[float, float, float]]:
    """Convert coordinates to a nested Python list of float tuples."""
    try:
        if hasattr(coords, "tolist"):
            coords = coords.tolist()
    except (AttributeError, TypeError):  # pragma: no cover - defensive
        pass
    return [(float(row[0]), float(row[1]), float(row[2])) for row in coords]


@time_it
def smiles_to_conformer_ensemble(
    smiles: str,
    max_num_confs: int = 5,
    multiplicity: int = 1,
    seed: int | None = None,
    config: "Config | None" = None,
) -> list[Structure]:
    """Generate multiple conformers from a SMILES string.

    Thin wrapper over :func:`gpuma.conformer_generation.embed.generate_ensembles`, which runs on the
    GPU when ``config.technical.device`` asks for one and falls back to morfeus
    on the CPU otherwise. Conformers are pruned by RMSD and sorted by energy.

    For more than one molecule prefer :func:`gpuma.conformer_generation.embed.generate_ensembles`
    directly -- it batches, which is what makes the GPU backend worthwhile.

    Parameters
    ----------
    smiles:
        Valid SMILES string representing the molecular structure.
    max_num_confs:
        Maximum number of conformers to return (default: ``5``).
    multiplicity:
        Spin multiplicity to set on all generated conformers (default: ``1``).
    seed:
        Optional random seed for reproducible conformer generation. This is now
        passed through to the embedding itself; previously it only seeded the
        Python and NumPy RNGs, which RDKit does not consult, so it had no
        effect on the generated geometries.
    config:
        gpuma configuration, used for device selection. Loaded from the default
        location if omitted.

    Returns
    -------
    list[Structure]
        List of conformer structures.

    Raises
    ------
    ValueError
        If the SMILES string is invalid or conformer generation fails.
    ImportError
        If :mod:`morfeus` or RDKit dependencies are not available.

    Notes
    -----
    Conformers are sorted by energy (lowest first) and pruned by RMSD to remove
    duplicates. The actual number returned may be less than ``max_num_confs``.

    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string cannot be empty")

    if max_num_confs <= 0:
        raise ValueError("max_num_confs must be positive")

    try:
        # Imported here rather than at module scope: embed imports this module's
        # coordinate helpers, so a top-level import would be circular.
        from .embed import DEFAULT_SEED, generate_ensembles

        ensembles = generate_ensembles(
            [smiles.strip()],
            max_num_confs=max_num_confs,
            config=config,
            multiplicity=multiplicity,
            seed=DEFAULT_SEED if seed is None else seed,
        )
        structures = ensembles[0]

        if not structures:
            raise ValueError("No valid conformers could be generated from SMILES")

        return structures

    except ImportError as exc:  # pragma: no cover - dependency error
        raise ImportError(
            "Required dependencies not found. Please install with: "
            "uv pip install 'gpuma' or install 'morfeus-ml rdkit'"
        ) from exc
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"Failed to generate conformers from SMILES '{smiles}': {exc}") from exc


def smiles_to_structure(smiles: str, config: "Config | None" = None) -> Structure:
    """Convert a SMILES string to a single 3D molecular structure.

    Returns the lowest-energy conformer. For more than one molecule prefer
    :func:`gpuma.conformer_generation.embed.generate_structures` directly -- it batches, which is
    what makes the GPU backend worthwhile.
    """
    if not smiles or not smiles.strip():
        raise ValueError("SMILES string cannot be empty")

    try:
        ensemble = smiles_to_conformer_ensemble(
            smiles.strip(), max_num_confs=1, config=config
        )

        if not ensemble:
            raise ValueError("No conformers generated from SMILES")

        return ensemble[0]

    except Exception as exc:
        if isinstance(exc, (ValueError, ImportError)):
            raise
        raise ValueError(f"Failed to generate structure from SMILES '{smiles}': {exc}") from exc
