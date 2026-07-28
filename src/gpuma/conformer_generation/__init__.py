"""Conformer generation for GPUMA.

This subpackage groups the SMILES-to-3D-geometry logic:

- :mod:`gpuma.conformer_generation.mol_utils` — per-molecule helpers
  (:func:`smiles_to_structure`, :func:`smiles_to_conformer_ensemble`) and
  the symbol/coordinate converters.
- :mod:`gpuma.conformer_generation.embed` — the batched, GPU-accelerated
  embedding backend (:func:`generate_structures`, :func:`generate_ensembles`).

The two modules are mutually coupled: ``mol_utils`` delegates to ``embed``,
and ``embed`` reuses ``mol_utils``' coordinate converters.
"""

from .embed import DEFAULT_SEED, generate_ensembles, generate_structures
from .mol_utils import smiles_to_conformer_ensemble, smiles_to_structure

__all__ = [
    "DEFAULT_SEED",
    "generate_ensembles",
    "generate_structures",
    "smiles_to_conformer_ensemble",
    "smiles_to_structure",
]
