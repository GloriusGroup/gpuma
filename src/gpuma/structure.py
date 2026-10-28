"""Data container for molecular structures in GPUMA."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Structure:
    """Container for a molecular structure used in GPUMA.

    Attributes
    ----------
    symbols : list[str]
        List of atomic symbols.
    coordinates : list[tuple[float, float, float]]
        ``N x 3`` list of floats for atomic positions in Angstrom.
    charge : int
        Total charge of the system.
    multiplicity : int
        Spin multiplicity of the system.
    energy : float | None
        Optional energy value of the structure in eV.
    comment : str
        Optional comment or metadata string.
    metadata : dict
        Free-form metadata dictionary for additional information.
    converged : bool | None
        Whether the optimization met its criterion; ``None`` if not optimized.
    """

    symbols: list[str]
    coordinates: list[tuple[float, float, float]]
    charge: int
    multiplicity: int
    energy: float | None = None
    comment: str = ""

    # Room for future metadata without breaking the public API
    metadata: dict[str, Any] = field(default_factory=dict)
    converged: bool | None = None

    @property
    def n_atoms(self) -> int:
        """Return the number of atoms in the structure.

        Returns
        -------
        int
            Number of atoms.
        """
        return len(self.symbols)

    def with_energy(self, energy: float | None) -> "Structure":
        """Set the energy of the structure and return the modified instance.

        Parameters
        ----------
        energy : float | None
            Energy value in eV to assign to this structure. ``None`` clears
            the current energy.

        Returns
        -------
        Structure
            The same :class:`Structure` instance, to allow method chaining.
        """
        self.energy = energy
        return self


def check_spin_state(symbols: list[str], charge: int, multiplicity: int) -> None:
    """Raise if ``charge`` and ``multiplicity`` are impossible for these atoms."""
    from ase.data import atomic_numbers

    unknown = sorted({s for s in symbols if s not in atomic_numbers})
    if unknown:
        raise ValueError(f"Unknown element symbol(s): {', '.join(unknown)}")
    electrons = sum(atomic_numbers[s] for s in symbols) - int(charge)
    unpaired = int(multiplicity) - 1
    if electrons < 0 or unpaired > electrons or (electrons - unpaired) % 2:
        raise ValueError(
            f"{electrons} electrons (charge {charge}) can't have multiplicity {multiplicity}; "
            "set the multiplicity (or charge) to match"
        )
