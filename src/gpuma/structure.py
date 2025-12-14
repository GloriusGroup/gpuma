from dataclasses import dataclass, field
from typing import Any


@dataclass
class Structure:
    """Container for a molecular structure used in GPUMA.

    Attributes
    ----------
    symbols:
        List of atomic symbols.
    coordinates:
        ``N x 3`` list of floats for atomic positions in Angstrom.
    charge:
        Total charge of the system.
    multiplicity:
        Spin multiplicity of the system.
    energy:
        Optional energy value of the structure in eV.
    comment:
        Optional comment or metadata string.
    metadata:
        Free-form metadata dictionary for additional information.
    """

    symbols: list[str]
    coordinates: list[tuple[float, float, float]]
    charge: int
    multiplicity: int
    energy: float | None = None
    comment: str = ""

    # Room for future metadata without breaking the public API
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        """Return the number of atoms in the structure."""
        return len(self.symbols)

    def with_energy(self, energy: float | None) -> "Structure":
        """Set the energy of the structure and return the modified instance.

        Parameters
        ----------
        energy:
            Energy value in eV to assign to this structure. ``None`` clears the
            current energy.

        Returns
        -------
        Structure
            The same :class:`Structure` instance, to allow method chaining.
        """
        self.energy = energy
        return self
