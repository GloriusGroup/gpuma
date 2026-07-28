"""Utility subpackage for GPUMA.

Groups cross-cutting helpers that the rest of the package depends on:

- :mod:`gpuma.utils.io_handler` — XYZ read/write and SMILES-to-XYZ helpers.
- :mod:`gpuma.utils.logging_utils` — logging configuration and the
  optimization summary logger.
- :mod:`gpuma.utils.decorators` — timing decorators/context managers.

Public names are re-exported here so ``from gpuma.utils import <name>`` works,
while the top-level :mod:`gpuma` package continues to re-export the same
functions it always has.
"""

from .decorators import TimingCapture, capture_timings, time_it, timed_block
from .io_handler import (
    natural_sort_key,
    read_multi_xyz,
    read_xyz,
    read_xyz_directory,
    save_as_single_xyz_files,
    save_multi_xyz,
    save_xyz_file,
    smiles_to_ensemble,
    smiles_to_xyz,
)
from .logging_utils import configure_logging, log_optimization_summary

__all__ = [
    # decorators
    "time_it",
    "timed_block",
    "capture_timings",
    "TimingCapture",
    # logging
    "configure_logging",
    "log_optimization_summary",
    # io
    "read_xyz",
    "read_multi_xyz",
    "read_xyz_directory",
    "natural_sort_key",
    "smiles_to_xyz",
    "smiles_to_ensemble",
    "save_xyz_file",
    "save_multi_xyz",
    "save_as_single_xyz_files",
]
