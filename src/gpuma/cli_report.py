"""Exit codes and the machine-readable ``--json`` run summary of the CLI."""

from __future__ import annotations

import contextlib
import ctypes
import os
import sys
import traceback
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .config import Config
from .models.device import _parse_device_string
from .structure import Structure

EXIT_OK = 0
EXIT_FAILURE = 1
EXIT_USAGE = 2
EXIT_INVALID_INPUT = 3
EXIT_PARTIAL = 4

SUMMARY_SCHEMA_VERSION = 1

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


@dataclass
class CommandResult:
    """Outcome of one CLI command, rendered by :func:`build_summary`."""

    command: str
    status: str  # "ok" | "partial" | "error" | "dry_run"
    exit_code: int
    output_file: str | None = None
    n_input: int | None = None
    structures: list[dict[str, Any]] = field(default_factory=list)
    error: dict[str, str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def gpuma_version() -> str | None:
    """Return the installed gpuma version, or ``None`` when not installed."""
    try:
        return version("gpuma")
    except PackageNotFoundError:
        return None


def structure_record(index: int, s: Structure) -> dict[str, Any]:
    """Return the JSON record for one returned structure."""
    return {
        "index": index,
        "n_atoms": s.n_atoms,
        "charge": s.charge,
        "multiplicity": s.multiplicity,
        "energy_ev": s.energy,
        "comment": s.comment,
    }


def redacted_config(config: Config) -> dict[str, Any]:
    """Return the config as a dict with the Hugging Face token masked."""
    data = config.to_dict()
    if data.get("model", {}).get("huggingface_token"):
        data["model"]["huggingface_token"] = "***"
    return data


def _raised_by_gpuma(exc: BaseException) -> bool:
    frames = traceback.extract_tb(exc.__traceback__)
    return not frames or os.path.abspath(frames[-1].filename).startswith(_PACKAGE_DIR)


def exit_code_for(exc: BaseException) -> int:
    """Map an exception to the CLI exit code; only gpuma's own input checks give 3."""
    invalid = isinstance(exc, (ValueError, FileNotFoundError, FileExistsError))
    return EXIT_INVALID_INPUT if invalid and _raised_by_gpuma(exc) else EXIT_FAILURE


def _resolved_device(config: Config) -> str:
    raw = str(config.technical.device)
    try:
        return _parse_device_string(raw)
    except ValueError:
        return raw


def build_summary(
    result: CommandResult,
    config: Config | None,
    wall_time_s: float,
    diagnostics: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return the single JSON object printed by ``--json``."""
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "gpuma_version": gpuma_version(),
        "command": result.command,
        "status": result.status,
        "exit_code": result.exit_code,
        "output_file": result.output_file,
        "device": _resolved_device(config) if config is not None else None,
        "model": (
            {"type": config.model.model_type, "name": config.model.model_name}
            if config is not None
            else None
        ),
        "config": redacted_config(config) if config is not None else None,
        "n_input": result.n_input,
        "n_returned": len(result.structures),
        "structures": result.structures,
        "wall_time_s": round(wall_time_s, 3),
        "diagnostics": diagnostics,
        "error": result.error,
    }
    summary.update(result.extra)
    return summary


@contextlib.contextmanager
def stdout_to_stderr():
    """Send everything printed during a ``--json`` run to stderr, Python and C level."""
    sys.stdout.flush()
    try:
        saved_fd = os.dup(1)
        os.dup2(2, 1)
    except OSError:
        saved_fd = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        # Drain Python and libc buffers while fd 1 still points at stderr.
        for stream in (sys.stdout, sys.__stdout__):
            if stream is not None:
                stream.flush()
        with contextlib.suppress(OSError, AttributeError):
            ctypes.CDLL(None).fflush(None)
        if saved_fd is not None:
            os.dup2(saved_fd, 1)
            os.close(saved_fd)
