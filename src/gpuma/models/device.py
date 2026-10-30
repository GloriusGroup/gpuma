"""Device-string handling for the GPUMA model backends.

Each backend resolves CUDA device strings slightly differently; these
helpers normalize a config device string and pin the active CUDA device so
that a requested ``cuda:N`` index is honoured by backends that otherwise
default-resolve a bare ``cuda`` to ``cuda:0``.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _parse_device_string(device: str) -> str:
    """Normalize a device string to ``"cpu"`` or ``"cuda[:N]"``.

    Raises :class:`ValueError` for unknown devices, for CUDA without a usable
    GPU and for a GPU index that does not exist.
    """
    dev = (device or "").strip().lower()
    if dev == "cpu":
        return "cpu"
    if not dev.startswith("cuda"):
        raise ValueError(f"Unknown device {device!r}; use 'cpu', 'cuda' or 'cuda:N'.")
    if not torch.cuda.is_available():
        raise ValueError(
            f"CUDA device requested ({device!r}) but CUDA is not available; "
            "run on a GPU node or set device to 'cpu'."
        )
    if dev == "cuda":
        return dev
    _, _, index = dev.partition(":")
    if not index.isdigit():
        raise ValueError(f"Invalid CUDA device {device!r}; use 'cuda' or 'cuda:N'.")
    num_gpus = torch.cuda.device_count()
    if int(index) >= num_gpus:
        raise ValueError(
            f"Requested GPU {device!r} but only {num_gpus} GPU(s) are visible "
            f"(cuda:0 to cuda:{num_gpus - 1})."
        )
    return dev


def _device_for_torch(device: str) -> torch.device:
    """Convert a config device string to a :class:`torch.device`."""
    return torch.device(_parse_device_string(device))


def _setup_fairchem_device(device: str) -> str:
    """Prepare the CUDA device for the Fairchem backend.

    Fairchem only accepts ``"cuda"`` or ``"cpu"`` — not ``"cuda:N"``.
    When a specific GPU index is requested (e.g. ``"cuda:1"``), this
    function calls :func:`torch.cuda.set_device` so that Fairchem's
    internal device resolution picks the correct GPU.

    Returns
    -------
    str
        ``"cuda"`` or ``"cpu"`` — safe to pass to Fairchem APIs.
    """
    normalized = _parse_device_string(device)
    if not normalized.startswith("cuda"):
        return "cpu"
    if ":" in normalized:
        idx = int(normalized.split(":")[1])
        torch.cuda.set_device(idx)
        logger.info("Selected GPU %d for Fairchem backend.", idx)
    return "cuda"


def _setup_orb_device(device: str) -> None:
    """Prepare the CUDA device for the ORB backend.

    ORB's pretrained loaders and ``OrbTorchSimModel`` default-resolve a
    bare ``"cuda"`` string (or no device at all) to ``cuda:0`` via
    :func:`torch.device`, regardless of what GPU index the caller
    requested. When a specific GPU index is requested (e.g. ``"cuda:1"``),
    this function calls :func:`torch.cuda.set_device` so that any later
    internal ``.to("cuda")`` calls inside orb-models pick the correct GPU.

    No-op for CPU / non-CUDA targets.
    """
    normalized = _parse_device_string(device)
    if not normalized.startswith("cuda"):
        return
    if ":" in normalized:
        idx = int(normalized.split(":")[1])
        torch.cuda.set_device(idx)
        logger.info("Selected GPU %d for ORB backend.", idx)


def _setup_sevennet_device(device: str) -> str:
    """Prepare the CUDA device for the SevenNet backend.

    SevenNet's ``SevenNetModel`` / ``SevenNetCalculator`` resolve a device
    string via :func:`torch.device`, so a ``"cuda:N"`` string is honoured
    directly. We additionally call :func:`torch.cuda.set_device` so that any
    internal ``.to("cuda")`` calls and the D3 kernels pick the requested GPU.

    Returns
    -------
    str
        The normalized device string (``"cpu"`` or ``"cuda[:N]"``), safe to
        pass to the SevenNet APIs.
    """
    normalized = _parse_device_string(device)
    if normalized.startswith("cuda") and ":" in normalized:
        idx = int(normalized.split(":")[1])
        torch.cuda.set_device(idx)
        logger.info("Selected GPU %d for SevenNet backend.", idx)
    return normalized


def _setup_aimnet_device(device: str) -> str:
    """Prepare the CUDA device for the AIMNet2 backend.

    ``AIMNet2Calculator`` honours ``"cuda:N"``; the current device is also set
    so its Warp neighbor-list kernels run on the requested GPU.

    Returns
    -------
    str
        The normalized device string (``"cpu"`` or ``"cuda[:N]"``).
    """
    normalized = _parse_device_string(device)
    if normalized.startswith("cuda") and ":" in normalized:
        idx = int(normalized.split(":")[1])
        torch.cuda.set_device(idx)
        logger.info("Selected GPU %d for AIMNet2 backend.", idx)
    return normalized
