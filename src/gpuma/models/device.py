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

    Falls back to ``"cpu"`` when CUDA is requested but unavailable.
    When a specific GPU index is requested but does not exist, falls
    back to ``"cuda:0"`` with a warning.
    """
    dev = (device or "").strip().lower()
    if dev == "cpu":
        return "cpu"
    if dev.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA device requested (%s) but CUDA is not available; falling back to 'cpu'.",
                device,
            )
            return "cpu"
        # Validate GPU index if specified
        if ":" in dev:
            try:
                idx = int(dev.split(":")[1])
            except (ValueError, IndexError):
                logger.warning(
                    "Invalid CUDA device index in '%s'; using default GPU.",
                    device,
                )
                return "cuda"
            num_gpus = torch.cuda.device_count()
            if idx >= num_gpus:
                logger.warning(
                    "Requested GPU %d (via '%s') but only %d GPU(s) available. "
                    "Falling back to cuda:0.",
                    idx,
                    device,
                    num_gpus,
                )
                return "cuda:0"
        return dev
    logger.warning("Unknown device '%s'; falling back to 'cpu'.", device)
    return "cpu"


def _device_for_torch(device: str) -> torch.device:
    """Convert a config device string to a :class:`torch.device`.

    Any invalid or unavailable CUDA specification falls back to CPU.
    """
    normalized = _parse_device_string(device)
    if normalized == "cpu":
        return torch.device("cpu")
    try:
        return torch.device(normalized)
    except (RuntimeError, ValueError):
        logger.warning("Invalid CUDA device '%s'; falling back to 'cpu'.", device)
        return torch.device("cpu")


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
