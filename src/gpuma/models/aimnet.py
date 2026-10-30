"""AIMNet2 backend loaders and input checks.

Provides the ASE-calculator and torch-sim model loaders for the AIMNet2
family (``aimnet`` package).  Every AIMNet2 model includes D3 dispersion,
so ``config.model.d3_correction`` is rejected.  AIMNet2 computes in
``float32``; like SevenNet, the torch-sim model is wrapped in ``sevenn``'s
generic ``Float64Wrapper`` for gpuma's ``float64`` batch pipeline.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from ..config import Config
from .base import AIMNET_ELEMENTS, AVAILABLE_AIMNET_MODELS, _verify_model_name_and_cache_dir
from .device import _setup_aimnet_device

logger = logging.getLogger(__name__)

#: Models trained on open-shell data; the others ignore the multiplicity.
OPEN_SHELL_AIMNET_MODELS: frozenset[str] = frozenset({"aimnet2-nse"})

#: Models whose geminal H–H repulsion collapses below ~0.75 Å (aimnetcentral #139).
SHORT_HH_AIMNET_MODELS: frozenset[str] = frozenset({"aimnet2", "aimnet2-2025"})

#: Upstream's rejection threshold for starting geometries.
MIN_HH_DISTANCE: float = 0.8


def _check_model_name(model_name: str) -> None:
    """Raise unless ``model_name`` is one of :data:`AVAILABLE_AIMNET_MODELS`."""
    if model_name not in AVAILABLE_AIMNET_MODELS:
        raise ValueError(
            f"Unknown AIMNet model name {model_name!r}. "
            f"Must be one of {list(AVAILABLE_AIMNET_MODELS)}"
        )


def _shortest_hh_distance(symbols: list[str], coordinates: Any) -> float | None:
    """Return the shortest H–H distance, or ``None`` without two H or without heavy atoms."""
    is_h = np.array([s == "H" for s in symbols])
    if is_h.sum() < 2 or is_h.all():
        return None
    xyz = np.asarray(coordinates, dtype=float)[is_h]
    distances = np.linalg.norm(xyz[:, None] - xyz[None], axis=-1)
    return float(distances[np.triu_indices(len(xyz), 1)].min())


def check_aimnet_structure(
    model_name: str, symbols: list[str], multiplicity: int, coordinates: Any = None
) -> None:
    """Raise if a structure is outside what ``model_name`` describes correctly.

    The H–H contact check needs ``coordinates`` and is skipped without them.
    """
    _check_model_name(model_name)
    unsupported = sorted(set(symbols) - set(AIMNET_ELEMENTS))
    if unsupported:
        raise ValueError(
            f"Element(s) {', '.join(unsupported)} not supported by {model_name}; "
            f"it covers {', '.join(AIMNET_ELEMENTS)}"
        )
    if int(multiplicity) != 1 and model_name not in OPEN_SHELL_AIMNET_MODELS:
        raise ValueError(
            f"{model_name} is closed-shell only (multiplicity {multiplicity}); "
            "use aimnet2-nse for open-shell structures"
        )
    if coordinates is not None and model_name in SHORT_HH_AIMNET_MODELS:
        shortest = _shortest_hh_distance(symbols, coordinates)
        if shortest is not None and shortest < MIN_HH_DISTANCE:
            raise ValueError(
                f"H–H distance {shortest:.2f} Å is below {MIN_HH_DISTANCE} Å, where "
                f"{model_name} has a spurious minimum; fix the starting geometry"
            )


def _load_aimnet_base(config: Config) -> Any:
    """Return an ``AIMNet2Calculator`` for the configured model (ensemble member 0)."""
    model_name, cache_dir = _verify_model_name_and_cache_dir(config)
    _check_model_name(model_name)
    if cache_dir is not None:
        # AIMNet reads its download directory from this variable only.
        os.environ["AIMNET_CACHE_DIR"] = str(cache_dir)

    from aimnet.calculators import AIMNet2Calculator  # type: ignore

    device = _setup_aimnet_device(str(config.technical.device))
    return AIMNet2Calculator(model_name, device=device)


def _load_aimnet_calculator(config: Config) -> Any:
    """Load an ``AIMNet2ASE`` calculator; it reads charge and multiplicity from ``atoms.info``."""
    from aimnet.calculators import AIMNet2ASE  # type: ignore

    return AIMNet2ASE(_load_aimnet_base(config))


def _load_aimnet_torchsim(config: Config) -> Any:
    """Load an ``AIMNet2TorchSim`` model wrapped for ``float64`` batch states."""
    from aimnet.calculators import AIMNet2TorchSim  # type: ignore
    from sevenn.torchsim import Float64Wrapper  # type: ignore

    return Float64Wrapper(AIMNet2TorchSim(_load_aimnet_base(config)))
