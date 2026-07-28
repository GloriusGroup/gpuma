"""SevenNet backend loaders.

Provides the ASE-calculator and torch-sim model loaders for the SevenNet
family.  SevenNet ships its own native D3 implementation
(``SevenNetD3Calculator`` / ``SevenNetD3Model``), reused here when
``config.model.d3_correction`` is enabled.  Because gpuma's batch pipeline
runs in ``float64`` while SevenNet is ``float32``-only, the torch-sim model
is wrapped in ``sevenn``'s ``Float64Wrapper``.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import Config
from .base import (
    AVAILABLE_SEVENNET_MODELS,
    _verify_model_name_and_cache_dir,
    _verify_model_path,
)
from .device import _setup_sevennet_device

logger = logging.getLogger(__name__)


def _resolve_sevennet_model_arg(config: Config) -> tuple[Any, str | None]:
    """Return ``(model_arg, modal)`` for the SevenNet backend.

    ``model_arg`` is either a local checkpoint :class:`~pathlib.Path` (when
    ``config.model.model_path`` points at an existing file) or a validated
    pretrained model-name string from :data:`AVAILABLE_SEVENNET_MODELS`.

    ``modal`` is the optional multi-modal fidelity selector
    (``config.model.model_modal``), required for checkpoints such as
    ``7net-mf-ompa`` and ``7net-omni``.
    """
    modal = config.model.get("model_modal", None)
    modal = str(modal) if modal else None

    model_path = _verify_model_path(config)
    if model_path is not None:
        return model_path, modal

    model_name, _ = _verify_model_name_and_cache_dir(config)
    if model_name not in AVAILABLE_SEVENNET_MODELS:
        raise ValueError(
            f"Unknown SevenNet model name {model_name!r}. "
            f"Must be one of {list(AVAILABLE_SEVENNET_MODELS)}, or supply a "
            "local checkpoint via config.model.model_path."
        )
    return model_name, modal


def _load_sevennet_calculator(config: Config) -> Any:
    """Load a ``SevenNetCalculator`` from a pretrained SevenNet model.

    When ``config.model.d3_correction`` is True the ``SevenNetD3Calculator``
    is used instead, which adds SevenNet's native DFT-D3 energy/force/stress
    contributions on top of every prediction.
    """
    # Validate the model name/path before importing sevenn so that an unknown
    # model raises a clear ValueError even when the optional dep is missing.
    model_arg, modal = _resolve_sevennet_model_arg(config)

    try:
        from sevenn.calculator import (  # type: ignore
            SevenNetCalculator,
            SevenNetD3Calculator,
        )
    except ImportError as exc:
        raise ImportError(
            "sevenn>=0.12.1 is required for SevenNet model support. "
            "Install it with: pip install 'sevenn[torchsim]>=0.12.1'"
        ) from exc

    device = _setup_sevennet_device(str(config.technical.device))

    if config.model.d3_correction:
        functional = str(config.model.d3_functional).lower()
        logger.info(
            "Applying SevenNet native D3 dispersion correction (functional=%s)",
            functional,
        )
        return SevenNetD3Calculator(
            model=model_arg,
            device=device,
            modal=modal,
            functional_name=functional,
        )
    return SevenNetCalculator(model=model_arg, device=device, modal=modal)


def _load_sevennet_torchsim(config: Config) -> Any:
    """Load a SevenNet torch-sim model for batch optimization.

    SevenNet only supports ``float32`` while gpuma's batch pipeline runs in
    ``float64``; the model is therefore wrapped in ``sevenn``'s
    ``Float64Wrapper`` so that state tensors are cast to ``float32`` around
    the model and outputs cast back to ``float64``.

    When ``config.model.d3_correction`` is True, ``SevenNetD3Model`` (SevenNet
    plus its native batched D3 kernel) is used before wrapping.
    """
    # Validate the model name/path before importing sevenn so that an unknown
    # model raises a clear ValueError even when the optional dep is missing.
    model_arg, modal = _resolve_sevennet_model_arg(config)

    try:
        from sevenn.torchsim import (  # type: ignore
            Float64Wrapper,
            SevenNetD3Model,
            SevenNetModel,
        )
    except ImportError as exc:
        raise ImportError(
            "sevenn[torchsim]>=0.12.1 is required for SevenNet batch support. "
            "Install it with: pip install 'sevenn[torchsim]>=0.12.1'"
        ) from exc

    device = _setup_sevennet_device(str(config.technical.device))

    if config.model.d3_correction:
        functional = str(config.model.d3_functional).lower()
        logger.info(
            "Applying SevenNet native D3 dispersion correction (functional=%s)",
            functional,
        )
        model = SevenNetD3Model(
            model_arg,
            modal=modal,
            device=device,
            functional_name=functional,
        )
    else:
        model = SevenNetModel(model_arg, modal=modal, device=device)

    # gpuma builds float64 batched states; SevenNet is float32-only.
    return Float64Wrapper(model)
