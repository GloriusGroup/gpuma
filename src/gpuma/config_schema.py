"""JSON Schema of the GPUMA configuration, derived from ``DEFAULT_CONFIG``."""

from __future__ import annotations

from typing import Any

from .config import (
    DEFAULT_CONFIG,
    VALID_BATCH_OPTIMIZERS,
    VALID_MEMORY_SCALES_WITH,
    VALID_MODEL_TYPES,
)

_ENUMS: dict[tuple[str, str], list[str]] = {
    ("model", "model_type"): sorted(VALID_MODEL_TYPES),
    ("optimization", "batch_optimizer"): sorted(VALID_BATCH_OPTIMIZERS),
    ("optimization", "batch_optimization_mode"): ["batch", "sequential"],
    ("technical", "memory_scales_with"): sorted(VALID_MEMORY_SCALES_WITH),
    ("technical", "logging_level"): ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
}

# Keys whose default is None but whose value is not a string.
_NULLABLE_TYPES: dict[tuple[str, str], str] = {
    ("optimization", "energy_convergence_criterion"): "number",
}

_PATTERNS: dict[tuple[str, str], str] = {
    ("technical", "device"): r"^(cpu|cuda(:\d+)?)$",
}


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return "string"


def _property(section: str, key: str, default: Any) -> dict[str, Any]:
    if default is None:
        prop: dict[str, Any] = {"type": [_NULLABLE_TYPES.get((section, key), "string"), "null"]}
    else:
        prop = {"type": _json_type(default)}
    prop["default"] = default
    if (section, key) in _ENUMS:
        prop["enum"] = _ENUMS[(section, key)]
    if (section, key) in _PATTERNS:
        prop["pattern"] = _PATTERNS[(section, key)]
    return prop


def config_json_schema() -> dict[str, Any]:
    """Return the JSON Schema (draft 2020-12) of a GPUMA config file."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "GPUMA configuration",
        "type": "object",
        "additionalProperties": True,
        "properties": {
            section: {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    key: _property(section, key, default) for key, default in keys.items()
                },
            }
            for section, keys in DEFAULT_CONFIG.items()
        },
    }
