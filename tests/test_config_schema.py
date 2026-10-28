"""Tests for the config JSON Schema and ``gpuma config --show/--schema``."""

import json

from gpuma.cli import main
from gpuma.config import (
    DEFAULT_CONFIG,
    VALID_BATCH_OPTIMIZERS,
    VALID_MEMORY_SCALES_WITH,
    VALID_MODEL_TYPES,
)
from gpuma.config_schema import config_json_schema


def test_schema_covers_every_default_key():
    schema = config_json_schema()
    for section, keys in DEFAULT_CONFIG.items():
        assert set(keys) <= set(schema["properties"][section]["properties"])


def test_schema_enums_match_validators():
    p = config_json_schema()["properties"]
    assert set(p["optimization"]["properties"]["batch_optimizer"]["enum"]) == VALID_BATCH_OPTIMIZERS
    assert set(p["model"]["properties"]["model_type"]["enum"]) == VALID_MODEL_TYPES
    scales = p["technical"]["properties"]["memory_scales_with"]["enum"]
    assert set(scales) == VALID_MEMORY_SCALES_WITH


def test_schema_types_and_defaults():
    p = config_json_schema()["properties"]
    assert p["optimization"]["properties"]["charge"] == {"type": "integer", "default": 0}
    assert p["model"]["properties"]["d3_correction"]["type"] == "boolean"
    crit = p["optimization"]["properties"]["energy_convergence_criterion"]
    assert crit["type"] == ["number", "null"]
    assert p["technical"]["properties"]["device"]["pattern"] == r"^(cpu|cuda(:\d+)?)$"
    assert p["model"]["additionalProperties"] is True


def _orb_config(tmp_path, **model):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"model": {"model_type": "orb", **model}}))
    return str(path)


def test_show_applies_overrides(tmp_path, capsys):
    assert main(["config", "--show", "-c", _orb_config(tmp_path), "--device", "cpu"]) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["technical"]["device"] == "cpu"
    assert shown["model"]["model_type"] == "orb"


def test_show_redacts_token(tmp_path, capsys):
    main(["config", "--show", "-c", _orb_config(tmp_path, huggingface_token="hf_secret")])
    out = capsys.readouterr().out
    assert "hf_secret" not in out and json.loads(out)["model"]["huggingface_token"] == "***"


def test_schema_command_prints_schema(capsys):
    assert main(["config", "--schema"]) == 0
    assert json.loads(capsys.readouterr().out) == config_json_schema()


def test_schema_command_with_json_puts_schema_in_summary(capsys):
    assert main(["config", "--schema", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == config_json_schema()


def test_schema_bounds_match_validators():
    tech = config_json_schema()["properties"]["technical"]["properties"]
    assert tech["memory_scaling_factor"]["exclusiveMinimum"] == 1
    assert (tech["max_memory_padding"]["minimum"], tech["max_memory_padding"]["maximum"]) == (
        0.05,
        0.95,
    )


def test_schema_step_bounds_match_validators():
    props = config_json_schema()["properties"]
    assert props["technical"]["properties"]["steps_between_swaps"]["minimum"] == 1
    assert props["optimization"]["properties"]["max_steps"] == {
        "type": "integer", "default": 10_000, "minimum": 1
    }
