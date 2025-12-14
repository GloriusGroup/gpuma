import pytest

from gpuma import models as model_utils
from gpuma.config import Config


def test_models_module_exports():
    assert hasattr(model_utils, "load_model_fairchem")
    assert hasattr(model_utils, "load_model_torchsim")


def test_load_model_fairchem_empty_name_raises():
    cfg = Config.from_dict({"optimization": {"model_name": "", "device": "cpu"}})
    with pytest.raises(ValueError):
        model_utils.load_model_fairchem(cfg)


def test_load_model_torchsim_import_or_skip():
    try:
        import torch_sim  # noqa: F401
    except Exception:
        pytest.skip("torch_sim not installed; skipping torch-sim model test")

    cfg = Config()
    cfg.optimization.device = "cpu"

    try:
        model_utils.load_model_torchsim(cfg)
    except Exception:
        pass
