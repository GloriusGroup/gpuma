"""Controlled imports: the scientific dependencies are unavailable on this host."""

import importlib
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest


@contextmanager
def isolated_modules():
    roots = ("gpuma", "ase", "torch", "torch_sim", "morfeus", "nvmolkit")
    before = {name: mod for name, mod in sys.modules.items() if name.split(".")[0] in roots}
    parents = {name: dict(vars(mod)) for name, mod in before.items()}
    root = Path(__file__).parents[2] / "src" / "gpuma"
    for name in tuple(sys.modules):
        if name == "gpuma" or name.startswith("gpuma."):
            del sys.modules[name]
    package = types.ModuleType("gpuma")
    package.__path__ = [str(root)]
    sys.modules["gpuma"] = package
    child = types.ModuleType("gpuma.conformer_generation")
    child.__path__ = [str(root / "conformer_generation")]
    sys.modules[child.__name__] = child
    package.conformer_generation = child
    utils = types.ModuleType("gpuma.utils")
    utils.__path__ = [str(root / "utils")]
    sys.modules[utils.__name__] = utils
    package.utils = utils
    try:
        yield lambda name: importlib.import_module("gpuma." + name)
    finally:
        for name in set(sys.modules) - set(before):
            if name.split(".")[0] in roots:
                del sys.modules[name]
        for name, mod in before.items():
            sys.modules[name] = mod
        for name, attrs in parents.items():
            mod = before[name]
            for key in set(vars(mod)) - set(attrs):
                delattr(mod, key)
            vars(mod).update(attrs)


@pytest.fixture
def modules():
    with isolated_modules() as load:
        yield load
