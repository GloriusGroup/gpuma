"""Tests for device parsing and setup helpers."""

import pytest
import torch

from gpuma.models import (
    _device_for_torch,
    _parse_device_string,
    _setup_fairchem_device,
)

from conftest import requires_gpu


class TestParseDeviceString:
    """Tests for _parse_device_string — normalizes device strings."""

    def test_cpu(self):
        """'cpu' and 'CPU' both normalize to 'cpu'."""
        assert _parse_device_string("cpu") == "cpu"
        assert _parse_device_string("CPU") == "cpu"

    @requires_gpu
    def test_cuda(self):
        """'cuda' is accepted when GPU is available."""
        assert _parse_device_string("cuda") == "cuda"

    @requires_gpu
    def test_cuda_with_valid_index(self):
        """'cuda:0' is accepted as-is."""
        assert _parse_device_string("cuda:0") == "cuda:0"

    def test_missing_gpu_index_raises(self, monkeypatch):
        """An index beyond the visible GPUs is an error, not a silent cuda:0."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        with pytest.raises(ValueError, match=r"cuda:5.*2 GPU"):
            _parse_device_string("cuda:5")

    def test_cuda_requested_without_cuda_raises(self, monkeypatch):
        """CUDA requested without a usable GPU is an error, not a silent CPU run."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        with pytest.raises(ValueError, match="CUDA is not available"):
            _parse_device_string("cuda")

    def test_cpu_default_config_still_resolves(self, monkeypatch):
        """The CPU default keeps working on machines without CUDA."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert _parse_device_string("cpu") == "cpu"

    @pytest.mark.parametrize("device", ["tpu", ""])
    def test_unknown_device_raises(self, device):
        """Unknown device strings are an error."""
        with pytest.raises(ValueError, match="Unknown device"):
            _parse_device_string(device)

    def test_non_integer_index_raises(self, monkeypatch):
        """A non-integer CUDA index is an error."""
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        with pytest.raises(ValueError, match="cuda:abc"):
            _parse_device_string("cuda:abc")


class TestDeviceForTorch:
    """Tests for _device_for_torch — creates torch.device objects."""

    def test_cpu(self):
        """'cpu' returns torch.device('cpu')."""
        assert _device_for_torch("cpu") == torch.device("cpu")

    @requires_gpu
    def test_cuda(self):
        """'cuda:0' returns a CUDA torch.device with index 0."""
        d = _device_for_torch("cuda:0")
        assert d.type == "cuda"
        assert d.index == 0


class TestSetupFairchemDevice:
    """Tests for _setup_fairchem_device — prepares device for Fairchem backend."""

    def test_cpu(self):
        """CPU input returns 'cpu'."""
        assert _setup_fairchem_device("cpu") == "cpu"

    @requires_gpu
    def test_plain_cuda(self):
        """Plain 'cuda' returns 'cuda' without calling set_device."""
        assert _setup_fairchem_device("cuda") == "cuda"

    @requires_gpu
    def test_cuda_with_index(self):
        """'cuda:0' calls set_device and returns plain 'cuda'."""
        assert _setup_fairchem_device("cuda:0") == "cuda"
