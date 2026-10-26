"""Choosing and checking the GPU, with torch's view of the GPUs faked."""

import pytest
import torch

from cfos_mapper import gpu


@pytest.fixture
def no_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", "12.8")


def test_gpu_count_without_nvidia_smi(monkeypatch):
    monkeypatch.setattr(gpu.shutil, "which", lambda name: None)
    assert gpu.gpu_count() == 0


def test_chosen_gpu_that_does_not_exist_stops(no_cuda, monkeypatch):
    monkeypatch.setattr(gpu, "gpu_count", lambda: 1)
    with pytest.raises(RuntimeError, match="GPU 0 to 0"):
        gpu.check_gpu("1")


def test_no_gpu_chosen_and_none_there_only_warns(no_cuda, monkeypatch, capsys):
    monkeypatch.setattr(gpu, "gpu_count", lambda: 0)
    gpu.check_gpu(None)
    assert "running on the CPU" in capsys.readouterr().out


def test_cpu_only_torch_is_named(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.version, "cuda", None)
    with pytest.raises(RuntimeError, match="CPU-only"):
        gpu.check_gpu("0")


def test_usable_gpu_says_nothing(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    gpu.check_gpu("0")
    assert capsys.readouterr().out == ""
