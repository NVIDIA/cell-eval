"""Tests for the unified cpu/gpu backend dispatcher."""

from __future__ import annotations

import os

import pytest

from cell_eval._backend import ENV_VAR, resolve_backend


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(ENV_VAR, raising=False)


def test_default_is_cpu():
    assert resolve_backend(None) == "cpu"


@pytest.mark.parametrize("alias", ["cpu", "illico", "CPU", "Illico"])
def test_cpu_aliases(alias):
    assert resolve_backend(alias) == "cpu"


@pytest.mark.parametrize("alias", ["gpu", "rsc", "GPU", "RSC"])
def test_gpu_aliases(alias):
    assert resolve_backend(alias) == "gpu"


def test_env_var_used_when_kwarg_none(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "gpu")
    assert resolve_backend(None) == "gpu"


def test_kwarg_overrides_env(monkeypatch):
    monkeypatch.setenv(ENV_VAR, "gpu")
    assert resolve_backend("cpu") == "cpu"


def test_invalid_value_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        resolve_backend("tpu")
