"""RGPOT_REGEN_HOSTS check of the scripts that regenerate committed data."""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
OVERRIDES = {
    "regen_dftd_goldens.py": "DFTD_ALLOW_REGEN",
    "regen_expr_goldens.py": "EXPR_ALLOW_REGEN",
    "regen_xckernel_goldens.py": "XCKERNEL_ALLOW_REGEN",
}
NEEDS_NUMPY = {"regen_xckernel_goldens.py"}


def _load(name: str):
    if name in NEEDS_NUMPY:
        pytest.importorskip("numpy")
    spec = importlib.util.spec_from_file_location(name[:-3], SCRIPTS / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("RGPOT_REGEN_HOSTS", raising=False)
    for var in set(OVERRIDES.values()):
        monkeypatch.delenv(var, raising=False)


@pytest.mark.parametrize("name", sorted(OVERRIDES))
def test_refuses_when_unset(name):
    with pytest.raises(SystemExit, match="RGPOT_REGEN_HOSTS"):
        _load(name)._require_regen_host()


@pytest.mark.parametrize("name", sorted(OVERRIDES))
def test_refuses_unlisted_host(name, monkeypatch):
    monkeypatch.setenv("RGPOT_REGEN_HOSTS", f"not-{socket.gethostname()}")
    with pytest.raises(SystemExit, match="RGPOT_REGEN_HOSTS"):
        _load(name)._require_regen_host()


@pytest.mark.parametrize("name", sorted(OVERRIDES))
@pytest.mark.parametrize("short", [False, True])
def test_accepts_listed_host(name, short, monkeypatch):
    host = socket.gethostname()
    if short:
        host = host.split(".")[0]
    monkeypatch.setenv("RGPOT_REGEN_HOSTS", f"workstation-a, {host}")
    _load(name)._require_regen_host()


@pytest.mark.parametrize("name", sorted(OVERRIDES))
def test_override_still_works(name, monkeypatch):
    monkeypatch.setenv(OVERRIDES[name], "1")
    _load(name)._require_regen_host()
    monkeypatch.setenv(OVERRIDES[name], "0")
    with pytest.raises(SystemExit, match="RGPOT_REGEN_HOSTS"):
        _load(name)._require_regen_host()
