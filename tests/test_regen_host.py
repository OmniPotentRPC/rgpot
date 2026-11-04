"""RGPOT_REGEN_HOSTS check of the scripts that regenerate committed data."""

from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
OVERRIDES = {
    "regen_dftd_goldens.py": "DFTD_ALLOW_REGEN",
    "regen_expr_goldens.py": "EXPR_ALLOW_REGEN",
    "regen_xckernel_goldens.py": "XCKERNEL_ALLOW_REGEN",
    "compare_xckernel_tda_nwchemc.py": "XCKERNEL_ALLOW_REGEN",
    "time_lj_rgpot_vs_pyeonclient.py": "TIME_LJ_ALLOW",
}
NEEDS_NUMPY = {"regen_xckernel_goldens.py", "compare_xckernel_tda_nwchemc.py"}


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


def _run_gen_libxckernel(tmp_path: Path, **env_extra: str):
    # A failing python3 stub stops the script at its sympy probe, before it
    # replaces third_party/libxckernel.
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "python3").write_text("#!/bin/sh\nexit 1\n")
    (stub / "python3").chmod(0o755)
    env = dict(os.environ)
    env.pop("RGPOT_REGEN_HOSTS", None)
    env.pop("XCKERNEL_ALLOW_REGEN", None)
    env["PATH"] = f"{stub}{os.pathsep}{env['PATH']}"
    env["LIBXCKERNEL_SRC"] = str(tmp_path)
    env.update(env_extra)
    return subprocess.run(
        ["bash", str(SCRIPTS / "gen_libxckernel.sh")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_gen_libxckernel_refuses_when_unset(tmp_path):
    out = _run_gen_libxckernel(tmp_path)
    assert out.returncode == 2
    assert "RGPOT_REGEN_HOSTS" in out.stderr
    assert "sympy" not in out.stderr


def test_gen_libxckernel_accepts_listed_host(tmp_path):
    host = socket.gethostname().split(".")[0]
    out = _run_gen_libxckernel(tmp_path, RGPOT_REGEN_HOSTS=f"workstation-a,{host}")
    assert out.returncode == 2
    assert "RGPOT_REGEN_HOSTS" not in out.stderr
    assert "need sympy+numpy" in out.stderr


def test_gen_libxckernel_override_still_works(tmp_path):
    out = _run_gen_libxckernel(tmp_path, XCKERNEL_ALLOW_REGEN="1")
    assert out.returncode == 2
    assert "RGPOT_REGEN_HOSTS" not in out.stderr
    assert "need sympy+numpy" in out.stderr
