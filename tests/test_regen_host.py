"""RGPOT_REGEN_HOSTS check of the scripts that regenerate committed data."""

from __future__ import annotations

import importlib.util
import os
import shutil
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


def _run_gen_libxckernel(
    tmp_path: Path,
    script: Path = SCRIPTS / "gen_libxckernel.sh",
    python3: str = "#!/bin/sh\nexit 1\n",
    **env_extra: str,
):
    # The default python3 stub fails, so the script stops at its sympy probe,
    # before it replaces third_party/libxckernel.
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "python3").write_text(python3)
    (stub / "python3").chmod(0o755)
    env = dict(os.environ)
    for var in ("RGPOT_REGEN_HOSTS", "XCKERNEL_ALLOW_REGEN", "RGPOT_HOST_LABEL"):
        env.pop(var, None)
    env["PATH"] = f"{stub}{os.pathsep}{env['PATH']}"
    env["LIBXCKERNEL_SRC"] = str(tmp_path)
    env.update(env_extra)
    return subprocess.run(
        ["bash", str(script)],
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


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("workstation-a", "workstation-a"),
        ("  lab box 7 ", "lab box 7"),
        (None, "unspecified"),
        ("", "unspecified"),
        (" \t ", "unspecified"),
    ],
)
def test_gen_libxckernel_pin_host_label(tmp_path, label, expected):
    # A copy of the script under tmp_path writes its PIN there, and the stub
    # stands in for the sympy probe and xckernel.catalog.
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    script = repo / "scripts" / "gen_libxckernel.sh"
    shutil.copy(SCRIPTS / "gen_libxckernel.sh", script)
    stub = '#!/bin/sh\nif [ "$1" = "-m" ]; then mkdir -p "$3/src"; fi\n'
    extra = {} if label is None else {"RGPOT_HOST_LABEL": label}
    out = _run_gen_libxckernel(
        tmp_path, script, stub, XCKERNEL_ALLOW_REGEN="1", **extra
    )
    assert out.returncode == 0, out.stderr
    pin = (repo / "third_party" / "libxckernel" / "PIN").read_text()
    committed = (SCRIPTS.parent / "third_party" / "libxckernel" / "PIN").read_text()
    assert pin.splitlines()[:-1] == committed.splitlines()[:-1]
    assert pin.splitlines()[-1] == f"host {expected}"
    full = socket.gethostname()
    for name in {full, full.split(".")[0]}:
        assert name not in pin.split()
        assert f"host {name}" not in pin
