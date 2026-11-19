"""Ship the Metatomic **dlopen** path: load installed engine, real forces."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

rgpot = pytest.importorskip("rgpot")


def _env_file(name: str) -> Path | None:
    """Return a set path. Skip is not used here: a missing set path fails."""
    if name not in os.environ:
        return None
    raw = os.environ[name].strip()
    path = Path(raw)
    if not raw or not path.is_file():
        pytest.fail(f"{name} is set to {raw!r}, which is not a file")
    return path


def _packaged_engine() -> Path | None:
    root = Path(rgpot.__file__).resolve().parent
    for candidate in (
        root.parent / ".rgpot.mesonpy.libs" / "libmetatomic_engine.so",
        root / "lib" / "libmetatomic_engine.so",
        root / "libmetatomic_engine.so",
    ):
        if candidate.is_file():
            return candidate
    lib_root = root / "lib"
    if lib_root.is_dir():
        for directory in sorted(lib_root.glob("torch-*")):
            for name in ("librgpot_metatomic_engine.so", "libmetatomic_engine.so"):
                candidate = directory / name
                if candidate.is_file():
                    return candidate
    return None


def _engine_path() -> Path | None:
    for name in ("RGPOT_METATOMIC_ENGINE", "METATOMIC_ENGINE"):
        env = _env_file(name)
        if env is not None:
            return env
    packaged = _packaged_engine()
    if packaged is not None:
        return packaged
    # default_metatomic_engine_path also reads the env vars. Those were
    # unset or already returned above, so a hit here is a packaged file.
    bundled = rgpot.default_metatomic_engine_path()
    if bundled and Path(bundled).is_file():
        return Path(bundled)
    return None


def _require_engine() -> Path:
    eng = _engine_path()
    if eng is None:
        pytest.skip(
            "libmetatomic_engine.so is absent (optional metatomic engine library)"
        )
    return eng


def _model_path() -> Path | None:
    env = _env_file("RGPOT_METATOMIC_MODEL")
    if env is not None:
        return env
    candidates = [
        Path.home()
        / "Git/Github/lab-cosmo/atomistic-cookbook/examples/eon-pet-neb/models/pet-mad-xs-v1.5.0.pt",
        Path.home()
        / "Git/Github/epfl/pixi_envs/atomistic-cookbook/atomistic-cookbook/examples/eon-pet-neb/models/pet-mad-xs-v1.5.0.pt",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def _rpath_lines(so: Path) -> list[str]:
    out = subprocess.check_output(["readelf", "-d", str(so)], text=True)
    return [
        ln
        for ln in out.splitlines()
        if "runpath" in ln.lower() or "rpath" in ln.lower()
    ]


def test_has_dlopen_frontend():
    assert rgpot.has_metatomic_dlopen is True
    assert callable(rgpot.evaluate_metatomic_dlopen)
    assert callable(rgpot.evaluate_metatomic)


def test_engine_env_set_but_missing_is_an_error(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "absent-engine.so"
    monkeypatch.setenv("RGPOT_METATOMIC_ENGINE", str(missing))
    with pytest.raises(pytest.fail.Exception, match="not a file"):
        _engine_path()


def test_model_env_set_but_missing_is_an_error(monkeypatch, tmp_path: Path) -> None:
    missing = tmp_path / "absent-model.pt"
    monkeypatch.setenv("RGPOT_METATOMIC_MODEL", str(missing))
    with pytest.raises(pytest.fail.Exception, match="not a file"):
        _model_path()


def test_engine_plugin_in_install_or_env():
    eng = _require_engine()
    env = os.environ.get("RGPOT_METATOMIC_ENGINE")
    if env is not None:
        assert eng.resolve() == Path(env.strip()).resolve()
    else:
        root = Path(rgpot.__file__).resolve().parent
        assert root == eng.parent or root in eng.parents
    magic = eng.read_bytes()[:4]
    assert magic in {
        b"\x7fELF",
        b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe",
        b"\xfe\xed\xfa\xce",
        b"\xce\xfa\xed\xfe",
    }


def test_engine_is_portable_plugin():
    eng = _require_engine()
    if not shutil.which("readelf"):
        pytest.skip("readelf missing")
    out = subprocess.check_output(["readelf", "-d", str(eng)], text=True)
    assert "libeonclib" not in out
    assert "eonclib" not in out
    for line in _rpath_lines(eng):
        # No absolute host paths — only $ORIGIN peers
        assert "/home/" not in line, line
        assert "/Users/" not in line, line
        assert "/bbdir" not in line, line
        assert ".mesonpy-" not in line, line
        assert "eOn-pyeon" not in line, line
        assert "$ORIGIN" in line or "Library runpath" in line


def test_dlopen_force_evaluation():
    eng = _require_engine()
    model = _model_path()
    if model is None:
        pytest.skip(
            "metatomic model file is absent; set RGPOT_METATOMIC_MODEL to a .pt path"
        )

    os.environ.pop("LD_LIBRARY_PATH", None)
    os.environ.pop("LIBRARY_PATH", None)
    os.environ["RGPOT_METATOMIC_ENGINE"] = str(eng)

    positions = np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], dtype=np.float64)
    atom_types = np.array([1, 1], dtype=np.int32)
    box = np.eye(3, dtype=np.float64) * 20.0

    energy, forces, variance = rgpot.evaluate_metatomic(
        positions,
        atom_types,
        box,
        model_path=str(model),
        engine_path=str(eng),
        device="cpu",
    )
    assert math.isfinite(float(energy)), energy
    assert forces.shape == (2, 3), forces.shape
    assert np.all(np.isfinite(forces)), forces
    print("RGPOT_DLOPEN_FORCE_OK", float(energy), forces.shape)


def test_available_engine_abis_and_torch_match():
    """Picker prefers lib/torch-X.Y matching installed torch when present."""
    abis = rgpot.available_metatomic_engine_abis()
    found = _engine_path()
    if found is None:
        pytest.skip(
            "libmetatomic_engine.so is absent (optional metatomic engine library)"
        )
    eng = str(found)
    # If multi-ABI pack is present, path should mention torch-X.Y
    if abis:
        assert "torch-" in eng or Path(eng).name == "libmetatomic_engine.so"
        try:
            import torch
            maj = ".".join(torch.__version__.split("+")[0].split(".")[:2])
            if maj in abis:
                assert f"torch-{maj}" in eng, (maj, eng, abis)
        except ImportError:
            pass
