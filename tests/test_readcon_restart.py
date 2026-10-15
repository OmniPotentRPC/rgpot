"""Byte-exact restart golden and the readcon 0.17.1 ASE helpers.

The golden files are the bytes ``ConFrameWriter::new`` wrote at readcon-core
694e808, with canonical left off and precision omitted. This module compares
those bytes. It does not format an expected CON string.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
RPC = ROOT / "CppCore" / "rgpot" / "rpc"
if str(RPC) not in sys.path:
    sys.path.insert(0, str(RPC))

readcon = pytest.importorskip("readcon")
ase = pytest.importorskip("ase")
from ase_helpers import atoms_from_con, atoms_to_con  # noqa: E402

REV = "694e808080c49979ee3099afcfa18e5420c40433"
GOLDEN = {
    "restart_exact.con": (
        469,
        "8326c1be79248dcddc49ec9528452867b6701a6af15fd898a377f423ed4a8c06",
    ),
    "restart_exact_vel.con": (
        653,
        "5d4cfa84a1e008bd715a523440343cffed657e4c2e1152d2fc1858a4dfbf07d3",
    ),
}


def _urls(name: str) -> list[str]:
    raw = "https://raw.githubusercontent.com/lode-org/readcon-core"
    return [
        f"{raw}/{REV}/resources/restart/{name}",
        f"{raw}/{REV}/resources/conformance/valid/{name}",
        f"{raw}/v0.17.1/resources/restart/{name}",
        f"https://github.com/lode-org/readcon-core/releases/download/v0.17.1/{name}",
    ]


def _download(url: str) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": "rgpot-readcon-golden"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            if getattr(response, "status", 200) != 200:
                return None
            return response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        return None
    except (urllib.error.URLError, TimeoutError, OSError):
        return None


def golden_bytes(name: str) -> bytes:
    size, digest = GOLDEN[name]
    vendored = (DATA / name).read_bytes()
    assert len(vendored) == size
    assert hashlib.sha256(vendored).hexdigest() == digest
    downloaded = None
    for url in _urls(name):
        body = _download(url)
        if body:
            downloaded = body
            break
    if downloaded is None:
        return vendored
    assert hashlib.sha256(downloaded).hexdigest() == digest, (
        f"{name} published at readcon-core does not match "
        f"ConFrameWriter::new at {REV}"
    )
    return downloaded


def _bits(array) -> bytes:
    return np.ascontiguousarray(array, dtype=np.float64).tobytes()


def test_readcon_is_0_17_1():
    version = tuple(int(part) for part in readcon.__version__.split(".")[:3])
    assert version >= (0, 17, 1)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_writer_rewrite_is_byte_exact(name, tmp_path):
    expected = golden_bytes(name)
    source = tmp_path / name
    source.write_bytes(expected)
    frame = readcon.read_first_frame(str(source))
    out = tmp_path / "rewrite.con"
    readcon.write_con(str(out), [frame])
    assert out.read_bytes() == expected


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_atoms_roundtrip_is_bitwise(name, tmp_path):
    expected = golden_bytes(name)
    source = tmp_path / name
    source.write_bytes(expected)
    frame = readcon.read_first_frame(str(source))
    xyz = np.asarray(frame.xyz)
    flat = np.ascontiguousarray(xyz, dtype=np.float64).ravel()
    assert any(bool(np.signbit(value) and value == 0.0) for value in flat)
    assert 1 in flat.view(np.uint64)

    atoms = atoms_from_con(source)
    assert atoms.calc is not None
    assert atoms.calc.__class__.__name__ == "SinglePointCalculator"
    assert atoms.get_potential_energy() == frame.energy
    assert _bits(atoms.get_positions()) == _bits(xyz)
    # The calculator keeps the parsed force rows. ASE's get_forces zeros a
    # fixed axis, so that call is not the bit witness.
    assert _bits(atoms.calc.results["forces"]) == _bits(frame.frc)
    if frame.vel is not None:
        assert _bits(atoms.get_velocities()) == _bits(frame.vel)

    sent = object()
    replaced = atoms_from_con(source, calc=sent)
    assert replaced.calc is sent

    written = tmp_path / "atoms.con"
    atoms_to_con(written, atoms)
    again = readcon.read_first_frame(str(written))
    assert _bits(again.xyz) == _bits(xyz)
    assert 1 in np.ascontiguousarray(again.xyz, dtype=np.float64).ravel().view(np.uint64)
    assert 1 in np.ascontiguousarray(again.frc, dtype=np.float64).ravel().view(np.uint64)
    if frame.vel is not None:
        assert again.vel is not None
        assert _bits(again.vel) == _bits(frame.vel)
        assert 1 in np.ascontiguousarray(again.vel, dtype=np.float64).ravel().view(np.uint64)


def test_atoms_to_con_omits_precision_and_leaves_canonical_off():
    source = inspect.getsource(atoms_to_con)
    calls = re.findall(r"write_con\((?:[^()]|\([^)]*\))*\)", source)
    assert calls
    for call in calls:
        assert "precision" not in call
        assert "canonical" not in call
    assert "readcon-db" not in Path(inspect.getfile(atoms_to_con)).read_text()


def test_pyproject_does_not_depend_on_readcon():
    assert "readcon" not in (ROOT / "pyproject.toml").read_text()


def test_cpmd_restart_keyword_stays_a_deck_flag():
    text = (ROOT / "tests" / "cpmd_params.py").read_text()
    assert "cpmd.restart = bool(" in text


def test_eonviz_pin_script_parses_the_readcon_floor():
    script = ROOT / "scripts" / "check_eonviz_pins.sh"
    ok = subprocess.run(["bash", str(script)], cwd=ROOT, capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert "eonviz pin check ok" in ok.stdout


def test_eonviz_pin_script_rejects_a_lower_readcon_floor(tmp_path):
    bad = tmp_path / "pixi.toml"
    bad.write_text(
        "\n".join(
            [
                "[feature.eonviz.pypi-dependencies]",
                'chemparseplot = "==1.11.0"',
                'rgpycrumbs = "==1.11.0"',
                'readcon = ">=0.14.7"',
            ]
        )
        + "\n"
    )
    env = os.environ.copy()
    env["EONVIZ_TOML"] = str(bad)
    proc = subprocess.run(
        ["bash", str(ROOT / "scripts" / "check_eonviz_pins.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "0.17.1" in proc.stderr


def test_landscape_script_is_not_collected_without_eonclient():
    """pytest only collects test_*.py, so the landscape script stays out of
    the job when eonclient is absent, the same skip as before this pin.
    """
    assert (ROOT / "pyproject.toml").read_text().count('python_files = ["test_*.py"]') == 1
    name = (ROOT / "tests" / "eon_min_landscape.py").name
    assert not name.startswith("test_")
