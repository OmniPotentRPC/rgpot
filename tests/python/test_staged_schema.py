# Copyright 2023--present rgpot developers
"""The installed package loads Potentials.capnp without the source tree."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("capnp")


def test_schema_loads_from_staged_package(tmp_path: Path) -> None:
    import rgpot

    package = Path(rgpot.__file__).resolve().parent
    site = package.parent
    stage = tmp_path / "site"
    shutil.copytree(
        package,
        stage / package.name,
        symlinks=False,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    libs = site / ".rgpot.mesonpy.libs"
    if libs.is_dir():
        shutil.copytree(libs, stage / libs.name, symlinks=False)
    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        stage = Path(sys.argv[1]).resolve()
        import rgpot
        from rgpot._schema import load_potentials_schema, schema_path

        loaded = Path(rgpot.__file__).resolve()
        assert stage in loaded.parents, loaded
        path = schema_path().resolve()
        assert path.is_file(), path
        assert path.name == "Potentials.capnp", path
        assert loaded.parent in path.parents, path
        common = path.parent / "common.capnp"
        assert common.is_file(), common
        schema = load_potentials_schema()
        assert schema.Potential is not None
        print(path)
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(stage)
    proc = subprocess.run(
        [sys.executable, "-c", script, str(stage)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    staged = Path(proc.stdout.strip())
    assert stage in staged.parents
