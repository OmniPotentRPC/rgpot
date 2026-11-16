"""rgpot imports through a symlinked ``_core`` with no library path set.

A build tree or a staged package links the extension into place. The
runpath that finds librgpot is relative to the extension, so the package
must open it from its resolved path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import rgpot

pytestmark = [
    pytest.mark.thread_unsafe,
    pytest.mark.skipif(
        sys.platform == "win32",
        reason="creating symlinks needs a privilege on Windows",
    ),
]

_PACKAGE = Path(rgpot.__file__).resolve().parent
_CORE = Path(rgpot._core.__file__).resolve()
_LIBRARY_PATHS = (
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "DYLD_FALLBACK_LIBRARY_PATH",
)


def test_symlinked_core_imports_without_a_library_path(tmp_path: Path) -> None:
    pkg = tmp_path / "rgpot"
    pkg.mkdir()
    for item in _PACKAGE.iterdir():
        if item.name == "__pycache__" or item.name.startswith("_core."):
            continue
        (pkg / item.name).symlink_to(item)
    (pkg / _CORE.name).symlink_to(_CORE)

    env = {k: v for k, v in os.environ.items() if k not in _LIBRARY_PATHS}
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "import rgpot\n"
        "print(rgpot.__file__)\n"
        "print(rgpot._core.__file__)\n"
        "print(rgpot.LJPot.__name__)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    package, core, name = out.stdout.split()
    assert Path(package).parent == pkg
    assert Path(core) == _CORE
    assert name == "LJPot"
