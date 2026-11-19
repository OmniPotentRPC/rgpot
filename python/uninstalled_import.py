# MIT License
# Copyright 2023--present rgpot developers
"""Import rgpot from a build tree stage with an empty library path."""

from __future__ import annotations

import os
import pathlib
import sys


def main() -> None:
    core_dir = pathlib.Path(os.environ["RGPOT_CORE_DIR"])
    src_pkg = pathlib.Path(os.environ["RGPOT_SRC_PKG"])
    stage = pathlib.Path(os.environ["RGPOT_STAGE"])
    pkg = stage / "rgpot"
    pkg.mkdir(parents=True, exist_ok=True)
    for item in src_pkg.iterdir():
        if item.name == "__pycache__" or item.name.startswith("_core."):
            continue
        if item.is_dir():
            continue
        dest = pkg / item.name
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        dest.symlink_to(item.resolve())
    cores = sorted(core_dir.glob("_core*.so")) + sorted(core_dir.glob("_core*.pyd"))
    if not cores:
        raise SystemExit(f"no extension in {core_dir}")
    core = cores[0]
    dest = pkg / core.name
    if dest.is_symlink() or dest.exists():
        dest.unlink()
    dest.symlink_to(core.resolve())
    os.environ.pop("LD_LIBRARY_PATH", None)
    os.environ.pop("DYLD_LIBRARY_PATH", None)
    sys.path.insert(0, str(stage))
    import rgpot

    if not callable(rgpot.evaluate_lj):
        raise SystemExit("evaluate_lj is missing")
    print(rgpot.__file__)


if __name__ == "__main__":
    main()
