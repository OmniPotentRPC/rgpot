#!/usr/bin/env python3
"""Build one rgpot wheel into the asv cache directory.

meson-python reads ``tool.meson-python.args`` and then appends
``setup-args`` from the caller. Meson keeps the last ``-D`` for each
option. Dispersion corrections and the metatomic engine stay off.
Fortran pots stay on, because ``WaterHPot`` is one of those kernels.
``-Dbuildtype=release`` selects the optimized build. Meson's own
default buildtype is debug.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SETUP_ARGS = (
    "-Dwith_python=true",
    "-Dwith_rpc=false",
    "-Dwith_rpc_client_only=false",
    "-Dwith_tests=false",
    "-Dwith_examples=false",
    "-Dwith_fortran_pots=enabled",
    "-Dwith_metatomic=false",
    "-Dwith_xtb=false",
    "-Dwith_xckernel=false",
    "-Dwith_dftd3=false",
    "-Dwith_dftd4=false",
    "-Dwith_expr=false",
    "-Dwith_tblite=false",
    "-Dwith_rust_core=false",
    "-Dwith_cache=false",
    "-Dwith_eigen=false",
    "-Dbuildtype=release",
)


REPAIR_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rgpot_repair_wheel.sh"


def require_elf_tools() -> None:
    if shutil.which("patchelf") is None:
        raise SystemExit("patchelf is required to rewrite the extension RPATH")
    if shutil.which("readelf") is None:
        raise SystemExit("readelf is required to read the extension SONAME")


def repair_wheel(cache: str) -> None:
    """meson-python drops the DT_SONAME file. _core NEEDs that name.

    ``--refresh-soname`` copies the versioned library onto that name and
    leaves every RUNPATH as meson-python wrote it.
    """
    wheels = sorted(Path(cache).glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected one wheel in {cache}, found {[p.name for p in wheels]}")
    subprocess.check_call(["bash", str(REPAIR_SCRIPT), "--refresh-soname", str(wheels[0])])


def record_buildoptions(cache: Path, build: Path) -> None:
    """Write ``meson introspect --buildoptions`` next to the wheel."""
    meson = shutil.which("meson")
    if meson is None:
        raise SystemExit("meson was not found on PATH")
    text = subprocess.check_output(
        [meson, "introspect", "--buildoptions", str(build)],
        text=True,
    )
    (cache / "buildoptions.json").write_text(text, encoding="utf-8")
    values = {item["name"]: item["value"] for item in json.loads(text)}
    buildtype = values.get("buildtype")
    if buildtype != "release":
        raise SystemExit(f"meson buildtype is {buildtype}")
    print(text, end="")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: build_wheel.py BUILD_CACHE_DIR BUILD_DIR")
    cache, source = sys.argv[1], sys.argv[2]
    # Meson probes the `c++` driver. Pin gcc so that probe links
    # libstdc++ the same way g++ does.
    os.environ["CC"] = "gcc"
    os.environ["CXX"] = "g++"
    require_elf_tools()
    # Exact versions are installed by the asv build_command before this script.
    py = sys.executable
    build = Path(cache) / "meson"
    command = [
        py,
        "-m",
        "pip",
        "wheel",
        "--no-build-isolation",
        "--no-deps",
        "-w",
        cache,
        source,
    ]
    for arg in SETUP_ARGS:
        command.extend(["--config-settings", f"setup-args={arg}"])
    command.extend(["--config-settings", "compile-args=-j4"])
    command.extend(["--config-settings", f"builddir={build}"])
    subprocess.check_call(command)
    record_buildoptions(Path(cache), build)
    repair_wheel(cache)


if __name__ == "__main__":
    main()
