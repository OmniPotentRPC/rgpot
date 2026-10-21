#!/usr/bin/env python3
"""Build the linkcell/minimage staticlib and copy it to the meson output.

The AVX-512 kernels in linkcell and minimage are behind runtime
dispatch, and they compile only on rustc 1.89 or newer (the release
that stabilised the avx512 target features). A default rustc older
than that is replaced with ``rustup run 1.89.0`` when that toolchain
is installed.
"""

import shutil
import subprocess
import sys


def rustc_minor(argv: list[str]) -> int:
    try:
        out = subprocess.check_output(argv + ["-vV"], text=True)
    except (OSError, subprocess.CalledProcessError):
        return 0
    for line in out.splitlines():
        if line.startswith("release:"):
            ver = line.split()[1].split("-")[0]
            parts = ver.split(".")
            if len(parts) >= 2 and parts[0] == "1":
                return int(parts[1])
    return 0


def cargo_argv(default_cargo: str) -> list[str]:
    if rustc_minor(["rustc"]) >= 89:
        return [default_cargo]
    if rustc_minor(["rustup", "run", "1.89.0", "rustc"]) >= 89:
        return ["rustup", "run", "1.89.0", "cargo"]
    sys.exit(
        "rgpot_fastnlist needs rustc >= 1.89 so the AVX-512 linkcell and "
        "minimage kernels are built. The rustc on PATH is older, and "
        "rustup has no 1.89.0 toolchain."
    )


def main() -> None:
    cargo, manifest, target_dir, output = sys.argv[1:5]
    subprocess.check_call(
        cargo_argv(cargo)
        + [
            "build",
            "--release",
            "--manifest-path",
            manifest,
            "--target-dir",
            target_dir,
            "--locked",
        ]
    )
    built = target_dir + "/release/librgpot_fastnlist.a"
    shutil.copyfile(built, output)


if __name__ == "__main__":
    main()
