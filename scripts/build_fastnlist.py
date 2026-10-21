#!/usr/bin/env python3
"""Build the linkcell/minimage staticlib and copy it to the meson output.

The AVX-512 kernels in linkcell and minimage are behind runtime
dispatch, and they compile only on rustc 1.89 or newer (the release
that stabilised the avx512 target features). A default rustc older
than that is replaced with ``rustup run 1.89.0`` when that toolchain
is installed.

Cargo names the archive ``librgpot_fastnlist.a`` on Unix and MinGW, and
``rgpot_fastnlist.lib`` on MSVC. ``--native-libs`` prints the system
libraries from ``rustc --print native-static-libs``, one token per line,
so meson can pass them as link arguments. ``--copy`` rebuilds that
archive and copies it to the meson output.
"""

import os
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


def artifact_name(output: str) -> str:
    if output.lower().endswith(".lib"):
        return "rgpot_fastnlist.lib"
    return "librgpot_fastnlist.a"


def find_artifact(target_dir: str, name: str) -> str:
    release = os.path.join(target_dir, "release", name)
    if os.path.isfile(release):
        return release
    if os.path.isdir(target_dir):
        for entry in os.listdir(target_dir):
            nested = os.path.join(target_dir, entry, "release", name)
            if os.path.isfile(nested):
                return nested
    sys.exit(
        f"rgpot_fastnlist did not produce {name} under {target_dir}. "
        "MSVC cargo writes rgpot_fastnlist.lib; Unix and MinGW cargo write "
        "librgpot_fastnlist.a."
    )


def native_libs(text: str) -> list[str]:
    marker = "native-static-libs:"
    found: list[str] = []
    for line in text.splitlines():
        if marker in line:
            found = line.split(marker, 1)[1].split()
    return found


def cargo_cmd(cargo: str, sub: str, manifest: str, target_dir: str) -> list[str]:
    return cargo_argv(cargo) + [
        sub,
        "--release",
        "--manifest-path",
        manifest,
        "--target-dir",
        target_dir,
        "--locked",
    ]


def print_native_libs(cargo: str, manifest: str, target_dir: str) -> list[str]:
    proc = subprocess.run(
        cargo_cmd(cargo, "rustc", manifest, target_dir)
        + ["--", "--print", "native-static-libs"],
        text=True,
        capture_output=True,
    )
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    libs = native_libs(proc.stdout + "\n" + proc.stderr)
    if not libs:
        sys.exit(
            "rustc did not print native-static-libs, so the fast list "
            "cannot be linked against the system libraries it needs."
        )
    return libs


def copy_archive(cargo: str, manifest: str, target_dir: str, lib_out: str) -> None:
    proc = subprocess.run(
        cargo_cmd(cargo, "build", manifest, target_dir),
        text=True,
    )
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    built = find_artifact(target_dir, artifact_name(lib_out))
    shutil.copyfile(built, lib_out)


def main() -> None:
    mode = sys.argv[1]
    if mode == "--native-libs":
        cargo, manifest, target_dir = sys.argv[2:5]
        for lib in print_native_libs(cargo, manifest, target_dir):
            print(lib)
        return
    if mode == "--copy":
        cargo, manifest, target_dir, lib_out = sys.argv[2:6]
        copy_archive(cargo, manifest, target_dir, lib_out)
        return
    sys.exit(
        "usage: build_fastnlist.py --native-libs|--copy cargo manifest "
        "target_dir [lib_out]"
    )


if __name__ == "__main__":
    main()
