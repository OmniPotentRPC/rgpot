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

On Darwin the cargo subprocess uses ``/usr/bin/ar`` and
``/usr/bin/ranlib``. Pixi exports ``AR`` as the conda archiver
(``arm64-apple-darwin20.0.0-ar``), rustc honours that variable when it
writes a staticlib, and Apple ld rejects the result
(``malformed archive, member exceeds file size``). The rest of the
meson build keeps the pixi archiver.

A static meson library then partial-links that archive to one
relocatable ``.o``. Apple ar places link arguments before the output
archive, and a custom-target ``.a`` lands in those arguments, so ar
tries to write the rust archive and looks for ``librgpot.a`` as a
member.
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


def cargo_env() -> dict[str, str] | None:
    """Environment for the cargo subprocess.

    ``None`` inherits the parent environment. Darwin overrides only the
    archiver rustc uses for this staticlib.
    """
    if sys.platform != "darwin":
        return None
    ar = "/usr/bin/ar"
    ranlib = "/usr/bin/ranlib"
    if not (os.path.isfile(ar) and os.access(ar, os.X_OK)):
        return None
    if not (os.path.isfile(ranlib) and os.access(ranlib, os.X_OK)):
        return None
    env = os.environ.copy()
    env["AR"] = ar
    env["RANLIB"] = ranlib
    return env


def print_native_libs(cargo: str, manifest: str, target_dir: str) -> list[str]:
    proc = subprocess.run(
        cargo_cmd(cargo, "rustc", manifest, target_dir)
        + ["--", "--print", "native-static-libs"],
        text=True,
        capture_output=True,
        env=cargo_env(),
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


def relocatable_object(archive: str, obj_out: str) -> None:
    """Partial-link every member of ``archive`` into one relocatable object."""
    if sys.platform == "darwin":
        cmd = [
            "/usr/bin/clang",
            "-r",
            "-nostdlib",
            "-Wl,-force_load," + os.path.abspath(archive),
            "-o",
            obj_out,
        ]
    elif sys.platform == "win32":
        sys.exit(
            "the fast pair list relocatable object is only built for a "
            "static Unix library"
        )
    else:
        cmd = [
            "ld",
            "-r",
            "-o",
            obj_out,
            "--whole-archive",
            archive,
            "--no-whole-archive",
        ]
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        sys.exit(proc.returncode)


def copy_archive(cargo: str, manifest: str, target_dir: str, lib_out: str) -> None:
    proc = subprocess.run(
        cargo_cmd(cargo, "build", manifest, target_dir),
        text=True,
        env=cargo_env(),
    )
    if proc.returncode != 0:
        sys.exit(proc.returncode)
    # The meson output is a relocatable object when the umbrella is a
    # static library. Cargo still writes the archive under target_dir.
    cargo_name = "rgpot_fastnlist.lib" if sys.platform == "win32" else "librgpot_fastnlist.a"
    if lib_out.lower().endswith(".lib"):
        cargo_name = "rgpot_fastnlist.lib"
    built = find_artifact(target_dir, cargo_name)
    if lib_out.endswith(".o"):
        relocatable_object(built, lib_out)
        return
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
