#!/usr/bin/env python3
"""Build the linkcell/minimage staticlib and copy it to the meson output."""

import shutil
import subprocess
import sys


def main() -> None:
    cargo, manifest, target_dir, output = sys.argv[1:5]
    subprocess.check_call(
        [
            cargo,
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
