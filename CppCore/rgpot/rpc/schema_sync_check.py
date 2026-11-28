#!/usr/bin/env python3
"""Compare a vendored schema file with the pinned one, ignoring CR/LF differences."""
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: schema_sync_check.py <vendored> <canonical>", file=sys.stderr)
        return 2
    a = pathlib.Path(sys.argv[1]).read_bytes().replace(b"\r\n", b"\n")
    b = pathlib.Path(sys.argv[2]).read_bytes().replace(b"\r\n", b"\n")
    if a == b:
        return 0
    print(
        "vendored schema file diverges from the pinned rgconf schema",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
