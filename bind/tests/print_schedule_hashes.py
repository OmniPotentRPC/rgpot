# MIT License
# Copyright 2023--present rgpot developers
"""Print schedule hashes at 1, 2, 3, and 8 workers."""

from __future__ import annotations

import hashlib

import _rgbind_test

FIXTURE = [
    (0, 10, False),
    (2, 10, False),
    (1, 20, True),
    (4, 10, False),
    (3, 30, False),
    (5, 20, True),
]


def main() -> None:
    first = ""
    for workers in (1, 2, 3, 8):
        text = _rgbind_test.schedule_canonical(FIXTURE, workers)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        print(f"schedule_sha256 workers={workers} {digest}")
        if not first:
            first = digest
        elif digest != first:
            raise SystemExit(f"hash changed at {workers} workers")
    print("identical=1")


if __name__ == "__main__":
    main()
