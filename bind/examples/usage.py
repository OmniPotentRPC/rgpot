# MIT License
# Copyright 2023--present rgpot developers
"""Doc example: the schedule hash does not depend on the worker count.

F8 says the worker count is not configuration. The same rows produce
one canonical schedule at 1, 2, 3, and 8 workers.
"""

from __future__ import annotations

import hashlib

import _rgbind_test

Rows = list[tuple[int, int, bool]]


def commit_order(rows: Rows, workers: int) -> str:
    """Return the canonical per-owner schedule."""
    return _rgbind_test.schedule_canonical(rows, workers)


def schedule_sha256(rows: Rows, workers: int) -> str:
    """SHA-256 of the canonical schedule."""
    text = commit_order(rows, workers)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
