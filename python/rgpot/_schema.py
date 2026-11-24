# Copyright 2023--present rgpot developers
"""Shared process-wide loader for the Potentials Cap'n Proto schema."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

_SCHEMA_LOCK = threading.Lock()
_SCHEMA: Any = None


def schema_path() -> Path:
    """Return the installed or source-tree ``Potentials.capnp`` path."""
    here = Path(__file__).resolve().parent / "ase" / "Potentials.capnp"
    if here.is_file():
        return here
    repo = (
        Path(__file__).resolve().parents[2]
        / "CppCore"
        / "rgpot"
        / "rpc"
        / "Potentials.capnp"
    )
    if repo.is_file():
        return repo
    raise FileNotFoundError("Potentials.capnp is not installed with rgpot.ase")


def load_potentials_schema() -> Any:
    """Load ``Potentials.capnp`` once per process and return the schema."""
    global _SCHEMA
    with _SCHEMA_LOCK:
        if _SCHEMA is None:
            import capnp  # pyright: ignore[reportMissingImports]

            _SCHEMA = capnp.load(str(schema_path()))
        return _SCHEMA
