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


def _declared_schema_id(path: Path) -> int | None:
    """Return the file id, the ``@0x...;`` declaration at the top of a schema."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("@0x") and stripped.endswith(";"):
            try:
                return int(stripped[3:-1], 16)
            except ValueError:
                return None
    return None


def _schema_already_loaded(schema_id: int) -> Any:
    """Return a schema module this process already loaded for ``schema_id``.

    Loading the same id from a second path aborts the interpreter. The
    installed copy and the source copy share that id, so a caller that
    loaded either one first has to be reused.
    """
    import capnp.lib.capnp as capnp_impl  # pyright: ignore[reportMissingImports]

    parser = getattr(capnp_impl, "_global_schema_parser", None)
    modules = getattr(parser, "modules_by_id", None)
    if modules is None:
        return None
    loaded = modules.get(schema_id)
    if loaded is None or getattr(loaded, "Potential", None) is None:
        return None
    return loaded


def load_potentials_schema() -> Any:
    """Load ``Potentials.capnp`` once per process and return the schema."""
    global _SCHEMA
    with _SCHEMA_LOCK:
        if _SCHEMA is None:
            import capnp  # pyright: ignore[reportMissingImports]

            path = schema_path()
            schema_id = _declared_schema_id(path)
            loaded = None if schema_id is None else _schema_already_loaded(schema_id)
            _SCHEMA = loaded if loaded is not None else capnp.load(str(path))
        return _SCHEMA
