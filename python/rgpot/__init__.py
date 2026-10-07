"""rgpot — potential energy surfaces for atomistic simulation.

Core: Lennard-Jones, ExprPot, D3Pot, D4Pot, and XcKernel (when the
wheel/meson build compiled them). Metatomic path uses **dlopen** of a
portable ``libmetatomic_engine.so`` (stable C ABI). Engines are multi-ABI:

  ``rgpot/lib/torch-X.Y/libmetatomic_engine.so``

selected from the installed ``torch`` major (same layout as metatomic-torch).

Supported torch majors for bundled engines: **2.7 and newer**. Older torch
builds may still import LJ, but Metatomic dlopen needs a matching engine.
"""

from __future__ import annotations

import os
from pathlib import Path

from rgpot._core import (
    LJPot,
    __version__,
    evaluate_lj,
    evaluate_metatomic_dlopen,
    has_cache,
    has_dftd3,
    has_dftd4,
    has_expr,
    has_metatomic_dlopen,
    has_xckernel,
)

try:
    from rgpot._core import PotentialCache
except ImportError:  # compiled without -Dwith_cache
    PotentialCache = None  # type: ignore[misc, assignment]

try:
    from rgpot._core import D3Pot
except ImportError:  # compiled without -Dwith_dftd3
    D3Pot = None  # type: ignore[misc, assignment]

try:
    from rgpot._core import D4Pot
except ImportError:
    D4Pot = None  # type: ignore[misc, assignment]

try:
    from rgpot._core import ExprPot
except ImportError:
    ExprPot = None  # type: ignore[misc, assignment]

try:
    from rgpot._core import XcKernel
except ImportError:
    XcKernel = None  # type: ignore[misc, assignment]


def _torch_major() -> str | None:
    """Return installed torch X.Y, or None if torch is not importable."""
    try:
        import torch

        v = torch.__version__.split("+", 1)[0]
        parts = v.split(".")
        return f"{parts[0]}.{parts[1]}"
    except Exception:
        return None


def default_metatomic_engine_path() -> str | None:
    """Return package-bundled engine matching installed torch ABI if possible."""
    here = Path(__file__).resolve().parent
    maj = _torch_major()
    candidates: list[Path] = []
    # Multi-ABI layout first (rgpot/lib/torch-X.Y/) — preferred product path
    if maj:
        candidates.append(here / "lib" / f"torch-{maj}" / "librgpot_metatomic_engine.so")
        candidates.append(here / "lib" / f"torch-{maj}" / "libmetatomic_engine.so")
    lib_root = here / "lib"
    if lib_root.is_dir():
        for d in sorted(lib_root.glob("torch-*")):
            candidates.append(d / "librgpot_metatomic_engine.so")
            candidates.append(d / "libmetatomic_engine.so")
    # Legacy single-engine layouts (last resort). The librgpot_ name is the
    # engine. libmetatomic_engine.so remains for one release.
    candidates.extend(
        [
            here / "lib" / "librgpot_metatomic_engine.so",
            here / "lib" / "libmetatomic_engine.so",
            here.parent / ".rgpot.mesonpy.libs" / "librgpot_metatomic_engine.so",
            here.parent / ".rgpot.mesonpy.libs" / "libmetatomic_engine.so",
            here / "librgpot_metatomic_engine.so",
            here / "libmetatomic_engine.so",
        ]
    )

    for c in candidates:
        if c.is_file():
            return str(c)

    env = os.environ.get("RGPOT_METATOMIC_ENGINE") or os.environ.get(
        "METATOMIC_ENGINE"
    )
    if env and Path(env).is_file():
        return env
    return None


def available_metatomic_engine_abis() -> list[str]:
    """List torch-X.Y majors for which a bundled engine is present.

    PyPI wheels ship engines for torch 2.7+ (see package README).
    """
    here = Path(__file__).resolve().parent / "lib"
    out: list[str] = []
    if not here.is_dir():
        return out
    for d in sorted(here.glob("torch-*")):
        if (d / "librgpot_metatomic_engine.so").is_file() or (
            d / "libmetatomic_engine.so"
        ).is_file():
            out.append(d.name.removeprefix("torch-"))
    return out


def evaluate_metatomic(
    positions,
    atom_types,
    box,
    *,
    model_path: str,
    engine_path: str | None = None,
    device: str = "cpu",
):
    """Force evaluation through MetatomicDlopen (real engine plugin)."""
    # Load torch via the Python extension first so the engine's DT_NEEDED
    # libtorch resolves to the same process image (avoids dual-load races).
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    eng = engine_path or default_metatomic_engine_path() or ""
    return evaluate_metatomic_dlopen(
        positions, atom_types, box, model_path, eng, device
    )


__all__ = [
    "LJPot",
    "D3Pot",
    "D4Pot",
    "ExprPot",
    "XcKernel",
    "PotentialCache",
    "evaluate_lj",
    "evaluate_metatomic",
    "evaluate_metatomic_dlopen",
    "default_metatomic_engine_path",
    "available_metatomic_engine_abis",
    "has_metatomic_dlopen",
    "has_cache",
    "has_expr",
    "has_dftd3",
    "has_dftd4",
    "has_xckernel",
    "__version__",
]
