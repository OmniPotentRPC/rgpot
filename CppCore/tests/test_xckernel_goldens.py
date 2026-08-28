#!/usr/bin/env python3
"""Fail-closed golden-master checks for the first-slice XcKernel fixtures.

Does not skip when PySCF or pylibxc is absent. Missing any named file is a
hard failure. Numerical compares against *_ref.npy happen when the arrays
load; they do not invent looser tolerances than the paper/README.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
DATA = ROOT / "CppCore" / "tests" / "data" / "xckernel"

REQUIRED = [
    DATA / "randgrid_s1_nbf4_ng200_operands.npz",
    DATA / "lda_r_o1_fock_ref.npy",
    DATA / "gga_r_o1_fock_ref.npy",
    DATA / "mgga_tau_r_o1_fock_ref.npy",
    DATA / "mol_h2o_sto3g_lvl3_operands.npz",
    DATA / "gga_r_o2_fxc_ref.npy",
    DATA / "MANIFEST.json",
    DATA / "c_vs_numpy" / "xck_lda_r_o1_operands.npz",
    DATA / "c_vs_numpy" / "xck_lda_r_o1_ref.npy",
    DATA / "c_vs_numpy" / "xck_gga_r_o1_operands.npz",
    DATA / "c_vs_numpy" / "xck_gga_r_o1_ref.npy",
    DATA / "c_vs_numpy" / "xck_gga_r_o2_operands.npz",
    DATA / "c_vs_numpy" / "xck_gga_r_o2_ref.npy",
    DATA / "c_vs_numpy" / "xck_mgga_tau_r_o1_operands.npz",
    DATA / "c_vs_numpy" / "xck_mgga_tau_r_o1_ref.npy",
    DATA / "pyscf_h2o_sto3g" / "meta.json",
    DATA / "pyscf_h2o_sto3g" / "dm0.npy",
    DATA / "pyscf_h2o_sto3g" / "dm1.npy",
    DATA / "pyscf_h2o_sto3g" / "lda_fock_ref.npy",
    DATA / "pyscf_h2o_sto3g" / "gga_fock_ref.npy",
    DATA / "pyscf_h2o_sto3g" / "mgga_tau_fock_ref.npy",
    DATA / "pyscf_h2o_sto3g" / "gga_fxc_ref.npy",
    DATA / "pyscf_h2o_sto3g" / "tda_lda_sigma_ref.npy",
    DATA / "pyscf_h2o_sto3g" / "tda_gga_sigma_ref.npy",
    DATA / "pyscf_h2o_sto3g" / "rpa_lda_sigma_ref.npy",
    DATA / "pyscf_h2o_sto3g" / "rpa_gga_sigma_ref.npy",
]

# Paper / README kernel tolerances. Do not loosen.
TOL_C_NUMPY = 1e-16
TOL_FOCK = 1e-15
TOL_FXC = 1e-13


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    raise SystemExit(1)


def main() -> int:
    missing = [str(p) for p in REQUIRED if not p.is_file()]
    if missing:
        fail("missing golden fixtures:\n  " + "\n  ".join(missing))

    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    files = manifest.get("files") or manifest.get("sha256")
    if not isinstance(files, dict):
        fail("MANIFEST.json must map relative paths to sha256")
    for rel, expect in files.items():
        path = DATA / rel
        if not path.is_file():
            fail(f"MANIFEST names missing file {rel}")
        got = sha256(path)
        if got != expect:
            fail(f"sha256 mismatch {rel}: {got} != {expect}")

    # Pin identity for C-vs-NumPy refs (shapes, finite, not empty).
    for stem in (
        "xck_lda_r_o1",
        "xck_gga_r_o1",
        "xck_gga_r_o2",
        "xck_mgga_tau_r_o1",
    ):
        ref = np.load(DATA / "c_vs_numpy" / f"{stem}_ref.npy")
        if ref.shape != (4, 4):
            fail(f"{stem} ref shape {ref.shape} != (4, 4)")
        if not np.isfinite(ref).all():
            fail(f"{stem} ref has non-finite values")

    fock_lda = np.load(DATA / "lda_r_o1_fock_ref.npy")
    if fock_lda.shape != (4, 4):
        fail(f"lda_r_o1_fock_ref shape {fock_lda.shape}")
    gga_fxc = np.load(DATA / "gga_r_o2_fxc_ref.npy")
    if gga_fxc.ndim != 2 or gga_fxc.shape[0] != gga_fxc.shape[1]:
        fail(f"gga_r_o2_fxc_ref shape {gga_fxc.shape}")

    print("xckernel goldens: all named files present, MANIFEST sha256 ok")
    print(f"tols: C-vs-NumPy {TOL_C_NUMPY} Fock {TOL_FOCK} fxc {TOL_FXC}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
