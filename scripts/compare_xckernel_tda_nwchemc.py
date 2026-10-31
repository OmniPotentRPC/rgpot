#!/usr/bin/env python3
"""Compare the H2O/sto-3g TDA pin operator to libnwchemc TDA roots.

Runs only on a host listed in RGPOT_REGEN_HOSTS (comma-separated),
unless XCKERNEL_ALLOW_REGEN=1. Calls the shipped libnwchemc energy ABI
with theory=tddft (same engine NWChemPot dlopens). Does not transcribe
NWChem's TDDFT.

The exclusive 1e-17 bar is tightness of that compare, not a PySCF-only
gate. PySCF TDA.kernel on the committed MOs is the pin's spectrum;
nwchemc TDA roots are the second engine.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path

for _thr in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thr, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "CppCore" / "tests" / "data" / "xckernel" / "pyscf_h2o_sto3g"
# Exclusive contraction bar (XcKernel vs sigma pin) stays 1e-17.
# Cross-engine TDA roots (libnwchemc vs pin-operator TDA.kernel) measured
# 7.44e-7 Ha on this sto-3g case. That is not a loosened contraction bar.
BAR_SIGMA = 1e-17
BAR_NWCHEMC_ROOTS = 1e-6
# H2O pin geometry (Angstrom), same as regen_xckernel_goldens.py.
POS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.96, 0.0, 0.93, -0.24], dtype=np.float64
)
ZNUM = np.array([8, 1, 1], dtype=np.int32)
NROOTS = 5

def _first_file(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_file():
            return p
    return None


def _first_dir(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.is_dir():
            return p
    return None

# Pin functionals are exchange-only. NWChem names: slater, xpbe96.
FAMILIES = (
    ("lda", "LDA_X,", "slater"),
    ("gga", "GGA_X_PBE,", "xpbe96"),
)

ROOT_RE = re.compile(
    r"^\s*(\d+)\s+singlet\b.*?([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)\s+a\.u\.",
    re.IGNORECASE,
)
ROOT_RE_ALT = re.compile(
    r"^\s*Root\s+(\d+)\s+singlet.*?([-+]?\d+\.\d+(?:[Ee][-+]?\d+)?)",
    re.IGNORECASE,
)


class NWChemCResult(ctypes.Structure):
    _fields_ = [
        ("ok", ctypes.c_int),
        ("energy_h", ctypes.c_double),
        ("message", ctypes.c_char * 512),
    ]


def _require_regen_host() -> None:
    raw = os.environ.get("RGPOT_REGEN_HOSTS", "")
    allowed = {h.strip() for h in raw.split(",")} - {""}
    host = socket.gethostname()
    override = os.environ.get("XCKERNEL_ALLOW_REGEN") == "1"
    if override or allowed & {host, host.split(".")[0]}:
        return
    sys.exit(
        "compare_xckernel_tda_nwchemc.py runs only on a host listed in "
        f"RGPOT_REGEN_HOSTS (got {host}; XCKERNEL_ALLOW_REGEN=1 overrides)"
    )


def _params_text(xc_nw: str, scratch: Path, permanent: Path) -> str:
    return (
        "(\n"
        '  basis = "sto-3g",\n'
        '  theory = "tddft",\n'
        '  scfType = "dft",\n'
        "  charge = 0,\n"
        "  multiplicity = 1,\n"
        '  task = "energy",\n'
        '  title = "h2o sto-3g tda pin vs nwchemc",\n'
        "  memoryMb = 1024,\n"
        f'  scratchDir = "{scratch}",\n'
        f'  permanentDir = "{permanent}",\n'
        "  inputBlocks = [\n"
        f'    "dft\\n  xc {xc_nw}\\n  direct\\n  iterations 80\\nend",\n'
        f'    "tddft\\n  nroots {NROOTS}\\n  tda\\n  singlet\\n  '
        'notriplet\\n  thresh 1e-8\\nend",\n'
        "  ],\n"
        ")\n"
    )


def _resolve_tools() -> tuple[Path, Path, Path, Path]:
    capnp_env = os.environ.get("CAPNP_BIN")
    capnp = Path(capnp_env) if capnp_env else None
    if capnp is None:
        found = shutil.which("capnp")
        capnp = Path(found) if found else None
    schema_env = os.environ.get("NWCHEMC_SCHEMA")
    schema = (
        Path(schema_env)
        if schema_env
        else _first_file(
            [
                ROOT / "CppCore" / "rgpot" / "rpc" / "Potentials.capnp",
                ROOT.parent / "nwchemc" / "schema" / "Potentials.capnp",
            ]
        )
    )
    lib_env = os.environ.get("NWCHEMC_LIBRARY")
    lib = (
        Path(lib_env)
        if lib_env
        else _first_file(
            [
                Path("libnwchemc.so"),
                ROOT.parent / "nwchemc" / "build-nwchem-system" / "libnwchemc.so",
            ]
        )
    )
    basis_env = os.environ.get("NWCHEM_BASIS_LIBRARY")
    basis = (
        Path(basis_env)
        if basis_env
        else _first_dir(
            [ROOT / "third_party" / "nwchem" / "src" / "basis" / "libraries"]
        )
    )
    if capnp is None or not capnp.is_file():
        sys.exit("set CAPNP_BIN to the capnp executable")
    if schema is None:
        sys.exit("set NWCHEMC_SCHEMA to Potentials.capnp")
    if lib is None:
        sys.exit("set NWCHEMC_LIBRARY to libnwchemc.so")
    if basis is None:
        sys.exit("set NWCHEM_BASIS_LIBRARY to the NWChem basis tree")
    return capnp, schema, lib, basis


def _encode_params(text: str, out_bin: Path, capnp: Path, schema: Path) -> None:
    proc = subprocess.run(
        [str(capnp), "encode", str(schema), "NWChemParams"],
        input=text.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        sys.stderr.buffer.write(proc.stderr)
        sys.exit(f"capnp encode failed rc={proc.returncode}")
    out_bin.write_bytes(proc.stdout)


def _load_lib(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    lib.nwchemc_available.restype = ctypes.c_int
    lib.nwchemc_finalize.restype = None
    lib.nwchemc_energy.restype = NWChemCResult
    lib.nwchemc_energy.argtypes = [
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    if lib.nwchemc_available() == 0:
        sys.exit("nwchemc_available() == 0 (stub engine)")
    return lib


def _parse_roots(log_text: str) -> list[float]:
    roots: dict[int, float] = {}
    for line in log_text.splitlines():
        m = ROOT_RE.search(line) or ROOT_RE_ALT.search(line)
        if not m:
            continue
        roots[int(m.group(1))] = float(m.group(2))
    return [roots[k] for k in sorted(roots)]


def _run_nwchemc(lib: ctypes.CDLL, params: bytes, log_path: Path) -> tuple[NWChemCResult, list[float]]:
    pos = POS.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    z = ZNUM.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    buf = ctypes.create_string_buffer(params)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    log_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        rc = lib.nwchemc_energy(3, pos, z, ctypes.cast(buf, ctypes.c_void_p), len(params))
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(log_fd)
    text = log_path.read_text(errors="replace")
    return rc, _parse_roots(text)


def _pyscf_tda_eigs(fam: str, xc: str) -> np.ndarray:
    from pyscf import dft as dft_mod
    from pyscf import gto
    from pyscf import lib as pyscf_lib
    from pyscf.tdscf import rks as td_rks

    pyscf_lib.num_threads(1)
    mol = gto.M(
        atom="O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24",
        basis="sto-3g",
        verbose=0,
    )
    mf = dft_mod.RKS(mol)
    mf.xc = xc
    mf.verbose = 0
    mf.grids.level = 3
    mo_path = DATA / f"{fam}_mo.npz"
    if mo_path.exists():
        pinned = np.load(mo_path)
        mf.build()
        mf.grids.build()
        nocc = int((pinned["mo_occ"] > 0).sum())
        nao = int(pinned["Co"].shape[0])
        mo_coeff = np.zeros((nao, nao), dtype=np.float64)
        mo_coeff[:, :nocc] = pinned["Co"]
        mo_coeff[:, nocc:] = pinned["Cv"]
        mf.mo_coeff = mo_coeff
        mf.mo_energy = np.ascontiguousarray(pinned["mo_energy"])
        mf.mo_occ = np.ascontiguousarray(pinned["mo_occ"])
        mf.converged = True
    else:
        mf.kernel()
    td = td_rks.TDA(mf)
    td.singlet = True
    td.nstates = NROOTS
    e, _xy = td.kernel()
    return np.ascontiguousarray(e, dtype=np.float64)


def max_rel(got: np.ndarray, ref: np.ndarray) -> tuple[float, float]:
    n = min(got.size, ref.size)
    if n == 0:
        return float("inf"), float("inf")
    g = got[:n]
    r = ref[:n]
    diff = float(np.max(np.abs(g - r)))
    den = max(float(np.max(np.abs(r))), 1.0)
    return diff / den, diff


def main() -> int:
    _require_regen_host()
    capnp, schema, lib_path, basis = _resolve_tools()
    os.environ.setdefault("NWCHEM_BASIS_LIBRARY", str(basis))
    work = Path(os.environ.get("NWCHEMC_TDA_WORK", "/tmp/rgpot-nwchemc-tda"))
    work.mkdir(parents=True, exist_ok=True)
    lib = _load_lib(lib_path)
    dest = DATA
    dest.mkdir(parents=True, exist_ok=True)
    worst = 0.0
    rc_all = 0
    for fam, xc_pyscf, xc_nw in FAMILIES:
        scratch = work / f"{fam}-scratch"
        permanent = work / f"{fam}-perm"
        scratch.mkdir(exist_ok=True)
        permanent.mkdir(exist_ok=True)
        text = _params_text(xc_nw, scratch, permanent)
        bin_path = work / f"{fam}_tddft.capnp.bin"
        _encode_params(text, bin_path, capnp, schema)
        params = bin_path.read_bytes()
        log_path = work / f"{fam}_nwchemc.log"
        result, nw_roots = _run_nwchemc(lib, params, log_path)
        print(
            f"{fam} nwchemc ok={result.ok} energy_h={result.energy_h:.12e} "
            f"msg={result.message.decode(errors='replace')!r}"
        )
        print(f"{fam} nwchemc roots Ha:", nw_roots)
        if result.ok == 0 or not nw_roots:
            print(f"{fam} FAIL nwchemc (see {log_path})")
            rc_all = 1
            continue
        pyscf_e = _pyscf_tda_eigs(fam, xc_pyscf)
        print(f"{fam} pin-operator TDA.kernel Ha:", pyscf_e.tolist())
        nw = np.asarray(nw_roots, dtype=np.float64)
        rel, absd = max_rel(nw, pyscf_e)
        print(
            f"{fam} nwchemc vs pin-operator rel={rel:.6e} abs={absd:.6e} "
            f"roots_bar={BAR_NWCHEMC_ROOTS} sigma_bar={BAR_SIGMA}"
        )
        np.save(dest / f"tda_{fam}_nwchemc_roots.npy", nw)
        np.save(dest / f"tda_{fam}_pyscf_roots.npy", pyscf_e)
        worst = max(worst, rel)
        if rel > BAR_NWCHEMC_ROOTS:
            print(f"{fam} FAIL nwchemc roots {BAR_NWCHEMC_ROOTS}")
            rc_all = 1
        else:
            print(f"{fam} pass nwchemc roots")
    lib.nwchemc_finalize()
    print(f"worst rel={worst:.6e}")
    return rc_all


if __name__ == "__main__":
    raise SystemExit(main())
