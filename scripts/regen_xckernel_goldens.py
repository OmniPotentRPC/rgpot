#!/usr/bin/env python3
"""Regenerate committed libxckernel golden masters. rg.terra only.

Do not invoke from meson test. Do not invent looser tolerances.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

import numpy as np

PIN = "d6a9d57ba3fe0f2763667ce168d0c0ef21cff4a4"
TERRA_HOSTS = {"rgam5terra", "rg.terra"}
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "CppCore" / "tests" / "data" / "xckernel"
H6 = ("xx", "xy", "xz", "yy", "yz", "zz")


def _require_terra() -> None:
    host = socket.gethostname().split(".")[0]
    if host not in TERRA_HOSTS and os.environ.get("XCKERNEL_ALLOW_REGEN") != "1":
        sys.exit(f"regen_xckernel_goldens.py runs on rg.terra only (got {host})")


def _ensure_xckernel() -> None:
    src = os.environ.get("LIBXCKERNEL_SRC")
    candidates = []
    if src:
        candidates.append(Path(src))
    candidates.extend(
        [
            Path("/tmp/libxckernel-src"),
            ROOT / "subprojects" / "libxckernel-d6a9d57",
        ]
    )
    for cand in candidates:
        if (cand / "xckernel").is_dir():
            sys.path.insert(0, str(cand))
            return
    sys.exit("LIBXCKERNEL_SRC /tmp/libxckernel-src missing")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """Classic uncompressed ZIP (no ZIP64) so the C++ golden reader stays simple."""
    import io
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        path, "w", compression=zipfile.ZIP_STORED, allowZip64=False
    ) as zf:
        for key, val in arrays.items():
            buf = io.BytesIO()
            np.save(buf, np.ascontiguousarray(val, dtype=np.float64))
            zf.writestr(f"{key}.npy", buf.getvalue())


def save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(arr, dtype=np.float64))


def _numpy_eval(ki, chi, dchi, lapl_chi, hess_chi, scal):
    from xckernel.emitters.cbackend import scal_order
    from xckernel.emitters.codegen import collapse, compile_function, generate_collapsed

    ck = collapse(ki)
    gen = generate_collapsed(ki, "npk", batch=False)
    fn = compile_function(gen)
    args = [scal["w"], chi, dchi]
    if gen.uses_lapl_chi:
        args.append(lapl_chi)
    if "hess_chi" in ck.params:
        args.append(hess_chi)
    for p in ck.params:
        if p in ("w", "chi", "dchi", "lapl_chi", "hess_chi"):
            continue
        if p.startswith("hess_rho"):
            args.append(np.stack([scal[f"{p}_{c}"] for c in H6]))
        elif p.startswith(("grad_rho", "jp")) or p.startswith("dgrad_rho"):
            args.append(np.stack([scal[f"{p}_{ax}"] for ax in "xyz"]))
        else:
            args.append(scal[p])
    return fn(*args), scal_order(ck)


def _expand_into(scal, key, val):
    val = np.ascontiguousarray(val, dtype=np.float64)
    if val.ndim == 2 and val.shape[0] == 3:
        for i, ax in enumerate("xyz"):
            scal[f"{key}_{ax}"] = np.ascontiguousarray(val[i])
        return
    if val.ndim == 2 and val.shape[0] == 6:
        for i, comp in enumerate(H6):
            scal[f"{key}_{comp}"] = np.ascontiguousarray(val[i])
        return
    scal[key] = np.ascontiguousarray(val.reshape(-1))


def regen_s2jz() -> dict:
    from pylibxc import LibXCFunctional

    from xckernel.engine.kernel import fock
    from xckernel.engine.response import response_fock
    from xckernel.tests.validate import ingredients_from_P, make_grid

    g = make_grid(nbf=4, npts=200, seed=1)
    family_fn = {"lda": "LDA_X", "gga": "GGA_X_PBE", "mgga_tau": "MGGA_X_SCAN"}
    family_vars = {
        "lda": ["rho"],
        "gga": ["rho", "sigma"],
        "mgga_tau": ["rho", "sigma", "tau"],
    }
    dchi_c = np.ascontiguousarray(np.transpose(g.dchi, (1, 0, 2)))
    save_npz(
        DATA / "randgrid_s1_nbf4_ng200_operands.npz",
        w=g.w,
        chi=g.chi,
        dchi=g.dchi,
        lapl_chi=g.lapl_chi,
        P=g.P,
    )
    for fam, xcname in family_fn.items():
        ing = ingredients_from_P(g, g.P)
        func = LibXCFunctional(xcname, "unpolarized")
        out = func.compute({v: ing[v] for v in family_vars[fam]})
        scal = {"w": np.ascontiguousarray(g.w)}
        if fam != "lda":
            _expand_into(scal, "grad_rho", ing["_grad"])
        for key, arr in out.items():
            if key == "zk" or arr is None:
                continue
            _expand_into(scal, key, arr)
        save_npz(DATA / f"{fam}_r_o1_scal.npz", **scal)
        ref, _ = _numpy_eval(
            fock(fam), g.chi, dchi_c, g.lapl_chi, np.zeros((6, 4, 200)), scal
        )
        save_npy(DATA / f"{fam}_r_o1_fock_ref.npy", ref)

    # GGA fxc on H2O/sto-3g (response_validate.check_pyscf operands)
    from pylibxc import LibXCFunctional as LXC
    from pyscf import dft, gto
    from pyscf.dft import numint

    from xckernel.tests.response_validate import _pert_fields

    mol = gto.M(
        atom="O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24",
        basis="sto-3g",
        verbose=0,
    )
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build()
    mf = dft.RKS(mol)
    mf.xc = "LDA,VWN"
    mf.verbose = 0
    mf.kernel()
    dm0 = mf.make_rdm1()
    rng = np.random.default_rng(0)
    a = rng.standard_normal(dm0.shape)
    dm1 = a + a.T
    ao = numint.eval_ao(mol, grids.coords, deriv=1)
    rho0 = numint.eval_rho(mol, ao, dm0, xctype="GGA")
    chi = np.ascontiguousarray(ao[0].T)
    dchi = np.ascontiguousarray(np.transpose(ao[1:4], (0, 2, 1)))
    grad_rho = rho0[1:4]
    inp = {
        "rho": rho0[0],
        "sigma": np.einsum("ig,ig->g", grad_rho, grad_rho),
    }
    xc_out = LXC("GGA_X_PBE", "unpolarized").compute(inp, do_fxc=True)
    rho1, grad1, tau1 = _pert_fields(dm1, chi, dchi)
    scal = {"w": np.ascontiguousarray(grids.weights)}
    _expand_into(scal, "grad_rho", grad_rho)
    _expand_into(scal, "grad_rho_p1", grad1)
    _expand_into(scal, "rho_p1", rho1)
    _expand_into(scal, "tau_p1", tau1)
    for key, arr in xc_out.items():
        if key == "zk" or arr is None:
            continue
        _expand_into(scal, key, arr)
    extra = dict(scal)
    extra.update(
        weights=grids.weights,
        chi=chi,
        dchi=dchi,
        dm0=dm0,
        dm1=dm1,
        w=grids.weights,
    )
    save_npz(DATA / "mol_h2o_sto3g_lvl3_operands.npz", **extra)
    ref, _ = _numpy_eval(
        response_fock("gga", 2),
        chi,
        dchi,
        np.zeros_like(chi),
        np.zeros((6,) + chi.shape),
        scal,
    )
    save_npy(DATA / "gga_r_o2_fxc_ref.npy", ref)
    return {
        "seed": 1,
        "nbf": 4,
        "ng": 200,
        "kernels": ["xck_lda_r_o1", "xck_gga_r_o1", "xck_mgga_tau_r_o1", "xck_gga_r_o2"],
        "libxc_ids": ["LDA_X", "GGA_X_PBE", "MGGA_X_SCAN"],
    }


def regen_c_vs_numpy() -> None:
    from xckernel.engine.kernel import fock
    from xckernel.engine.response import response_fock
    from xckernel.tests.cbackend_validate import _operands

    from xckernel.emitters.cbackend import scal_order
    from xckernel.emitters.codegen import collapse

    cases = [
        ("xck_lda_r_o1", fock("lda")),
        ("xck_gga_r_o1", fock("gga")),
        ("xck_gga_r_o2", response_fock("gga", 2)),
        ("xck_mgga_tau_r_o1", fock("mgga_tau")),
    ]
    dest = DATA / "c_vs_numpy"
    dest.mkdir(parents=True, exist_ok=True)
    for name, ki in cases:
        ck = collapse(ki)
        chi, dchi, lapl_chi, hess_chi, scal = _operands(ck, 4, 60, 11)
        payload = dict(scal)
        payload.update(chi=chi, dchi=dchi, lapl_chi=lapl_chi, hess_chi=hess_chi)
        save_npz(dest / f"{name}_operands.npz", **payload)
        ref, names = _numpy_eval(ki, chi, dchi, lapl_chi, hess_chi, scal)
        assert names == scal_order(ck)
        save_npy(dest / f"{name}_ref.npy", ref)


def _mol_grid():
    from pyscf import dft, gto
    from pyscf.dft import numint

    mol = gto.M(
        atom="O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24",
        basis="sto-3g",
        verbose=0,
    )
    grids = dft.gen_grid.Grids(mol)
    grids.level = 3
    grids.build()
    mf = dft.RKS(mol)
    mf.xc = "LDA,VWN"
    mf.verbose = 0
    mf.kernel()
    dm0 = mf.make_rdm1()
    rng = np.random.default_rng(0)
    a = rng.standard_normal(dm0.shape)
    dm1 = a + a.T
    return mol, grids, mf, dm0, dm1, numint


def regen_pyscf() -> None:
    from pylibxc import LibXCFunctional
    from pyscf.dft import numint as ni_mod

    from xckernel.engine.kernel import fock
    from xckernel.engine.response import response_fock
    from xckernel.tests.response_validate import _pert_fields

    mol, grids, mf, dm0, dm1, numint = _mol_grid()
    dest = DATA / "pyscf_h2o_sto3g"
    dest.mkdir(parents=True, exist_ok=True)
    save_npy(dest / "dm0.npy", dm0)
    save_npy(dest / "dm1.npy", dm1)
    meta = {
        "nao": int(mol.nao),
        "ngrid": int(len(grids.weights)),
        "geometry": "O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24",
        "basis": "sto-3g",
        "grids.level": 3,
        "seeds": {"dm1": 0, "tda_z": 4, "rpa_xy": 5, "nz": 3},
        "functionals": {
            "lda": "LDA_X,",
            "gga": "GGA_X_PBE,",
            "mgga_tau": "MGGA_X_SCAN,",
        },
        "libxckernel_rev": PIN,
    }
    try:
        import pyscf
        import pylibxc

        meta["pyscf_version"] = getattr(pyscf, "__version__", "unknown")
        meta["pylibxc_version"] = getattr(pylibxc, "__version__", "unknown")
    except Exception:
        pass
    (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    specs = [
        ("lda", "LDA_X", "LDA", 0, "lda_fock"),
        ("gga", "GGA_X_PBE", "GGA", 1, "gga_fock"),
        ("mgga_tau", "MGGA_X_SCAN", "MGGA", 1, "mgga_tau_fock"),
    ]
    ni = ni_mod.NumInt()
    for fam, xcname, xctype, deriv, stem in specs:
        ao = numint.eval_ao(mol, grids.coords, deriv=deriv)
        rho0 = numint.eval_rho(mol, ao, dm0, xctype=xctype)
        if xctype == "LDA":
            chi = np.ascontiguousarray(ao.T)
            dchi = np.zeros((3,) + chi.shape)
            grad_rho = None
            inp = {"rho": rho0}
        else:
            chi = np.ascontiguousarray(ao[0].T)
            dchi = np.ascontiguousarray(np.transpose(ao[1:4], (0, 2, 1)))
            grad_rho = rho0[1:4]
            inp = {
                "rho": rho0[0],
                "sigma": np.einsum("ig,ig->g", grad_rho, grad_rho),
            }
            if xctype == "MGGA":
                inp["tau"] = rho0[-1]
        xc_out = LibXCFunctional(xcname, "unpolarized").compute(inp)
        scal = {"w": np.ascontiguousarray(grids.weights)}
        if grad_rho is not None:
            _expand_into(scal, "grad_rho", grad_rho)
        for key, arr in xc_out.items():
            if key == "zk" or arr is None:
                continue
            _expand_into(scal, key, arr)
        extra = dict(scal)
        extra.update(chi=chi, dchi=dchi, w=grids.weights, dm0=dm0)
        save_npz(dest / f"{stem}_operands.npz", **extra)
        ref_xk, _ = _numpy_eval(
            fock(fam),
            chi,
            dchi,
            np.zeros_like(chi),
            np.zeros((6,) + chi.shape),
            scal,
        )
        _, _, vref = ni.nr_rks(mol, grids, xcname + ",", dm0)
        err = float(np.max(np.abs(ref_xk - vref)))
        scale = float(np.max(np.abs(vref)) or 1.0)
        rel = err / scale
        print(f"  {fam} Fock vs nr_rks abs={err:.3e} rel={rel:.3e}")
        if rel > 1e-14:
            sys.exit(f"{fam} Fock vs nr_rks rel={rel:.3e} exceeds ~1e-15 decade")
        save_npy(dest / f"{stem}_ref.npy", vref)
        save_npy(dest / f"{stem}_xk.npy", ref_xk)

    # GGA fxc
    ao = numint.eval_ao(mol, grids.coords, deriv=1)
    rho0 = numint.eval_rho(mol, ao, dm0, xctype="GGA")
    chi = np.ascontiguousarray(ao[0].T)
    dchi = np.ascontiguousarray(np.transpose(ao[1:4], (0, 2, 1)))
    grad_rho = rho0[1:4]
    inp = {
        "rho": rho0[0],
        "sigma": np.einsum("ig,ig->g", grad_rho, grad_rho),
    }
    xc_out = LibXCFunctional("GGA_X_PBE", "unpolarized").compute(inp, do_fxc=True)
    rho1, grad1, tau1 = _pert_fields(dm1, chi, dchi)
    scal = {"w": np.ascontiguousarray(grids.weights)}
    _expand_into(scal, "grad_rho", grad_rho)
    _expand_into(scal, "grad_rho_p1", grad1)
    _expand_into(scal, "rho_p1", rho1)
    _expand_into(scal, "tau_p1", tau1)
    for key, arr in xc_out.items():
        if key == "zk" or arr is None:
            continue
        _expand_into(scal, key, arr)
    extra = dict(scal)
    extra.update(chi=chi, dchi=dchi, w=grids.weights, dm0=dm0, dm1=dm1)
    save_npz(dest / "gga_fxc_operands.npz", **extra)
    ref_xk, _ = _numpy_eval(
        response_fock("gga", 2),
        chi,
        dchi,
        np.zeros_like(chi),
        np.zeros((6,) + chi.shape),
        scal,
    )
    Rref = ni.nr_rks_fxc(mol, grids, "GGA_X_PBE,", dm0, dm1, hermi=0)
    err = float(np.max(np.abs(ref_xk - Rref)))
    scale = float(np.max(np.abs(Rref)) or 1.0)
    if err / scale > 1e-13:
        sys.exit(f"GGA fxc vs nr_rks_fxc rel={err / scale:.3e} exceeds 1e-13")
    save_npy(dest / "gga_fxc_ref.npy", Rref)

    # TDA / RPA pins from live PySCF (tda_validate.py convention)
    from pyscf import dft as dft_mod
    from pyscf.tdscf.rhf import gen_tdhf_operation

    rngz = np.random.default_rng(4)
    rngxy = np.random.default_rng(5)
    for fam, xc in (("lda", "LDA_X,"), ("gga", "GGA_X_PBE,")):
        mf_x = dft_mod.RKS(mol)
        mf_x.xc = xc
        mf_x.verbose = 0
        mf_x.kernel()
        td = mf_x.TDA()
        td.singlet = True
        vind, _hdiag = td.gen_vind(mf_x)
        nocc = int((mf_x.mo_occ > 0).sum())
        nvir = mol.nao - nocc
        zs = rngz.standard_normal((3, nocc, nvir))
        sig = vind(zs.reshape(3, -1)).reshape(3, nocc, nvir)
        save_npy(dest / f"tda_{fam}_sigma_ref.npy", sig)
        op, _ = gen_tdhf_operation(mf_x, singlet=True)
        xys = rngxy.standard_normal((3, 2, nocc, nvir))
        rpa = op(xys.reshape(3, -1)).reshape(3, 2, nocc, nvir)
        save_npy(dest / f"rpa_{fam}_sigma_ref.npy", rpa)


def write_manifest() -> None:
    files = {}
    for path in sorted(DATA.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "MANIFEST.json":
            continue
        rel = path.relative_to(DATA).as_posix()
        files[rel] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".npy":
            arr = np.load(path)
            files[rel]["shape"] = list(arr.shape)
            files[rel]["dtype"] = str(arr.dtype)
    man = {
        "libxckernel_rev": PIN,
        "regen": "scripts/regen_xckernel_goldens.py",
        "host_only": "rg.terra",
        "tolerances": {
            "c_vs_numpy": 1e-16,
            "fock_vs_pyscf": 1e-15,
            "fxc_vs_pyscf": 1e-13,
            "tda_rpa_vs_pyscf": 1e-17,
        },
        "seeds": {"randgrid": 1, "c_vs_numpy": 11, "dm1": 0, "tda": 4, "rpa": 5},
        "nbf": 4,
        "ng_rand": 200,
        "ng_cnp": 60,
        "kernels": [
            "xck_lda_r_o1",
            "xck_gga_r_o1",
            "xck_gga_r_o2",
            "xck_mgga_tau_r_o1",
        ],
        "libxc_ids": ["LDA_X", "GGA_X_PBE", "MGGA_X_SCAN"],
        "files": files,
    }
    (DATA / "MANIFEST.json").write_text(json.dumps(man, indent=2) + "\n")
    print("MANIFEST.json")
    for rel, rec in files.items():
        print(f"  {rec['sha256']}  {rel}")


def main(argv=None) -> int:
    _require_terra()
    _ensure_xckernel()
    p = argparse.ArgumentParser()
    p.add_argument("--s2jz", action="store_true")
    p.add_argument("--c-vs-numpy", action="store_true")
    p.add_argument("--pyscf", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args(argv)
    do_all = args.all or not (args.s2jz or args.c_vs_numpy or args.pyscf)
    if do_all or args.s2jz:
        regen_s2jz()
    if do_all or args.c_vs_numpy:
        regen_c_vs_numpy()
    if do_all or args.pyscf:
        regen_pyscf()
    write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
