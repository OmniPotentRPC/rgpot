#!/usr/bin/env python3
"""Regenerate first-slice XcKernel golden masters.

Run on rg.terra only. Do not invoke from meson test or on the laptop.

    pixi run -e xckerneltest python scripts/regen_xckernel_goldens.py
    pixi run -e xckerneltest python scripts/regen_xckernel_goldens.py --c-vs-numpy
    pixi run -e xckerneltest python scripts/regen_xckernel_goldens.py --pyscf
    pixi run -e xckerneltest python scripts/regen_xckernel_goldens.py --all

pylibxc comes from conda-forge libxc. Do not pip-install pylibxc or pylibxc2.
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

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "CppCore" / "tests" / "data" / "xckernel"
PIN = "d6a9d57ba3fe0f2763667ce168d0c0ef21cff4a4"
FAMILY_FUNCTIONAL = {
    "lda": "LDA_X",
    "gga": "GGA_X_PBE",
    "mgga_tau": "MGGA_X_SCAN",
}

# Refuse a laptop-side regen unless the operator explicitly overrides.
_TERRA_HINTS = ("terra", "rg.terra")


def _on_terra() -> bool:
    if os.environ.get("RGPOT_XCKERNEL_REGEN_OK") == "1":
        return True
    host = socket.gethostname().lower()
    return any(h in host for h in _TERRA_HINTS)


def _require_terra() -> None:
    if not _on_terra():
        sys.stderr.write(
            "regen_xckernel_goldens.py runs on rg.terra only "
            "(set RGPOT_XCKERNEL_REGEN_OK=1 to override)\n"
        )
        raise SystemExit(2)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_xckernel() -> None:
    try:
        import xckernel  # noqa: F401
    except ImportError:
        src = Path("/tmp/libxckernel-src")
        if not src.is_dir():
            sys.stderr.write(
                "xckernel is not importable; clone the pin to "
                "/tmp/libxckernel-src or set PYTHONPATH\n"
            )
            raise
        sys.path.insert(0, str(src))


def _save_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.ascontiguousarray(arr, dtype=np.float64))
    print(f"  {_sha256(path)}  {path.relative_to(ROOT)}")


def _save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = {k: np.ascontiguousarray(v, dtype=np.float64) for k, v in arrays.items()}
    np.savez(path, **packed)
    print(f"  {_sha256(path)}  {path.relative_to(ROOT)}")


def _libxc_unpol(name: str, inp: dict, do_fxc: bool = False) -> dict:
    from pylibxc import LibXCFunctional

    func = LibXCFunctional(name, "unpolarized")
    out = func.compute(inp, do_fxc=do_fxc)
    return {k: np.ascontiguousarray(v).reshape(-1) for k, v in out.items() if v is not None}


def _ingredients(P, chi, dchi, lapl_chi=None):
    """chi (nbf,ng); dchi (nbf,3,ng) or (3,nbf,ng)."""
    if dchi.shape[0] == 3 and dchi.ndim == 3:
        dchi_nbf = np.transpose(dchi, (1, 0, 2))  # (nbf,3,ng)
        dchi_abi = dchi
    else:
        dchi_nbf = dchi
        dchi_abi = np.transpose(dchi, (1, 0, 2))
    rho = np.einsum("uv,ug,vg->g", P, chi, chi)
    grad = np.einsum("uv,uig,vg->ig", P, dchi_nbf, chi) + np.einsum(
        "uv,ug,vig->ig", P, chi, dchi_nbf
    )
    sigma = np.einsum("ig,ig->g", grad, grad)
    tau = 0.5 * np.einsum("uv,uig,vig->g", P, dchi_nbf, dchi_nbf)
    out = {
        "rho": rho,
        "sigma": sigma,
        "tau": tau,
        "grad": grad,
        "dchi_abi": dchi_abi,
        "dchi_nbf": dchi_nbf,
    }
    if lapl_chi is not None:
        lapl = (
            np.einsum("uv,ug,vg->g", P, lapl_chi, chi)
            + 2.0 * np.einsum("uv,uig,vig->g", P, dchi_nbf, dchi_nbf)
            + np.einsum("uv,ug,vg->g", P, chi, lapl_chi)
        )
        out["lapl"] = lapl
    return out


def _numpy_fock(family: str, chi, dchi_abi, scal, lapl_chi=None, hess_chi=None):
    from xckernel.emitters.codegen import compile_function, generate
    from xckernel.engine.kernel import fock

    gen = generate(fock(family), f"fock_{family}")
    fn = compile_function(gen)
    args = [scal["w"], chi, dchi_abi]
    if gen.uses_lapl_chi:
        args.append(lapl_chi)
    if getattr(gen, "uses_hess_chi", False) and hess_chi is not None:
        args.append(hess_chi)
    if gen.uses_grad_rho:
        args.append(
            np.stack([scal["grad_rho_x"], scal["grad_rho_y"], scal["grad_rho_z"]])
        )
    for name in gen.libxc_args:
        args.append(scal[name])
    return fn(*args)


def _numpy_response(family: str, order: int, chi, dchi_abi, scal, lapl_chi=None):
    from xckernel.emitters.codegen import compile_function, generate
    from xckernel.engine.response import response_fock

    gen = generate(response_fock(family, order), f"resp_{family}_{order}")
    fn = compile_function(gen)
    args = [scal["w"], chi, dchi_abi]
    if gen.uses_lapl_chi:
        args.append(lapl_chi)
    if gen.uses_grad_rho:
        args.append(
            np.stack([scal["grad_rho_x"], scal["grad_rho_y"], scal["grad_rho_z"]])
        )
    for lbl in gen.pert_grads or []:
        args.append(
            np.stack(
                [
                    scal[f"grad_rho_{lbl}_x"],
                    scal[f"grad_rho_{lbl}_y"],
                    scal[f"grad_rho_{lbl}_z"],
                ]
            )
        )
    for name in gen.pert_scalars or []:
        args.append(scal[name])
    for name in gen.libxc_args:
        args.append(scal[name])
    return fn(*args)


def regen_randgrid() -> None:
    from xckernel.tests.validate import make_grid

    print("randgrid Fock (nbf=4, npts=200, seed=1)")
    g = make_grid(nbf=4, npts=200, seed=1)
    dchi_abi = np.ascontiguousarray(np.transpose(g.dchi, (1, 0, 2)))
    packed = {
        "w": g.w,
        "chi": g.chi,
        "dchi": g.dchi,
        "lapl_chi": g.lapl_chi,
        "P": g.P,
    }
    for family, func in FAMILY_FUNCTIONAL.items():
        ing = _ingredients(g.P, g.chi, g.dchi, g.lapl_chi)
        inp = {"rho": ing["rho"]}
        if family != "lda":
            inp["sigma"] = ing["sigma"]
        if family == "mgga_tau":
            inp["tau"] = ing["tau"]
        xc = _libxc_unpol(func, inp, do_fxc=False)
        packed[f"{family}_vrho"] = xc["vrho"]
        if family != "lda":
            packed[f"{family}_vsigma"] = xc["vsigma"]
            packed["grad_rho_x"] = ing["grad"][0]
            packed["grad_rho_y"] = ing["grad"][1]
            packed["grad_rho_z"] = ing["grad"][2]
        if family == "mgga_tau":
            packed[f"{family}_vtau"] = xc["vtau"]
        scal = {"w": g.w, "vrho": xc["vrho"]}
        if family != "lda":
            scal.update(
                {
                    "grad_rho_x": ing["grad"][0],
                    "grad_rho_y": ing["grad"][1],
                    "grad_rho_z": ing["grad"][2],
                    "vsigma": xc["vsigma"],
                }
            )
        if family == "mgga_tau":
            scal["vtau"] = xc["vtau"]
        # Flatten into the shared npz under the kernel scal names prefixed
        # by family so one file holds every first-slice Fock operand.
        for k, v in scal.items():
            packed[f"{family}__{k}"] = v
        F = _numpy_fock(family, g.chi, dchi_abi, scal, g.lapl_chi)
        _save_npy(DATA / f"{family}_r_o1_fock_ref.npy", F)
        # Also store unprefixed scal for the family last-written... C++ loads
        # unprefixed names. Write per-family extra keys AND a last-family
        # overwrite is wrong. C++ looks up scal_names() in the same npz.
        # Store each family's unprefixed names in a sibling npz instead.
        _save_npz(
            DATA / f"{family}_r_o1_operands.npz",
            chi=g.chi,
            dchi=g.dchi,
            lapl_chi=g.lapl_chi,
            **scal,
        )
    _save_npz(DATA / "randgrid_s1_nbf4_ng200_operands.npz", **packed)
    # C++ randgrid test looks up unprefixed scal in the shared npz. Duplicate
    # LDA/GGA/mGGA names collide. Merge GGA+mGGA extras (superset) into the
    # shared file under the unprefixed names of the last family only is not
    # enough. Rewrite the C++ path: it already accepts sibling
    # <kernel>_operands.npz via the pyscf case. For randgrid we also write
    # those siblings above. Put GGA-superset unprefixed keys (w, vrho,
    # grad_rho_*, vsigma, vtau) from mgga so all three kernels can find
    # their names; LDA ignores extra keys.
    ing = _ingredients(g.P, g.chi, g.dchi, g.lapl_chi)
    xc_m = _libxc_unpol("MGGA_X_SCAN", {
        "rho": ing["rho"], "sigma": ing["sigma"], "tau": ing["tau"]
    })
    extra = {
        "w": g.w,
        "vrho": xc_m["vrho"],
        "vsigma": xc_m["vsigma"],
        "vtau": xc_m["vtau"],
        "grad_rho_x": ing["grad"][0],
        "grad_rho_y": ing["grad"][1],
        "grad_rho_z": ing["grad"][2],
    }
    # Wrong: mixing SCAN vrho into LDA Fock. C++ must use sibling npz.
    # Keep the shared file as specified (w, chi, dchi, lapl_chi, P) plus
    # prefixed extras only.
    del extra


def regen_gga_fxc_mol() -> None:
    from pylibxc import LibXCFunctional
    from pyscf import dft, gto
    from pyscf.dft import numint
    from xckernel.tests.response_validate import _pert_fields

    print("GGA fxc H2O sto-3g grids.level=3")
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
    out = LibXCFunctional("GGA_X_PBE", "unpolarized").compute(inp, do_fxc=True)
    libxc = {
        k: np.ascontiguousarray(v).reshape(-1)
        for k, v in out.items()
        if k != "zk" and v is not None
    }
    rho1, grad1, _tau1 = _pert_fields(dm1, chi, dchi)
    scal = {
        "w": np.ascontiguousarray(grids.weights),
        "grad_rho_x": np.ascontiguousarray(grad_rho[0]),
        "grad_rho_y": np.ascontiguousarray(grad_rho[1]),
        "grad_rho_z": np.ascontiguousarray(grad_rho[2]),
        "grad_rho_p1_x": np.ascontiguousarray(grad1[0]),
        "grad_rho_p1_y": np.ascontiguousarray(grad1[1]),
        "grad_rho_p1_z": np.ascontiguousarray(grad1[2]),
        "rho_p1": np.ascontiguousarray(rho1),
        "v2rho2": libxc["v2rho2"],
        "v2sigma2": libxc["v2sigma2"],
        "vsigma": libxc["vsigma"],
        "v2rhosigma": libxc["v2rhosigma"],
    }
    R = _numpy_response("gga", 2, chi, dchi, scal)
    _save_npy(DATA / "gga_r_o2_fxc_ref.npy", R)
    packed = {
        "weights": grids.weights,
        "w": grids.weights,
        "chi": chi,
        "dchi": dchi,
        "dm0": dm0,
        "dm1": dm1,
    }
    packed.update(libxc)
    packed.update(scal)
    _save_npz(DATA / "mol_h2o_sto3g_lvl3_operands.npz", **packed)


def regen_c_vs_numpy() -> None:
    from xckernel.emitters.cbackend import scal_order
    from xckernel.emitters.codegen import collapse, compile_function, generate_collapsed
    from xckernel.engine.kernel import fock
    from xckernel.engine.response import response_fock

    print("C-vs-NumPy pins (nbf=4, ng=60, seed=11)")
    cases = [
        ("xck_lda_r_o1", fock("lda")),
        ("xck_gga_r_o1", fock("gga")),
        ("xck_gga_r_o2", response_fock("gga", 2)),
        ("xck_mgga_tau_r_o1", fock("mgga_tau")),
    ]
    nbf, ng, seed = 4, 60, 11
    dest = DATA / "c_vs_numpy"
    dest.mkdir(parents=True, exist_ok=True)
    for name, ki in cases:
        ck = collapse(ki)
        gen = generate_collapsed(ki, "npk", batch=False)
        fn = compile_function(gen)
        rng = np.random.default_rng(seed)
        chi = np.ascontiguousarray(rng.standard_normal((nbf, ng)))
        dchi = np.ascontiguousarray(rng.standard_normal((3, nbf, ng)))
        lapl_chi = np.ascontiguousarray(rng.standard_normal((nbf, ng)))
        hess_chi = np.ascontiguousarray(rng.standard_normal((6, nbf, ng)))
        scal = {n: np.ascontiguousarray(rng.standard_normal(ng)) for n in scal_order(ck)}
        scal["w"] = np.abs(scal["w"]) + 0.1
        args = [scal["w"], chi, dchi]
        if gen.uses_lapl_chi:
            args.append(lapl_chi)
        if "hess_chi" in ck.params:
            args.append(hess_chi)
        for p in ck.params:
            if p in ("w", "chi", "dchi", "lapl_chi", "hess_chi"):
                continue
            if p.startswith("hess_rho"):
                args.append(np.stack([scal[f"{p}_{c}"] for c in ("xx", "xy", "xz", "yy", "yz", "zz")]))
            elif p.startswith(("grad_rho", "jp")):
                args.append(np.stack([scal[f"{p}_{ax}"] for ax in "xyz"]))
            else:
                args.append(scal[p])
        ref = fn(*args)
        _save_npy(dest / f"{name}_ref.npy", ref)
        _save_npz(
            dest / f"{name}_operands.npz",
            chi=chi,
            dchi=dchi,
            lapl_chi=lapl_chi,
            hess_chi=hess_chi,
            **scal,
        )


def regen_pyscf() -> None:
    from pylibxc import LibXCFunctional
    from pyscf import dft, gto, lib
    from pyscf.dft import numint
    from xckernel.tests.response_validate import _pert_fields

    print("PySCF H2O sto-3g cross-check refs")
    dest = DATA / "pyscf_h2o_sto3g"
    dest.mkdir(parents=True, exist_ok=True)
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
    ni = numint.NumInt()

    meta = {
        "nao": int(mol.nao),
        "ngrid": int(grids.weights.size),
        "geometry": "O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24",
        "basis": "sto-3g",
        "grids.level": 3,
        "seeds": {"dm1": 0, "tda_zs": 4, "rpa_xys": 5},
        "functionals": {
            "lda": "LDA_X,",
            "gga": "GGA_X_PBE,",
            "mgga_tau": "MGGA_X_SCAN,",
        },
        "libxckernel_rev": PIN,
        "pyscf": getattr(lib, "__version__", "unknown"),
    }
    try:
        import pylibxc

        meta["pylibxc"] = getattr(pylibxc, "__version__", "conda-forge-libxc")
    except Exception:
        meta["pylibxc"] = "unknown"
    dest.joinpath("meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  {_sha256(dest / 'meta.json')}  {(dest / 'meta.json').relative_to(ROOT)}")
    _save_npy(dest / "dm0.npy", dm0)
    _save_npy(dest / "dm1.npy", dm1)

    def fock_ref(xc: str) -> np.ndarray:
        _e, _v, vmat = ni.nr_rks(mol, grids, xc, dm0)
        return np.ascontiguousarray(vmat)

    def fxc_ref(xc: str) -> np.ndarray:
        return np.ascontiguousarray(ni.nr_rks_fxc(mol, grids, xc, dm0, dm1, hermi=0))

    _save_npy(dest / "lda_fock_ref.npy", fock_ref("LDA_X,"))
    _save_npy(dest / "gga_fock_ref.npy", fock_ref("GGA_X_PBE,"))
    _save_npy(dest / "mgga_tau_fock_ref.npy", fock_ref("MGGA_X_SCAN,"))
    _save_npy(dest / "gga_fxc_ref.npy", fxc_ref("GGA_X_PBE,"))

    # Per-kernel operands so XcKernel can reproduce the PySCF refs.
    for family, func, xctype, deriv in (
        ("lda", "LDA_X", "LDA", 0),
        ("gga", "GGA_X_PBE", "GGA", 1),
        ("mgga_tau", "MGGA_X_SCAN", "MGGA", 1),
    ):
        ao = numint.eval_ao(mol, grids.coords, deriv=deriv)
        rho0 = numint.eval_rho(mol, ao, dm0, xctype=xctype)
        if xctype == "LDA":
            chi = np.ascontiguousarray(ao.T)
            dchi = np.zeros((3,) + chi.shape)
            inp = {"rho": rho0}
            grad_rho = None
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
        xc = LibXCFunctional(func, "unpolarized").compute(inp, do_fxc=True)
        libxc = {
            k: np.ascontiguousarray(v).reshape(-1)
            for k, v in xc.items()
            if k != "zk" and v is not None
        }
        scal = {"w": np.ascontiguousarray(grids.weights), "vrho": libxc["vrho"]}
        if grad_rho is not None:
            scal.update(
                {
                    "grad_rho_x": np.ascontiguousarray(grad_rho[0]),
                    "grad_rho_y": np.ascontiguousarray(grad_rho[1]),
                    "grad_rho_z": np.ascontiguousarray(grad_rho[2]),
                    "vsigma": libxc["vsigma"],
                }
            )
        if "vtau" in libxc:
            scal["vtau"] = libxc["vtau"]
        if family != "lda":
            rho1, grad1, tau1 = _pert_fields(dm1, chi, dchi)
            scal.update(
                {
                    "rho_p1": np.ascontiguousarray(rho1),
                    "grad_rho_p1_x": np.ascontiguousarray(grad1[0]),
                    "grad_rho_p1_y": np.ascontiguousarray(grad1[1]),
                    "grad_rho_p1_z": np.ascontiguousarray(grad1[2]),
                    "tau_p1": np.ascontiguousarray(tau1),
                }
            )
            for k in ("v2rho2", "v2sigma2", "v2rhosigma", "v2tau2", "v2rhotau", "v2sigmatau"):
                if k in libxc:
                    scal[k] = libxc[k]
        _save_npz(
            dest / f"xck_{family}_r_o1_operands.npz",
            chi=chi,
            dchi=dchi,
            **scal,
        )
        if family == "gga":
            _save_npz(
                dest / "xck_gga_r_o2_operands.npz",
                chi=chi,
                dchi=dchi,
                **scal,
            )

    # TDA / RPA pins vs PySCF (LDA/GGA only). Match tda_validate.py.
    from pyscf.tdscf.rhf import gen_tdhf_operation

    nocc = mol.nelectron // 2
    nvir = mol.nao - nocc
    rng4 = np.random.default_rng(4)
    zs = rng4.standard_normal((3, nocc, nvir))
    rng5 = np.random.default_rng(5)
    xys = rng5.standard_normal((3, 2, nocc, nvir))
    for tag, xc in (("lda", "LDA_X,"), ("gga", "GGA_X_PBE,")):
        mf_x = dft.RKS(mol)
        mf_x.xc = xc
        mf_x.grids = grids
        mf_x.verbose = 0
        mf_x.kernel()
        td = mf_x.TDA()
        td.singlet = True
        vind, _hdiag = td.gen_vind(mf_x)
        sig = vind(zs.reshape(3, -1)).reshape(3, nocc, nvir)
        _save_npy(dest / f"tda_{tag}_sigma_ref.npy", sig)
        rvind, _ = gen_tdhf_operation(mf_x, singlet=True)
        rpa = rvind(xys.reshape(3, -1)).reshape(3, 2, nocc, nvir)
        _save_npy(dest / f"rpa_{tag}_sigma_ref.npy", rpa)


def write_manifest() -> None:
    files = {}
    for path in sorted(DATA.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "MANIFEST.json":
            continue
        rel = str(path.relative_to(DATA))
        files[rel] = _sha256(path)
    manifest = {
        "libxckernel_rev": PIN,
        "families": ["lda", "gga", "mgga_tau"],
        "max_order": 2,
        "seeds": {"randgrid": 1, "c_vs_numpy": 11, "dm1": 0},
        "nbf_randgrid": 4,
        "ng_randgrid": 200,
        "kernels": [
            "xck_lda_r_o1",
            "xck_gga_r_o1",
            "xck_gga_r_o2",
            "xck_mgga_tau_r_o1",
        ],
        "libxc_ids": FAMILY_FUNCTIONAL,
        "regen": "scripts/regen_xckernel_goldens.py --all",
        "files": files,
    }
    dest = DATA / "MANIFEST.json"
    dest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"  {_sha256(dest)}  {dest.relative_to(ROOT)}")


def main() -> int:
    _require_terra()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--c-vs-numpy", action="store_true")
    p.add_argument("--pyscf", action="store_true")
    p.add_argument("--all", action="store_true")
    args = p.parse_args()
    _ensure_xckernel()
    do_s2jz = args.all or not (args.c_vs_numpy or args.pyscf)
    if args.all:
        args.c_vs_numpy = True
        args.pyscf = True
        do_s2jz = True
    DATA.mkdir(parents=True, exist_ok=True)
    if do_s2jz:
        regen_randgrid()
        regen_gga_fxc_mol()
    if args.c_vs_numpy:
        regen_c_vs_numpy()
    if args.pyscf:
        regen_pyscf()
    write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
