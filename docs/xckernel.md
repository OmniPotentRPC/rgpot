# XcKernel: in-process libxckernel contractions

`rgpot::XcKernel` is the first-slice host for
[libxckernel](https://github.com/susilehtola/libxckernel)
(arXiv:2608.26440, pin `d6a9d57ba3fe0f2763667ce168d0c0ef21cff4a4`).
It is **not** a geometry PES.

## What it is

- New C++ type under `CppCore/rgpot/XcKernel/`.
- Inputs: AO collocation (`chi`, `dchi`, optional `lapl_chi` / `hess_chi`),
  grid weights, named Libxc derivative arrays, and (for fxc) perturbed fields.
- Outputs: AO matrices accumulated `+=` (XC Fock at order 1, fxc contraction
  at order 2).
- Term ownership is XC-only. Coulomb, Hartree-Fock exact exchange, and
  range-separated exchange stay host-owned.

## What it is not

- Not a `Potential` subclass and not a `PotType`.
- No `PotentialConfig.xckernel` arm (rgpot-qf6b). The wire schema still
  carries `ForceInput` / `PotentialResult` / `PotentialConfig` for energy
  surfaces. Kernel operands are not those carriers, and no DFT host sends
  `chi` / `dchi` / weights over potserv in this slice.
- The default wheel and default meson build do **not** ship the Python
  generator. `with_xckernel` defaults to false.

## Build

Meson option, same shape as `with_xtb`:

    meson setup bbdir-xck -Dwith_xckernel=true -Dwith_tests=true

Generate the first-slice C package **on rg.terra only**:

    pixi install -e xckernel
    pixi run -e xckernel -- bash scripts/gen_libxckernel.sh

Families: `lda,gga,mgga_tau`. `max_order`: 2 (27 C kernels: 3 energy helpers
+ 9 Fock + 15 fxc). Later slices (o3/o4, GIAO, noncollinear, hmgga,
`mgga_lapl`) stay out of this tree.

## Dependencies

Libxc owns the functional-derivative tower. Numerical evaluation needs
`pylibxc`. **`pylibxc` is not installable from PyPI**; the `pylibxc2` name
there is an unrelated empty stub. Use conda-forge `pylibxc` (libxc-feedstock
Python output) via the pixi feature `xckernel`. PySCF lives only in
`xckerneltest` for golden masters.

    pixi install -e xckernel        # libxc + sympy + numpy; `import pylibxc`
    pixi install -e xckerneltest    # plus pyscf + scipy
    # wrong: python -m pip install pylibxc
    # wrong: python -m pip install pylibxc2

The compiled libxckernel runtime does not link pylibxc. The host passes
already-mixed derivative arrays.

## Golden masters

Fixtures live under `CppCore/tests/data/xckernel/`. Tests fail closed if a
named file is missing. Regenerator:

    pixi run -e xckerneltest -- python scripts/regen_xckernel_goldens.py

That script refuses to run off rg.terra. Tolerances are the paper/README
bars: C vs NumPy `1e-16`, Fock vs PySCF exclusive `1e-15`, fxc vs PySCF
`1e-13`, TDA/RPA sigma vs PySCF exclusive `1e-17`. `--pyscf` Fock compares
long-double stage A/B to live `nr_rks` and exits when `rel > 1e-15`.
`--tda-rpa` (also part of `--pyscf`) replays committed MOs, writes host-J
/ `st_o2_p` operands, and exits when live `gen_vind` /
`gen_tdhf_operation` drifted past exclusive `1e-17`. Do not invent
looser values.

TDA/RPA assembly is `XcKernel::tdaSigma` / `rpaSigma` over the singlet
`xck_*_st_o2_p` C kernels plus host Coulomb. Transition densities, ov
projection, and perturbed fields follow the PySCF `gen_vind` /
`gen_tdhf_operation` einsum order (scale `Co` by occupancy first;
`eval_rho` GEMM for `rho_p1`). Coulomb `J` stays host-owned (pinned
`tda_*_j.npy` / `rpa_*_j.npy` from PySCF `get_j` on that same DM).
