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
- Grimme D3/D4 stay in-process `Potential` summands (`D3Pot` / `D4Pot`).
  Do not fold them into `contract` or `applyFxc`. There is no
  `PotentialConfig.d3` / `.d4` arm; a DFT host adds D4 as the Edisp
  term next to this kernel and VV10 (rgpot-8mse).
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

## Stage B

`CppCore/rgpot/XcKernel/xckernel/evaluator.hpp` specializes `stage_b<double>`.
Each block of 256 grid points is one `dgemm`, or one `dsyr2k` when `U` and `V` are the same array, and the block results are added in long double.
`stage_a` still accumulates in long double.
The build links FlexiBLAS when pkg-config finds it, and otherwise the generic blas dependency.
On a release build linked to OpenBLAS 0.3.34 pthreads, with one thread bound to its own core, the random grid (nbf 200, 100000 points) median for `xck_gga_r_o1` was 859.8 ms against 35554.2 ms for the long-double loop.
Four threads on four distinct cores brought that median to 515.0 ms.
On the H2O grid, nbf is 7, and four threads are slower than one.
`xckernel_bench` prints the median, minimum, and maximum over its repetitions.

## Dependencies

Libxc implements the functional-derivative tower. Numerical evaluation needs
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
`--tda-rpa` (also part of `--pyscf`) pins PySCF to one OpenMP thread,
replays committed MOs, reports live host-J without overwriting the
committed `tda_*_j.npy` / `rpa_*_j.npy` pins, writes `st_o2_p`
operands, and exits when live `gen_vind` / `gen_tdhf_operation`
drifted past exclusive `1e-17`. Multi-thread `gen_vind` jitters 1-2 ulp on this case, which is
already past that bar. A fresh RKS kernel on the same mol/xc is past
that bar (MO/energy noise ~1e-15); the exclusive gate is MO-replay
`gen_vind`, not a new SCF. Same-SCF extras and `MANIFEST.json` are
written before that exit. Do not invent looser values.

`scripts/compare_xckernel_tda_nwchemc.py` (rg.terra) calls `libnwchemc`
`nwchemc_energy` with `theory=tddft` / TDA on the same H2O/sto-3g
geometry and writes `tda_{lda,gga}_nwchemc_roots.npy`. That is the
second engine check of the pin operator (TDA.kernel on committed MOs).
The exclusive 1e-17 bar remains the C sigma contraction; the nwchemc
root residual is the measured engine value (`1e-6`).

TDA/RPA assembly is `XcKernel::tdaSigma` / `rpaSigma` over the singlet
`xck_*_st_o2_p` kernels plus host Coulomb. LDA wv is
`w * rho * (v2rho2_0 + v2rho2_1)` (one fused product, matching
`nr_rks_fxc_st`) and stage B tiles the grid at PySCF `BLKSIZE` (128).
GGA `st_o2_p` `applyFxc` uses the generated 7-term monomials through
the host long-double evaluator (`contract()`). Double tiled stage B
misses exclusive `1e-17` vs live `gen_vind` on this sto-3g pin.
Transition densities and ov projection follow the PySCF `lib.einsum`
contraction path (`qo,xov->vxq` then `vxq,pv->xpq` for the TDA DM;
`pv,xpq->vxq` then `vxq,qo->xov` for the ov block). Perturbed fields
follow PySCF `eval_rho` (`c0 = ao @ dm`, GGA `hermi=0` adds
`ao @ dm.T`). Coulomb `J` stays host-owned (pinned `tda_*_j.npy` /
`rpa_*_j.npy` from the `get_j` call inside `gen_vind` /
`gen_tdhf_operation`).
