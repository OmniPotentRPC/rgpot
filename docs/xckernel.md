# XcKernel: in-process XC response contractions

`rgpot::XcKernel` is the first-slice host for
[libxckernel](https://github.com/susilehtola/libxckernel) (arXiv:2608.26440).
It is **not** a geometry PES. Do not subclass `Potential`. Do not add a
`PotentialConfig.xckernel` arm (rgpot-qf6b): the operands are not
`ForceInput`, the results are not `PotentialResult`, and no DFT host yet
configures this over potserv.

## What it owns

libxckernel contracts grid collocation and named Libxc derivative arrays
into AO Fock-like matrices. **Term ownership is XC-only.** Coulomb, HF,
and range-separated exchange stay host-owned. The compiled library never
evaluates functionals; the host passes coefficient-mixed derivative
arrays.

First slice (meson `-Dwith_xckernel=true`, default **false**):

- families: `lda`, `gga`, `mgga_tau`
- max order: 2 (Fock o1 + fxc o2)
- 27 C kernels (GIAO skipped; no o3/o4, no cmgga/hmgga)

Pin: `d6a9d57ba3fe0f2763667ce168d0c0ef21cff4a4` in
`subprojects/libxckernel.wrap`. Generated C lives under
`third_party/libxckernel/` (produced on **rg.terra** by
`scripts/gen_libxckernel.sh`).

## Deps (pixi feature, not the default env)

```
pixi install -e xckernelbld      # conda-forge libxc + sympy + numpy
pixi install -e xckerneltest     # plus pyscf for golden regen
```

`pylibxc` is a conda-forge output of the `libxc` feedstock (`pixi add
pylibxc`, never PyPI). Do **not** `pip install pylibxc` (not on PyPI) and
do **not** `pip install pylibxc2` (unrelated empty stub). The compiled
`XcKernel` runtime does not need pylibxc.

## Build (rg.terra only)

```
pixi shell -e xckernelbld
meson setup bbdir-xck -Dwith_xckernel=true -Dwith_tests=true
meson compile -C bbdir-xck
meson test -C bbdir-xck --suite xckernel --print-errorlogs
```

Default `-Dwith_xckernel=false` leaves the existing configure unchanged.

## Golden masters

Pinned arrays live under `CppCore/tests/data/xckernel/`. Tests fail closed
if any named file is missing. Tolerances are the paper/README values:

| comparison | tol |
|---|---|
| C backend vs NumPy | 1e-16 |
| Fock vs PySCF `nr_rks` / `nr_uks` | 1e-15 |
| fxc vs PySCF `nr_*_fxc` | 1e-13 |

Regen (rg.terra only, never implicit in meson test):

```
pixi run -e xckerneltest python scripts/regen_xckernel_goldens.py --all
```

## API sketch

```cpp
#include "rgpot/XcKernel/XcKernel.hpp"

rgpot::XcKernel k("xck_gga_r_o2");
auto names = k.scal_names();   // self-describing, from the C ABI
auto F = k.contract(npts, nbf, chi, dchi, /*lapl*/nullptr, /*hess*/nullptr, scal);
```

Missing named operands throw before the C call. Do not share one `out`
buffer across threads (per-instance reentrancy).
