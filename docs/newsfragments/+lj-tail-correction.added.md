`LJPot` takes an opt-in `tail_correction` flag on `LJConfig` (default `false`)
and threads it through `PotentialConfig.lj`
(`LennardJonesParams.tailCorrection`), `PotentialConfig.composite`,
`BuiltinParams`, `rgpot::spec::build`, `rgpot_core::lj::LjConfig`,
`rgpot_lj_eval_ext`, `liblj_engine`, `rgpot.LJPot`, `rgpot.SumPot`,
`rgpot.ExprPot` and `RgpotCalculator`. On a fully periodic cell it adds the
continuum long-range correction for the energy, the Cauchy stress and the
per-atom energies and virials, and leaves the pair cutoff unshifted. With a
nonzero `switch_width` the correction also restores the part of the bare 12-6
term that the quintic switch removes over `[cutoff - switch_width, cutoff]` via
adaptive Gauss-Kronrod quadrature precomputed at construction, and unsupported
modes (`switch_width >= cutoff`, staged or domain-decomposed evaluation,
`LJClusterPot`, and non-3D-periodic `RgpotCalculator` cells) are rejected.
