<!-- Copyright 2023--present rgpot developers -->

`fortranpots::TersoffPot` now advertises and publishes analytic Cauchy stress
on both dispatch paths of `PotType::TersoffSi`: the all-silicon Fortran gather
and the mixed-species fallback to `TersoffSiC`. The gather computes `dE/dε`
analytically in the same Voigt order and sign convention as `stress.hpp`.
