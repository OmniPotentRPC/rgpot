<!-- Copyright 2023--present rgpot developers -->

Add a lammpc-backed constructor path for `LammpsPot` so rgpot can evaluate full
LAMMPS `pair_style` setups from `PotentialConfig.lammps` (`LammpsParams`),
while preserving the existing LJ plugin constructor. The Python
`rgpot_lammps_potential()` binding now has an overload that accepts flat
`PotentialConfig` bytes, and the external C ABI adds `rgpot_lammpc_potential()`.
