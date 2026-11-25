<!-- Copyright 2023--present rgpot developers -->

Add a lammpc-backed constructor path for `LammpsPot` so rgpot can evaluate full
LAMMPS `pair_style` setups from `PotentialConfig.lammps` (`LammpsParams`), while
preserving the existing LJ plugin constructor.
The Python `rgpot_lammps_potential()` binding now has an overload that accepts
flat `PotentialConfig` bytes, and the external C ABI adds
`rgpot_lammpc_potential()`.
The lammpc backend passes the same `Capabilities` handshake as the other
profile backends and is refused, naming its path, on a mismatch. `LammpsPot`
reports stress only when that backend declares the `stress` operation, and its
result-cache key covers the engine path and the configuration in both modes.
