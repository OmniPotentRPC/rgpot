`SuttonChenPot` writes the Cauchy stress when `SuttonChenConfig::stress` is set.
The flag is off by default, and with it clear the energy, forces, capabilities and parameter key are unchanged.
The stress matches LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) `pair_style eam/fs` to 2e-13 eV/Angstrom^3 on a cubic and a sheared platinum cell.
