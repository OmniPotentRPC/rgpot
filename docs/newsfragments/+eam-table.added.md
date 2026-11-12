`EAMTablePot` reads embedded atom method (EAM) and Finnis-Sinclair potentials from the LAMMPS (Large-scale Atomic/Molecular Massively Parallel Simulator) `funcfl`, `setfl` and `eam/fs` tables, with energy, forces and an optional Cauchy stress.
It is built only with `-Dwith_eam_table=true` (CMake `RGPOT_WITH_EAM_TABLE`), so the default build is unchanged.
On `Cu_u3.eam` and `CuNi.eam.alloy` from the LAMMPS `potentials` directory it matches `pair_style eam` and `eam/alloy` to 3.1e-12 eV in the energy and 1.7e-14 eV/Angstrom in the forces.
