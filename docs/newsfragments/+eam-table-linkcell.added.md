<!-- Copyright 2023--present rgpot developers -->

Added `EamTable::cu_u3(path)`, `EamTablePotential::cu_u3(path, periodic)`,
`EamTable::from_bytes(bytes, format)`, and
`EamTablePotential::from_bytes(bytes, format, periodic)` constructors for
user-supplied LAMMPS EAM tables without embedding `Cu_u3.eam` in the crate,
and routed `EamTablePotential` neighbour lists and optional Verlet skin
(`with_skin`) through `linkcell::SkinList`.
