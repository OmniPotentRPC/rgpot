<!-- Copyright 2023--present rgpot developers -->

Added `EamTable::cu_u3()` and `EamTablePotential::cu_u3(periodic)` constructors
for the bundled `Cu_u3.eam` table, and routed `EamTablePotential` neighbour
lists and optional Verlet skin (`with_skin`) through `linkcell::SkinList`.
