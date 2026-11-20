<!-- Copyright 2023--present rgpot developers -->

Added safe Rust EAM table factories in `rgpot-core`:
`eam_table_from_file(...)` and `eam_table_from_data(...)`, both returning
owned `PotentialImpl` handles suitable for downstream crates.
The factories validate geometry before crossing the C++ boundary, accept
nonperiodic inputs without a cell, and preserve minimal funcfl table grids.
