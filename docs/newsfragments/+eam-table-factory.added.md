<!-- Copyright 2023--present rgpot developers -->

Added safe Rust EAM table factories in `rgpot-core`:
`eam_table_from_file(...)` and `eam_table_from_data(...)`, both returning
owned `PotentialImpl` handles suitable for downstream crates. Funcfl tables
keep both grid endpoints when converted for the C++ factory.
