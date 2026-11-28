Added safe Rust EAM table factories in `rgpot-core`:
`eam_table_from_file(...)` and `eam_table_from_data(...)`, both returning
owned `PotentialImpl` handles suitable for downstream crates. A funcfl table
gives the same result through either factory.
