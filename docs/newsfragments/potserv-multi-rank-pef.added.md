Added multi-rank PEF `potserv` for NWChem/CPMD: under `mpirun -np P`, rank 0
serves Cap'n Proto TCP while ranks 1..P-1 mirror `calculate`/`configure` via
host-owned MPI (same engine env as single-rank).
