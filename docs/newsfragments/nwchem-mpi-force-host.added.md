Added `nwchem_mpi_force_host`: multi-rank SPMD host that drives `NWChemPot`
under `mpirun` so every rank enters the public `nwchemc_energy_gradient` path
(host-owned MPI; Cap'n Proto stays free of ranks/comms).
