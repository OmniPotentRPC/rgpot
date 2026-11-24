The nblib session engine opens a GROMACS run input file. The spec `tpr PATH`
evaluates the nbnxm, PME, listed and position-restraint forces of the file in
its triclinic cell, spreads the forces of virtual sites, exposes the massive
atoms, and holds the SETTLE and LINCS distances in every run. Features of the
file it does not evaluate are refused by name. Build with `-Dnblib_source` and
`-Dnblib_build` naming the GROMACS trees of `nblib_prefix`.
