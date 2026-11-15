The PLUMED action `RGPOT` evaluates any rgpot potential by name:
`POTENTIAL=Name OPTIONS=key=value,...` with `ATOMS` and `ELEMENTS` or
`SPECIES1`, `SPECIES2`, ... for the element of each atom. The energy comes
with matching forces and box derivatives, from the potential's stress or,
with `VIRIAL=FD` or no stress, from a strain difference. The cell is the
PLUMED box, or with `NOPBC` a cluster cell padded by `VACUUM`. Units are
converted from PLUMED's, rank 0 evaluates and the result is broadcast, and
the old `U0`, `PSI` and `CUTOFF` input still gives the shifted LJ dimer.
`gmx mdrun -plumed` runs it with zero GROMACS parameters. A non-positive
`cutoff` passed to `spec::build` now keeps the potential's own cutoff.
