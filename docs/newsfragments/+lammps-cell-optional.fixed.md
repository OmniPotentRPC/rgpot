`rgpot.lammps.FixExternal` now reads `caps()["cell_optional"]` before
`caps()["periodic"]`. A potential that reports it, such as `rgpot.D2Pot`, is
accepted on `boundary f f f` and evaluated with an all-zero cell, so it returns
the energy and forces of the isolated system instead of being refused; periodic
boxes still pass the box cell. A potential that is periodic without
`cell_optional` is still refused on a free box, and the error now says why. The
`caps()` dictionary of every bound potential gained the boolean entry
`cell_optional`, true for `rgpot.D2Pot` and `rgpot.D2CpmdPot`.
