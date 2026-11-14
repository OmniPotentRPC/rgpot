An opt-in all-images mode for `LJPot`, `MorsePot`, `SuttonChenPot`,
`EAMTablePot` and the cell-list `EAMPot`.
It sums every periodic image inside the cutoff, including an atom's own
images, so cells narrower than twice the cutoff (a one-atom fcc primitive
cell, a thin slab, a small triclinic cell) give the energy, forces and
stress of the infinite crystal, as ASE and LAMMPS do.
Self-image terms count once per pair of opposite shifts.
Turn it on with `LJPot(config, nlist::Images::All)`,
`MorsePot(config, nlist::Images::All)`, `SuttonChenConfig::images`,
`EAMTableConfig::images`, `EAMCellConfig::images`, `periodic =
RGPOT_SC_ALL_IMAGES` in the Sutton-Chen C entry point, `all_images=True`
on the Python `LJPot` and `MorsePot`, and `all_images=True` in the
`RgpotCalculator` Lennard-Jones keywords.
The pairs come from the existing `nlist::Builder` half table, with shift
bounds from the perpendicular widths of the cell.
