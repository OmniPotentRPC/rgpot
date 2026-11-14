The minimum image no longer drops pairs without a word.
When the cutoff is more than half a perpendicular width of the cell,
`LJPot`, `MorsePot`, periodic `SuttonChenPot` with a positive cutoff,
`EAMTablePot` and the cell-list `EAMPot` check the pairs.
If an atom is inside the cutoff of its own image, or a pair has two images
inside it, they throw `std::invalid_argument` (`ValueError` in Python)
and point to the all-images mode.
Otherwise they return the same minimum-image result as before.
`EAMTablePot` used to reject every such cell; it now accepts those whose
pairs have one image each.
Cells at least twice the cutoff across take the old path unchanged.

- librgpot's ELF soversion is 7.
  LJPot and MorsePot hold an image mode and an all-images pair list.
  A caller built against soversion 6 does not load librgpot.so.7.
  librgpot_mpi stays at soversion 3.
