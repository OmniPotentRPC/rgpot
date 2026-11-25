`ORCAPot` writes `%scf ConvForced true end` and refuses the gradient of a call
whose output reports an unconverged SCF. The result-cache key now holds the
`orca` path with symlinks resolved and the size, modification time, change time,
device, inode and content hash of that file, with the charge and multiplicity,
so results of another ORCA installation or spin state are not served. A padded
cluster box, such as the cube of at least 100 Angstrom from `rgpot.ase`, is
accepted, and a cell in which the atoms would wrap is refused with a message
naming the cell vector. `ORCAConfig::restart` now defaults to false, so every
call starts from ORCA's own guess. A `maxcore_mb` of 0 leaves `%maxcore` out,
and the comment on it now gives ORCA 6.1's default of 4096 MB.
