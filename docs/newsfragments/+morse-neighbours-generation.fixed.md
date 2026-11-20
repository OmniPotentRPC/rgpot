`Neighbours` now assigns a process-unique generation identifier (`Neighbours::generation()`) at construction and on every rebuild, copy and move through a single exported counter in `librgpot`, so `MorsePot` stored-pair workspaces and `fastTiles` stencil caches never reuse stale data when a new object is allocated at a recycled address.
A fully periodic general cell, including a triclinic cell with three positive edges, is binned by wrapped fractional coordinates so streaming sheared walks with zero skin do not miss pairs across cell faces.
A non-finite fractional coordinate is rejected.
A fully periodic general cell whose reciprocal basis is not finite and positive is rejected and is not binned in Cartesian coordinates.
Cartesian binning remains the path when the cell is not general, or when an axis is open or its diagonal entry is not positive.
`librgpot`'s C++ soversion is bumped to 9.
