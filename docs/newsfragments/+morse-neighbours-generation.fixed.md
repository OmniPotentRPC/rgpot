`Neighbours::generation()` is a process-unique identifier that changes at
construction and on every rebuild, copy and move.
It comes from one counter exported by `librgpot`.
The `MorsePot` stored-pair workspace and the `fastTiles` stencil cache are keyed
on it instead of an object address, so an object built at the address of a
destroyed one does not reuse that object's pair index, slice splits or stencil.
A fully periodic general cell, including a triclinic cell with three positive
edges, is binned by wrapped fractional coordinates, so a sheared stream with
zero skin does not miss pairs across cell faces.
A non-finite fractional coordinate is an error.
A fully periodic general cell whose reciprocal basis is not finite and positive
is an error and is not binned in Cartesian coordinates.
Cartesian binning is used when the cell is not general, or when an axis is open
or its diagonal entry is not positive.
The C++ soversion of `librgpot` is 9.
