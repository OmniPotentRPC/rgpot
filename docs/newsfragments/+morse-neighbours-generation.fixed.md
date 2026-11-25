`Neighbours::generation()` is a process-unique identifier that changes at
construction and on every rebuild, copy and move.
It comes from one counter exported by `librgpot`.
The `MorsePot` stored-pair workspace and the `fastTiles` stencil cache are keyed
on it instead of an object address, so an object built at the address of a
destroyed one does not reuse that object's pair index, slice splits or stencil.
Periodic and partially periodic skewed cells are binned in fractional coordinates
from the internal minimum-image inverse, wrapped on periodic axes and spanned
on open axes, so a sheared stream with zero skin does not miss pairs across
cell faces.
A singular or ill-conditioned cell in fractional binning, or a non-finite
fractional coordinate, is an error instead of falling back to Cartesian
binning.
Orthorhombic cells and fully open skewed cells use Cartesian binning.
The C++ soversion of `librgpot` is 9.
