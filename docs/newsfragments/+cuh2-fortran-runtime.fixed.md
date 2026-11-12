`librgpot_cuh2` links the Fortran runtime again.
The plugin is a C++ link, so the runtime was not inferred and the
shared library was missing `_gfortran_*`.
