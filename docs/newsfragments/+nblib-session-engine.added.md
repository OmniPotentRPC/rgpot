Added a GROMACS nblib engine for the generic MD session ABI, built with
=-Dwith_nblib=enabled=. nblib supplies the forces of a topology given as text;
Langevin dynamics on a hyperplane, planar walls, the constraint force per step
and the seeded random stream are the driver the OpenMM engine runs, so both
engines answer the same calls with the same noise. =nblib_prefix= names a
GROMACS install, and =engines/nblib/build_gromacs.sh= builds the pinned release
with the nblib API.
