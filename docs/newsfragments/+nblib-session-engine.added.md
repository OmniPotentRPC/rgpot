Added a GROMACS nblib engine for the MD session ABI. Build it with
`-Dwith_nblib=auto` and point `-Dnblib_prefix` at a GROMACS install that
ships the nblib API. `engines/nblib/build_gromacs.sh` builds the pinned
release. nblib supplies the forces of a topology given as text. Langevin
dynamics, the hyperplane, planar walls, the constraint force and the seeded
random stream are the driver the OpenMM session runs. Neighbour exchange
returns unsupported, and the library stays loaded because it links GROMACS.
