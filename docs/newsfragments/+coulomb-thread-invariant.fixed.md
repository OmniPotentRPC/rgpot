`CoulombPot` DSF, Ewald and SPME, dispersion, and the D3 and D4 pair reductions
give bit-identical energies, forces and stress at every thread count, and the
long-range potentials honour the count set on them. Their sums keep one partial
per fixed chunk, sized from the pair, atom, wave or mesh counts and not from
the team, and fold the partials in chunk order. SPME adds its terms in the
order of the one-thread path when the mesh runs beside real space, so a
charged cell keeps the same energy bits. Without an explicit count a
long-range call still sizes its team from the OpenMP pool; a call made inside
a region the caller opened runs on its own thread and leaves the pool alone,
and an explicit count wins over the recorded caller team. Each `CoulombPot`
keeps its own screen cache, wave tables and mesh grids, so two instances on
different threads share no scratch state, and a warm call whose pairs have not
changed since its own last call skips the pair fingerprint, whichever term
refreshed a shared pair list.
