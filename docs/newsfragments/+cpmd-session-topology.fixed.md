A CPMD calculation that adds an atom, or changes which elements are
present, now gets a new engine session and a cold SCF instead of an
error. The same atom count and element counts still reuse the
converged wavefunction.
