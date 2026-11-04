`forceBatch` now zeroes the energy and forces of every result slot before the kernel runs, as it already did for the variance and stress.
Before, a reused buffer kept an earlier call's energy and forces when the kernel threw, for example when a plugin reported an error or returned a non-finite value.
