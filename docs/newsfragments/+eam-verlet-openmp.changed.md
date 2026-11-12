`EAMPot` keeps a Verlet neighbour list and reuses it while every atom
moves by at most half of one fifth of the widest cutoff.
One thread matches the previous energy and forces bit for bit.
`OMP_NUM_THREADS` above 1 splits atoms across threads; a fixed count
repeats bit for bit and stays within a relative tolerance of 1e-9 of
the one-thread result.
