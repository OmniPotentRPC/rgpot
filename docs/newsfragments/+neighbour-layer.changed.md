EAM, Lennard-Jones, Sutton-Chen, Stillinger-Weber and Tersoff share
one neighbour layer. It bins atoms into cells, folds the minimum image,
and either stores a Verlet list or walks the cells without storing pairs.
The mode follows the memory the OS reports as available, or a caller can
select it. `OMP_NUM_THREADS` above 1 splits atoms across threads. One
thread matches the previous energy and forces bit for bit. A fixed thread
count repeats bit for bit and stays within a relative tolerance of 1e-9
of the one-thread result.
