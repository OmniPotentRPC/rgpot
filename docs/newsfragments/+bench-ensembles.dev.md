`engines/lammps/bench_pair_rgpot.py --ensemble npt|deform` times the boxes under `fix npt` or `fix deform`, which change the cell on every step, as well as under NVE.
It also records the list builds that the plugin logs for each timed run on the gather path.
