`pair_style rgpot` with `mode auto`, the default, takes the domain path whenever the potential allows it, on one rank as on several: a potential built by name or with `u0 psi` that declares a finite range, has the density and force stages, and runs with the analytic virial.
The rule rests on what the potential can do, not on the rank count or the system, because the domain path is then the faster one; on one rank the 16384-atom Lennard-Jones box of `bench_pair_rgpot.py` takes 8.27 ms per step by domain against 15.10 ms gathered, and the 32000-atom CuNi box 46.84 against 70.10 ms.
One-rank runs of LJ, `u0 psi`, Morse, SuttonChen and EAMTable therefore agree with earlier releases to rounding, not bit for bit, and `mode gather` keeps the old path.
Engines, potentials without a finite range or the two stages, and `virial fd` still gather on rank 0, and `init` logs the reason.
`mode gather` and `mode domain` are unchanged.
