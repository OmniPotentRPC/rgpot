<!-- Copyright 2023--present rgpot developers -->

Added generated reference data for the QSC potential under `CppCore/tests/data/qsc/` with their generators in `scripts/qsc_reference/`: energies and forces of the real eOn `QSC` class from a pinned source snapshot for pure elements, and a 50-digit evaluation of the published equations (energy, forces and stress) for mixtures, cross-checked against LAMMPS `pair_style eam/fs`. `QSCPotTest` and `tests/python/test_qsc_reference.py` compare `QSCPot` with both at 1e-12 relative.
