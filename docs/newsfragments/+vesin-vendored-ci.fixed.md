<!-- Copyright 2023--present rgpot developers -->

Meson and CMake now stop at configure time unless the vendored vesin tree is
v0.6.0 plus local eOn patches. potctl and the potentials CI install vesin
wheels from that 0.6 series, and the lockstep check runs in the Python CI
tests.
