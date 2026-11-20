<!-- Copyright 2023--present rgpot developers -->

Meson and CMake read the vendored vesin release from third_party/vesin/VERSION
and stop at configure time when the bundled source does not record that pin.
A carriage return before the newline on the marker line is accepted. potctl
and the potentials CI install vesin wheels from that minor series, and the
lockstep check runs in the Python CI tests.
