<!-- Copyright 2023--present rgpot developers -->
# ORCAPot: ORCA through a child process

`ORCAPot` writes an ORCA input with `! EnGrad`, runs the `orca` executable in
a working directory of its own (closing extra file descriptors before `exec`)
and reads the `.engrad` file. ORCA is neither
linked nor loaded, and no install path is built in: the executable is
`ORCAConfig::executable` or `orca` on `PATH` (including symlinks to
executables). Stale `.engrad` and `.gbw` files are removed before each run,
and any removal failure other than a missing file throws `ORCAError`. Input
formatting and `.engrad` parsing use the C locale. The child is the leader of
a new process group. `ORCAConfig::timeout_s` bounds the wait (one hour by
default; 0 waits with no limit). Past the limit the group is sent SIGTERM
and, if any process in the group is still running, SIGKILL even if the leader
has exited. The child is reaped and the call
throws `ORCAError`.

| File | Role |
| --- | --- |
| `ORCAPot.hpp`, `ORCAPot.cc` | the `Potential`, the child process and the restart |
| `ORCAFiles.hpp`, `ORCAFiles.cc` | input text and `.engrad` parsing, no processes |

Units: ORCA works in Angstrom for `* xyz`, Hartree and Hartree/Bohr in the
`.engrad`. The frontend returns eV and eV/Angstrom using `units.hpp`.

See `docs/orgmode/howto/orca.org` and the ORCA section of
`docs/orgmode/reference/cpp_backends.org`.
