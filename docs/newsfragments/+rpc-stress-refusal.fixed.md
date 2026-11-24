<!-- Copyright 2023--present rgpot developers -->

The Rust RPC server now fails a stress request that its potential answers without a stress, with the message the C++ server uses, instead of replying with energy and forces and no stress. It also serves the `calculateStress` method. A cell whose volume overflows or is not finite no longer yields a zero stress tensor from the Rust EAM table; the stress is left unset, as in C++.
