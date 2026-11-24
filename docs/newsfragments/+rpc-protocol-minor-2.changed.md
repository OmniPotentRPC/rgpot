The RPC protocol minor is now 2 in the C++, Rust and Python mirrors and in the generated C header.
The wire check reads the minor from all three sources and fails when they disagree, and it keeps the orbital-key introduction minor from exceeding the current minor.
