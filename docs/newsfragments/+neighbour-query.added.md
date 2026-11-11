A public neighbour query returns a structure-of-arrays table or a chunk
of a walk. A pair list is a half shell or a full shell. A cluster of four
is a separate shape and stores every neighbour. Metatomic and UMA read
that table. The C ABI is version 4: `rgpot_nlist_request_init` sets the
defaults, `bytes` is the struct size, and `rgpot_nlist_strerror` names a
code. Potentials.capnp carries the same columns, plus a cursor so a client
can stream one chunk at a time. The query walks the same cell list as the
classical potentials, so a half list of a million atoms does not visit
every pair.
