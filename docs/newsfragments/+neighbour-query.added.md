A public neighbour query returns a structure-of-arrays table or a chunk
of a walk, with a half list, a full list, or clusters of four. Metatomic
and UMA read that table. The C ABI is version 1, and Potentials.capnp
carries an optional request and a chunk that need not be the whole list.
