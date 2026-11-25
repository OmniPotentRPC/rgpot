After 3.8.0, librgpot moves to soversion 10. `ORCAPot` now loads liborcac
and holds its state behind one pointer, and `ORCAConfig` gained the guess
policy, the library path, user blocks and the fresh retry, with `restart` kept
as a deprecated member, so a program built against `librgpot.so.9` is
rebuilt. The C ABI stamp
of `rgpot.h` is unchanged.
