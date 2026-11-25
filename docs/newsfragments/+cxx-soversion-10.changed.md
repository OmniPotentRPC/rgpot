After 3.8.0, librgpot moves to soversion 10. `ORCAPot` now loads liborcac
and holds its state behind one pointer, and `ORCAConfig` replaced `restart`
with the guess policy and gained the library path, user blocks and the fresh
retry, so a program built against `librgpot.so.9` is rebuilt. The C ABI stamp
of `rgpot.h` is unchanged.
