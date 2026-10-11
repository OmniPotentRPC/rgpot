3.5.0 changed the C++ layout of PotCaps, LJConfig, LJClusterConfig and
MorseConfig and should have bumped the librgpot soversion.
It kept soversion 3, so a caller built against 3.4.0 loaded 3.5.0's
librgpot.so.3 and crashed.
3.5.1 installs that layout as librgpot.so.4.
Callers built against 3.4.0 fail to load it.
Callers built against 3.5.0 need to relink.
The C ABI stamp stays at major 1, minor 0, layout revision 1, because no C
struct changed layout and the C API has no shared library of its own.
librgpot_mpi stays at soversion 3, because its C ABI did not change.
