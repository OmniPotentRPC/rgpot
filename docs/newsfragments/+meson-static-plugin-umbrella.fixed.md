Meson static builds with RPC or Fortran plugins also build a shared `librgpot`
for those loadable plugins, so each plugin resolves the `Neighbours` generation
counter from one umbrella instead of embedding a private copy. The soversion
check now verifies these symbols on macOS as well as Linux, and CI covers the
static-plus-RPC configuration.
