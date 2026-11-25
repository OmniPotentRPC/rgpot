Meson static builds outside Windows also build a shared `librgpot` for their
loadable plugins: `librgpot_cuh2`, the RPC engines, and the `engines/` plugins
such as `liblj_engine` and `libnative_lj_engine`. Each plugin resolves the
`Neighbours` generation counter from that one umbrella instead of embedding a
private copy. The `CxxSoversion` test checks those plugins too, and reads the
symbols with `nm` on ELF and Mach-O binaries alike, chosen from each file's
magic bytes. CI covers the static-plus-RPC configuration.
