# ORCACPot: pure consumer of the split orcac shim (liborcac.so), resolved by
# dlopen like libmopacc. Nothing here links or loads ORCA.

orcacpot = static_library(
    'orcacpot',
    'ORCACPot.cc',
    # _deps_sans_core, as for mopacpot: the umbrella link_wholes rgpot_core.
    dependencies: _deps_sans_core + [mopac_dl_dep],
    cpp_args: _args,
    include_directories: _incdirs,
    pic: true,
    install: false,
)
