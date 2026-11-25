# Copyright 2023--present rgpot developers
# Static convenience lib; ships inside the librgpot umbrella (link_whole).
# The orca executable is found at run time: nothing here links or loads ORCA.
orca_legacy_pot = static_library(
    'orca_legacy_pot',
    'ORCAFiles.cc',
    'ORCAGuards.cc',
    'ORCALegacyPot.cc',
    dependencies: _deps,
    cpp_args: _args,
    include_directories: ['../../'],
    pic: true,
    install: false,
)
