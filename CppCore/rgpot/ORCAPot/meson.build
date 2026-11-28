# Copyright 2023--present rgpot developers
# Static convenience lib; ships inside the librgpot umbrella (link_whole).
# The orca executable is found at run time: nothing here links or loads ORCA.
orca_pot = static_library(
    'orca_pot',
    'ORCAFiles.cc',
    'ORCAPot.cc',
    dependencies: _deps,
    cpp_args: _args,
    include_directories: ['../../'],
    pic: true,
    install: false,
)
