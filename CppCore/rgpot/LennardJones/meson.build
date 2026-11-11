# Static convenience lib; ships inside the librgpot umbrella (link_whole).
lennard_jones = static_library(
    'lennard_jones',
    'LJPot.cc',
    'LJClusterPot.cc',
    dependencies: _deps,
    cpp_args: _args + _fp_contract_off,
    include_directories: ['../../'],
    pic: true,
    install: false,
)
