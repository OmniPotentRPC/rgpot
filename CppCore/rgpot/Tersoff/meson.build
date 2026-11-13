# Static convenience lib; ships inside the librgpot umbrella (link_whole).
tersoff_sic = static_library(
    'tersoff_sic',
    'TersoffSiC.cc',
    dependencies: _deps,
    cpp_args: _args + _fp_contract_off,
    include_directories: ['../../'],
    pic: true,
    install: false,
)
