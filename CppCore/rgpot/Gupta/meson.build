# Static convenience lib; ships inside the librgpot umbrella (link_whole).
gupta = static_library(
    'gupta',
    'gupta.c',
    'GuptaPot.cc',
    dependencies: _deps,
    c_args: _fp_contract_off,
    cpp_args: _args + _fp_contract_off,
    include_directories: ['.', '../../'],
    pic: true,
    install: false,
)
