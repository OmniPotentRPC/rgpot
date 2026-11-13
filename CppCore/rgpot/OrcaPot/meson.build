# ORCA EnGrad subprocess. No library to link; the `orca` binary is
# resolved when a force is requested.
orcapot = static_library(
    'orcapot',
    'OrcaPot.cc',
    'OrcaEngineAbi.cc',
    dependencies: _deps,
    cpp_args: _args,
    include_directories: ['../../'],
    pic: true,
    install: false,
)
