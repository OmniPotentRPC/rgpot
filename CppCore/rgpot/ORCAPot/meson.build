# ORCA is a subprocess. The frontend always builds on POSIX; the binary
# is resolved at evaluation time.
orcapot = static_library(
    'orcapot',
    'ORCAPot.cc',
    dependencies: _deps,
    cpp_args: _args,
    include_directories: ['../../'],
    pic: true,
    install: false,
)
