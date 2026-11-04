# Static convenience lib; ships inside the librgpot umbrella (link_whole).
# fp-contract off so the unit Lennard-Jones tree is not fused.
reduced_pair = static_library(
    'reduced_pair',
    'ReducedPair.cc',
    dependencies: _deps,
    cpp_args: _args + ['-ffp-contract=off', '-fno-fast-math'],
    include_directories: ['../../'],
    pic: true,
    install: false,
)
