# Static convenience lib; ships inside the librgpot umbrella (link_whole).
lennard_jones = static_library(
    'lennard_jones',
    'LJPot.cc',
    'LJClusterPot.cc',
    'LJClusterKernel.cc',
    dependencies: _deps,
    # Contraction off: the cluster energy pins are the uncontracted
    # i < j sum. The kernel adds each pair in that order.
    cpp_args: _args + ['-ffp-contract=off'],
    include_directories: ['../../'],
    pic: true,
    install: false,
)
