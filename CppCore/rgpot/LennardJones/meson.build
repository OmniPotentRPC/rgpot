# target("avx2") on the cluster kernel is an ifunc resolver. A TSan
# build leaves that unit uninstrumented; see _no_tsan_ifunc in
# CppCore/meson.build. The pot sources stay instrumented.
_lj_kernel = static_library(
    'lj_cluster_kernel',
    'LJClusterKernel.cc',
    dependencies: _deps,
    cpp_args: _args + ['-ffp-contract=off'] + _no_tsan_ifunc,
    include_directories: ['../../'],
    pic: true,
    install: false,
)
# Static convenience lib; ships inside the librgpot umbrella (link_whole).
lennard_jones = static_library(
    'lennard_jones',
    'LJPot.cc',
    'LJClusterPot.cc',
    dependencies: _deps,
    # Contraction off: the cluster energy pins are the uncontracted
    # i < j sum. The kernel adds each pair in that order.
    cpp_args: _args + ['-ffp-contract=off'],
    include_directories: ['../../'],
    link_whole: [_lj_kernel],
    pic: true,
    install: false,
)
