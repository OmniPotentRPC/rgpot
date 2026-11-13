# target("avx2") on the 12-6 walk is compiled into that function only.
# A TSan build leaves the resolver uninstrumented; see _no_tsan_ifunc.
_sc_kernel = static_library(
    'sutton_chen_kernel',
    'SuttonChenKernel.cc',
    dependencies: _deps,
    cpp_args: _args + _fp_contract_off + _no_tsan_ifunc,
    include_directories: ['../../'],
    pic: true,
    install: false,
)
# Static convenience lib; ships inside the librgpot umbrella (link_whole).
sutton_chen = static_library(
    'sutton_chen',
    'SuttonChenPot.cc',
    'SuttonChenScales.cc',
    dependencies: _deps,
    cpp_args: _args,
    include_directories: ['../../'],
    link_whole: [_sc_kernel],
    pic: true,
    install: false,
)
