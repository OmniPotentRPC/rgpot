# Static convenience lib; ships inside the librgpot umbrella (link_whole).
_lj_src = files('LJPot.cc', 'LJClusterPot.cc')
if have_krs
    _lj_src += files('LJKokkos.cc')
endif
lennard_jones = static_library(
    'lennard_jones',
    _lj_src,
    dependencies: _deps + krs_deps,
    cpp_args: _args,
    include_directories: ['../../'],
    pic: true,
    install: false,
)
