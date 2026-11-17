# NblibPot: dlopen frontend of librgpot_nblib (engines/nblib).
#
# GROMACS is linked into librgpot_nblib only. This archive resolves it at
# run time, so librgpot carries no GROMACS dependency.

nblib_dl_dep = dependency('dl', required: false)
if not nblib_dl_dep.found()
    nblib_dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
endif

nblibpot = static_library(
    'nblibpot',
    'NblibPot.cc',
    # _deps_sans_core: the umbrella link_wholes rgpot_core itself.
    dependencies: _deps_sans_core + [nblib_dl_dep],
    cpp_args: _args,
    include_directories: _incdirs,
    pic: true,
    install: false,
)
