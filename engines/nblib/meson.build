# librgpot_nblib: GROMACS nblib potential ABI plus the canonical
# rgpot_md_session_* ABI. One topology yields a potential and an MD session.
#
# nblib ships with GROMACS built with -DGMX_INSTALL_NBLIB_API=ON
# -DGMX_INSTALL_LEGACY_API=ON -DBUILD_SHARED_LIBS=ON; engines/nblib/
# build_gromacs.sh builds the pinned release. The prefix comes from
# nblib_prefix. A double-precision GROMACS (libgromacs_d) is detected and
# gets -DGMX_DOUBLE=1, which nblib's headers do not set themselves.

nblib_prefix = get_option('nblib_prefix')
nblib_search = []
nblib_compile = []
if nblib_prefix != ''
    foreach sub : ['lib', 'lib64']
        nblib_search += [nblib_prefix / sub]
    endforeach
    nblib_compile += ['-isystem', nblib_prefix / 'include']
endif
nblib_lib = cppc.find_library(
    'nblib_gmx',
    dirs: nblib_search,
    required: nblib_opt.enabled(),
)
nblib_gmx_double = cppc.find_library(
    'gromacs_d',
    dirs: nblib_search,
    required: false,
)
nblib_gmx_core = nblib_gmx_double.found() ? nblib_gmx_double : cppc.find_library(
    'gromacs',
    dirs: nblib_search,
    required: nblib_opt.enabled(),
)
if nblib_lib.found() and nblib_gmx_core.found() and nblib_gmx_double.found()
    nblib_compile += ['-DGMX_DOUBLE=1']
endif
nblib_rpath = nblib_prefix != '' ? nblib_prefix / 'lib' : ''

if nblib_lib.found() and nblib_gmx_core.found()
nblib_engine = shared_library(
    'rgpot_nblib',
    files(
        'NblibTopology.cc',
        'NblibEngine.cc',
        'NblibSession.cc',
        'SessionCApi.cc',
    ),
    dependencies: [nblib_lib, nblib_gmx_core],
    link_with: rgpot,
    include_directories: [engines_inc, rgpot_inc, include_directories('.')],
    cpp_args: _args + nblib_compile,
    build_rpath: nblib_rpath,
    install_rpath: nblib_rpath,
    install: not meson.is_subproject(),
)

else
    message('nblib engine skipped: GROMACS nblib libraries were not found')
endif
