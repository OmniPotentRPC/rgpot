# libnblib_engine: the canonical rgpot_md_session_* ABI with forces from
# GROMACS nblib. One topology text yields an MD session.
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
    foreach sub : ['lib', 'lib64', 'lib.AVX2_256', 'lib.AVX_256', 'lib.SSE2']
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
    required: nblib_opt.enabled(),
)
nblib_gmx_core = nblib_gmx_double
if nblib_lib.found() and nblib_gmx_double.found()
    nblib_compile += ['-DGMX_DOUBLE=1']
    if nblib_prefix != ''
        nblib_gmx = find_program(
            nblib_prefix / 'bin' / 'gmx_d',
            required: true,
        )
    else
        nblib_gmx = find_program('gmx_d', required: true)
    endif
    nblib_version = run_command(
        nblib_gmx,
        '--version',
        check: true,
    ).stdout()
    if not nblib_version.contains('2025.5')
        error('the nblib engine requires GROMACS 2025.5')
    endif
endif
nblib_rpath = nblib_prefix != '' ? (
    nblib_prefix / 'lib' + ':' + nblib_prefix / 'lib64'
) : ''

if nblib_lib.found() and nblib_gmx_core.found()
    nblib_engine = shared_library(
        'nblib_engine',
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
    nblib_session_test = executable(
        'test_nblib_session',
        files('test_nblib_session.cpp'),
        dependencies: lj_dep,
        link_with: rgpot,
        include_directories: [engines_inc, rgpot_inc],
        cpp_args: _args,
    )
    nblib_analytic_test = executable(
        'test_nblib_analytic',
        files('test_nblib_analytic.cpp'),
        link_with: rgpot,
        include_directories: [
            engines_inc,
            rgpot_inc,
            include_directories('../..'),
        ],
        cpp_args: _args,
    )
    have_nblib_engine = true
    test(
        'nblib_analytic',
        nblib_analytic_test,
        args: [nblib_engine.full_path()],
        timeout: 300,
    )
    test(
        'nblib_session',
        nblib_session_test,
        args: [nblib_engine.full_path()],
        timeout: 120,
    )

else
    message(
        'nblib engine skipped: double-precision GROMACS nblib libraries were not found',
    )
endif
