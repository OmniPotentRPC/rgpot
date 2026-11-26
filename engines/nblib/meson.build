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
nblib_rpath_dirs = []
if nblib_prefix != ''
    foreach sub : ['lib', 'lib64', 'lib.AVX2_256', 'lib.AVX_256', 'lib.SSE2']
        candidate = nblib_prefix / sub
        nblib_search += [candidate]
        if fs.is_dir(candidate)
            nblib_rpath_dirs += [candidate]
        endif
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
nblib_rpath = ':'.join(nblib_rpath_dirs)

if nblib_lib.found() and nblib_gmx_core.found()
    nblib_engine = shared_library(
        'nblib_engine',
        files(
            'NblibEngine.cc',
            'NblibSession.cc',
            'NblibTopology.cc',
            'SessionCApi.cc',
        ) + engine_abi_stamp_source,
        dependencies: [nblib_lib, nblib_gmx_core],
        # librgpot hides C++ symbols. The listed virial uses MinimumImage,
        # so the pic archive is linked here.
        link_with: [rgpot_minimage, rgpot_exact_fma],
        include_directories: [engines_inc, rgpot_inc, include_directories('.')],
        cpp_args: rgpot_consumer_args + nblib_compile + ['-DRGPOT_ENGINE_BUILD'],
        # GROMACS keeps process state. A dlclose must not unmap it.
        link_args: cppc.get_supported_link_arguments(['-Wl,-z,nodelete']),
        build_rpath: nblib_rpath,
        install_rpath: nblib_rpath,
        install: not meson.is_subproject(),
    )
    standalone_engine_plugins += nblib_engine
    nblib_env = environment()
    foreach dir : nblib_rpath_dirs
        nblib_env.prepend('LD_LIBRARY_PATH', dir)
    endforeach
    nblib_session_test = executable(
        'test_nblib_session',
        files('test_nblib_session.cpp'),
        dependencies: [lj_dep, engines_rgpot],
        include_directories: [engines_inc, rgpot_inc],
        cpp_args: rgpot_consumer_args,
    )
    nblib_analytic_test = executable(
        'test_nblib_analytic',
        files('test_nblib_analytic.cpp'),
        dependencies: [engines_rgpot],
        include_directories: [
            engines_inc,
            rgpot_inc,
            include_directories('../..'),
        ],
        cpp_args: rgpot_consumer_args,
    )
    have_nblib_engine = true
    test(
        'nblib_analytic',
        nblib_analytic_test,
        args: [nblib_engine.full_path()],
        depends: nblib_engine,
        env: nblib_env,
        timeout: 300,
    )
    test(
        'nblib_session',
        nblib_session_test,
        args: [nblib_engine.full_path()],
        depends: nblib_engine,
        env: nblib_env,
        timeout: 120,
    )
else
    message('nblib engine skipped: GROMACS nblib libraries were not found')
endif
