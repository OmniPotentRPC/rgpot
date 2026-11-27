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
    nblib_sources = files(
        'NblibEngine.cc',
        'NblibSession.cc',
        'NblibTopology.cc',
        'SessionCApi.cc',
    )
    nblib_deps = [nblib_lib, nblib_gmx_core]
    # The TPR session calls libgromacs directly. GROMACS installs the library
    # and not the headers of those calls, so the source and build trees of
    # the prefix supply them.
    nblib_source = get_option('nblib_source')
    nblib_build = get_option('nblib_build')
    nblib_have_tpr = false
    if nblib_source != '' and nblib_build != ''
        nblib_have_tpr = true
        nblib_sources += files('TprEngine.cc', 'TprSession.cc')
        nblib_deps += dependency('openmp', language: 'cpp')
        nblib_compile += ['-DRGPOT_NBLIB_HAS_TPR', '-DHAVE_CONFIG_H']
        nblib_modules = run_command(
            find_program('find'),
            nblib_source / 'src/gromacs',
            '-mindepth',
            '2',
            '-maxdepth',
            '2',
            '-name',
            'include',
            '-type',
            'd',
            check: true,
        ).stdout().strip().split('\n')
        foreach dir : nblib_modules + [
            nblib_source / 'src',
            nblib_source / 'src/include',
            nblib_source / 'src/external',
            nblib_source / 'src/external/thread_mpi/include',
            nblib_source / 'api/legacy/include',
            nblib_build / 'src/include',
            nblib_build / 'api/legacy/include',
        ]
            nblib_compile += ['-isystem', dir]
        endforeach
        # The headers fix the SIMD layout the library was built with.
        if cppc.compiles(
            '''#include "config.h"
            #if !GMX_SIMD_X86_AVX2_256
            #error other SIMD
            #endif''',
            args: ['-I' + nblib_build / 'src/include'],
            name: 'GROMACS built for AVX2',
        )
            nblib_compile += ['-mavx2', '-mfma']
        elif cppc.compiles(
            '''#include "config.h"
            #if !GMX_SIMD_X86_AVX_512
            #error other SIMD
            #endif''',
            args: ['-I' + nblib_build / 'src/include'],
            name: 'GROMACS built for AVX-512',
        )
            nblib_compile += ['-mavx512f', '-mavx512dq', '-mfma']
        endif
    endif
    nblib_engine = shared_library(
        'nblib_engine',
        nblib_sources,
        dependencies: nblib_deps,
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
    if nblib_have_tpr
        test(
            'nblib_tpr_dispersion',
            python3_runner,
            args: [
                files('test_tpr_dispersion.py'),
                nblib_engine.full_path(),
                nblib_prefix / 'bin' / 'gmx_d',
                meson.current_source_dir() / 'data' / 'tip4p_dispcorr',
            ],
            timeout: 120,
        )
    endif

else
    message('nblib engine skipped: GROMACS nblib libraries were not found')
endif
