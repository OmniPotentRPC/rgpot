# librgpot_nblib: GROMACS nblib behind the rgpot_nblib_* C ABI
# (CppCore/rgpot/NblibPot/nblib_c_abi.h). One topology yields a potential
# (energy, forces, virial) and an MD session over nblib's integrator.
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
nblib_lib = cppc.find_library('nblib_gmx', dirs: nblib_search, required: true)
nblib_gmx_double = cppc.find_library(
    'gromacs_d',
    dirs: nblib_search,
    required: false,
)
nblib_gmx_core = nblib_gmx_double.found() ? nblib_gmx_double : cppc.find_library(
    'gromacs',
    dirs: nblib_search,
    required: true,
)
if nblib_gmx_double.found()
    nblib_compile += ['-DGMX_DOUBLE=1']
endif
nblib_rpath = nblib_prefix != '' ? nblib_prefix / 'lib' : ''

nblib_engine = shared_library(
    'rgpot_nblib',
    files(
        'NblibTopology.cc',
        'NblibEngine.cc',
        'NblibSession.cc',
        'NblibCApi.cc',
    ),
    dependencies: [nblib_lib, nblib_gmx_core],
    link_with: rgpot,
    include_directories: [engines_inc, rgpot_inc, include_directories('.')],
    cpp_args: _args + nblib_compile,
    build_rpath: nblib_rpath,
    install_rpath: nblib_rpath,
    gnu_symbol_visibility: 'hidden',
    install: not meson.is_subproject(),
)

nblib_here = meson.current_source_dir()
nblib_pytest = python3_engines.found() ? run_command(
    python3_engines,
    '-c',
    'import pytest, numpy',
    check: false,
).returncode() == 0 : false
if nblib_pytest
    nblib_env = environment()
    nblib_env.set('RGPOT_NBLIB_ENGINE', nblib_engine.full_path())
    nblib_env.set('RGPOT_EVAL', rgpot_eval.full_path())
    nblib_args = [
        '-m',
        'pytest',
        '-q',
        '-p',
        'no:cacheprovider',
        '--rootdir',
        nblib_here,
    ]
    test(
        'nblib_api',
        python3_engines,
        args: nblib_args + [files('test_nblib_api.py')],
        env: nblib_env,
        depends: [nblib_engine, rgpot_eval],
        workdir: meson.current_build_dir(),
        timeout: 300,
    )

    nblib_gmx_bin = get_option('nblib_gmx')
    if nblib_gmx_bin == '' and nblib_prefix != ''
        nblib_gmx_prog = find_program(
            nblib_prefix / 'bin' / 'gmx_d',
            required: false,
        )
        nblib_gmx_bin = nblib_gmx_prog.found() ? nblib_gmx_prog.full_path() : ''
    endif
    if nblib_gmx_bin == ''
        nblib_gmx_prog = find_program('gmx_d', required: false)
        nblib_gmx_bin = nblib_gmx_prog.found() ? nblib_gmx_prog.full_path() : ''
    endif
    if nblib_gmx_bin != ''
        nblib_ref_env = nblib_env
        nblib_ref_env.set('RGPOT_GMX', nblib_gmx_bin)
        test(
            'nblib_gmx_rerun',
            python3_engines,
            args: nblib_args + ['-s', files('test_nblib_rerun.py')],
            env: nblib_ref_env,
            depends: [nblib_engine, rgpot_eval],
            workdir: meson.current_build_dir(),
            timeout: 900,
        )
    else
        message(
            'nblib mdrun -rerun tests skipped: set nblib_gmx to a double-precision gmx_d',
        )
    endif
else
    message('nblib tests skipped: needs python3 with pytest and numpy')
endif
