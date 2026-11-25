# Engine drivers. Each one is built when its library is found.
# with_* = enabled and a missing library is an error.

fs = import('fs')
engines_inc = include_directories('.')
rgpot_inc = include_directories(
    '../CppCore',
    '../include',
    '../rgpot-core/include',
)
# include_directories() rejects an absolute path that sits inside the
# source tree. A pixi prefix and a LAMMPS checkout under the workspace
# both do. A path relative to this file is accepted; a path outside the
# tree can stay absolute.
project_root = meson.project_source_root()
engines_here = meson.current_source_dir()
dl_dep = cppc.find_library('dl', required: false)
thread_dep = dependency('threads')
# A Windows DLL exports only what is marked dllexport, and librgpot marks
# nothing but its version. On Windows the engines link the objects the
# umbrella is made of, as the CppCore tests do.
if host_system == 'windows'
    engines_rgpot = declare_dependency(
        link_with: _linkto,
        dependencies: _deps + [vesin_dep, fastnlist_dep] + fortran_rt_deps,
    )
else
    engines_rgpot = declare_dependency(link_with: rgpot)
endif
python3_runner = find_program('python3', required: true)
engine_exports_script = files('../scripts/check_engine_exports.py')
engine_exports_expected = files('engine_exports_expected.txt')
# The dynamic table is the engine C ABI. A coverage build otherwise
# exports the gcov runtime (__gcov_* and mangle_path).
engine_export_link_args = []
if host_machine.system() == 'linux'
    engine_export_link_args = [
        '-Wl,--version-script,' + engines_here / 'engine_exports.map',
    ]
elif host_machine.system() == 'darwin'
    engine_export_link_args = [
        '-Wl,-exported_symbols_list,' + engines_here / 'engine_exports.darwin.txt',
    ]
endif

coupled_well_engine = shared_library(
    'coupled_well_engine',
    files('coupled_well_engine.cpp'),
    include_directories: [engines_inc, rgpot_inc],
    cpp_args: _args + [
        '-DRGPOT_ENGINE_BUILD',
        '-DRGPOT_ENGINE_PROJECT_VERSION="@0@"'.format(meson.project_version()),
        '-fvisibility-inlines-hidden',
    ],
    link_args: engine_export_link_args,
    gnu_symbol_visibility: 'hidden',
    install: not meson.is_subproject(),
)
if not meson.is_subproject()
    install_headers(
        '../include/rgpot/coupled_well_engine.h',
        subdir: 'rgpot',
    )
endif
test(
    'coupled_well_engine',
    executable(
        'test_coupled_well_engine',
        files('test_coupled_well_engine.c'),
        link_with: coupled_well_engine,
        dependencies: thread_dep,
        include_directories: [engines_inc, rgpot_inc],
        c_args: [
            '-DRGPOT_ENGINE_PROJECT_VERSION="@0@"'.format(
                meson.project_version(),
            ),
        ],
        link_language: 'cpp',
    ),
    timeout: 60,
)
test(
    'coupled_well_engine_exports',
    python3_runner,
    args: [
        engine_exports_script,
        coupled_well_engine.full_path(),
        engine_exports_expected,
    ],
    depends: coupled_well_engine,
    timeout: 30,
)

potlibc_opt = get_option('with_potlibc')
potlibc_dep = dependency('potlibc', required: potlibc_opt)
if potlibc_dep.found()
    potlibc_engine = shared_library(
        'potlibc_engine',
        files('potlibc_engine.c'),
        dependencies: [potlibc_dep, thread_dep],
        include_directories: [engines_inc, rgpot_inc],
        c_args: [
            '-DRGPOT_ENGINE_BUILD',
            '-DRGPOT_ENGINE_PROJECT_VERSION="@0@"'.format(
                meson.project_version(),
            ),
        ],
        link_args: engine_export_link_args,
        gnu_symbol_visibility: 'hidden',
        install: not meson.is_subproject(),
    )
    if not meson.is_subproject()
        install_headers(
            '../include/rgpot/potlibc_engine.h',
            subdir: 'rgpot',
        )
    endif
    potlibc_engine_fail_lock = shared_library(
        'potlibc_engine_fail_lock',
        files('potlibc_engine.c'),
        dependencies: [potlibc_dep, thread_dep],
        include_directories: [engines_inc, rgpot_inc],
        c_args: [
            '-DRGPOT_ENGINE_BUILD',
            '-DRGPOT_POTLIBC_TEST_LOCK_FAIL',
            '-DRGPOT_ENGINE_PROJECT_VERSION="@0@"'.format(
                meson.project_version(),
            ),
        ],
        gnu_symbol_visibility: 'hidden',
        install: false,
    )
    test(
        'potlibc_engine',
        executable(
            'test_potlibc_engine',
            files('test_potlibc_engine.c'),
            dependencies: [dl_dep, thread_dep],
            include_directories: [engines_inc, rgpot_inc],
        ),
        args: [
            potlibc_engine.full_path(),
            potlibc_engine_fail_lock.full_path(),
        ],
        depends: [potlibc_engine, potlibc_engine_fail_lock],
        timeout: 120,
    )
    test(
        'potlibc_engine_exports',
        python3_runner,
        args: [
            engine_exports_script,
            potlibc_engine.full_path(),
            engine_exports_expected,
        ],
        depends: potlibc_engine,
        timeout: 30,
    )
endif

lj_lib = static_library(
    'rgpot_lj_eval',
    files('lj_eval.cpp'),
    include_directories: [engines_inc, rgpot_inc],
    # LJPot.hpp reaches fmt through base_types.hpp unless pure_lib.
    dependencies: [dl_dep] + (has_fmt ? [fmt_dep] : []),
    cpp_args: _args,
    pic: true,
)
lj_dep = declare_dependency(
    link_with: lj_lib,
    include_directories: [engines_inc, rgpot_inc],
    dependencies: has_fmt ? [fmt_dep] : [],
)

# pot_spec.cpp reaches fmt through base_types.hpp unless pure_lib.
pot_spec_deps = has_fmt ? [fmt_dep] : []
pot_spec_args = _args
if _rpc_stack
    pot_spec_deps += [ptlrpc_hdr_dep]
    pot_spec_args += ['-DRGPOT_POT_SPEC_HAS_CPMD=1']
endif
pot_spec_lib = static_library(
    'rgpot_pot_spec',
    files('pot_spec.cpp'),
    include_directories: [engines_inc, rgpot_inc],
    dependencies: pot_spec_deps,
    cpp_args: pot_spec_args,
    pic: true,
)
pot_spec_dep = declare_dependency(
    link_with: pot_spec_lib,
    dependencies: pot_spec_deps,
    include_directories: [engines_inc, rgpot_inc],
    compile_args: pot_spec_args,
)
if mdi_enabled
    rgpot_mdi = executable(
        'rgpot-mdi',
        files('rgpot_mdi.cpp'),
        dependencies: [
            pot_spec_dep,
            rgpot_rust_lib,
            thread_dep,
            dl_dep,
            engines_rgpot,
        ],
        cpp_args: _args,
        install: not meson.is_subproject(),
    )
endif
rgpot_eval = executable(
    'rgpot-eval',
    files('rgpot_eval.cpp'),
    dependencies: [pot_spec_dep, engines_rgpot],
    cpp_args: _args,
)
if mdi_enabled and get_option('with_tests')
    if has_pymdi
        test(
            'mdi_concrete_potentials',
            find_program('python3'),
            args: [
                files('mdi_roundtrip.py'),
                rgpot_mdi,
                rgpot_eval,
            ],
            timeout: 120,
        )
    else
        message('MDI TCP test skipped: pymdi is not installed')
    endif
endif

# i-PI driver client: answers an i-PI server with any pot_spec potential.
if has_ipi
    rgpot_ipi_driver = executable(
        'rgpot-ipi-driver',
        files('rgpot_ipi_driver.cpp'),
        dependencies: [pot_spec_dep, engines_rgpot],
        cpp_args: _args,
        install: not meson.is_subproject(),
    )
    # Not a meson test: the timings in docs/orgmode/howto/ipi.org come from it.
    executable(
        'rgpot-ipi-bench',
        files('ipi/bench_ipi_driver.cpp'),
        dependencies: [pot_spec_dep, thread_dep, engines_rgpot],
        cpp_args: _args,
        install: false,
    )
endif

rgpot_lj = executable(
    'rgpot-lj',
    files('rgpot_lj.cpp'),
    dependencies: [lj_dep, engines_rgpot],
    include_directories: rgpot_inc,
    cpp_args: _args,
    install: not meson.is_subproject(),
)
test(
    'lj_eval',
    executable(
        'test_lj_eval',
        files('test_lj_eval.cpp'),
        dependencies: [lj_dep, engines_rgpot],
        cpp_args: _args,
    ),
    timeout: 60,
)

lj_engine = shared_library(
    'lj_engine',
    [files('lj_engine.cpp'), engine_abi_stamp_source],
    dependencies: [lj_dep, engines_rgpot],
    include_directories: rgpot_inc,
    cpp_args: _args + ['-DRGPOT_ENGINE_BUILD'],
    install: not meson.is_subproject(),
)
standalone_engine_plugins = [lj_engine]
if potlibc_dep.found()
    standalone_engine_plugins += potlibc_engine
endif
test(
    'lj_engine',
    executable(
        'test_engine',
        files('test_engine.cpp'),
        dependencies: [lj_dep, engines_rgpot],
        cpp_args: _args,
    ),
    args: [lj_engine.full_path()],
    timeout: 60,
)

capi_threads = executable(
    'capi_threads',
    files('capi_threads.c'),
    include_directories: rgpot_inc,
    dependencies: [thread_dep, engines_rgpot],
    link_language: 'cpp',
)
test('capi_threads', capi_threads, kwargs: _rt_kw['capi_threads'])
# A second binary built only with the thread sanitizer. b_sanitize is
# already on every compile line, and -fsanitize=thread cannot be combined
# with address or undefined. It is a test binary, so it follows
# with_tests: gcc-toolset-14 in the manylinux images that build the wheel
# accepts -fsanitize=thread when compiling but has no libtsan to link.
# ThreadSanitizer has no Windows runtime.
if (
    host_system != 'windows'
    and get_option('with_tests')
    and cppc.has_argument('-fsanitize=thread')
    and get_option('b_sanitize') == 'none'
)
    capi_tsan = executable(
        'capi_threads_tsan',
        files('capi_threads.c'),
        include_directories: rgpot_inc,
        c_args: ['-fsanitize=thread'],
        link_args: ['-fsanitize=thread'],
        dependencies: [thread_dep, engines_rgpot],
        link_language: 'cpp',
    )
    # GCC's runtime (14 through 16) aborts with "unexpected memory
    # mapping" when the kernel randomises mmap with more than 28 bits, as
    # Ubuntu 24.04 and later do. LLVM's runtime re-executes itself with
    # ASLR off; setarch -R does that for GCC's. A sandbox may refuse the
    # personality call, so the wrapper is used only when it works here.
    tsan_cmd = capi_tsan
    tsan_args = []
    setarch = find_program('setarch', required: false)
    if (
        not meson.is_cross_build()
        and setarch.found()
        and run_command(setarch, '-R', 'true', check: false).returncode() == 0
    )
        tsan_cmd = setarch
        tsan_args = ['-R', capi_tsan]
    endif
    test(
        'capi_threads_tsan',
        tsan_cmd,
        args: tsan_args,
        kwargs: _rt_kw['capi_threads_tsan'],
    )
endif

python3_engines = find_program('python3', required: false)

# rgpot-ipi-driver under ASE's SocketIOCalculator, and under i-PI when
# i-pi is on PATH. The module imports numpy and ASE, so the test is
# registered only when the python3 that runs it has both.
if has_ipi and python3_engines.found()
    ipi_pytest = run_command(
        python3_engines,
        '-c',
        'import pytest, numpy, ase',
        check: false,
    )
    if ipi_pytest.returncode() == 0
        ipi_env = environment()
        ipi_env.set('RGPOT_IPI_DRIVER', rgpot_ipi_driver.full_path())
        if get_option('with_python')
            ipi_env.set(
                'RGPOT_PYTHON_BUILD',
                meson.project_build_root() / 'python',
            )
        endif
        test(
            'ipi_driver_ase',
            python3_engines,
            args: [
                '-m',
                'pytest',
                '-q',
                '-rs',
                '-p',
                'no:cacheprovider',
                '--rootdir',
                engines_here / 'ipi',
                files('ipi/test_ipi_driver.py'),
            ],
            env: ipi_env,
            workdir: meson.current_build_dir(),
            timeout: 600,
        )
    else
        message('i-PI driver tests skipped: needs pytest, numpy and ase')
    endif
endif

lammps_opt = get_option('with_lammps')
if lammps_opt.allowed() and python3_engines.found()
    # Drop '' and '.'. Meson runs this from engines/, and engines/lammps
    # would otherwise import as the LAMMPS module.
    lammps_py = run_command(
        python3_engines,
        '-c',
        'import sys; sys.path = [p for p in sys.path if p not in ("", ".")]; import lammps',
        check: false,
    )
    if lammps_py.returncode() != 0 and lammps_opt.enabled()
        error(
            'with_lammps is enabled but the lammps Python module was not found',
        )
    endif
    if lammps_py.returncode() == 0
        lammps_args = [rgpot_lj.full_path()]
        lammps_src = get_option('lammps_src')
        if lammps_src != ''
            # -DLAMMPS_SMALLBIG matches the default LAMMPS integer layout.
            # The plugin is loaded by an lmp built from the same headers.
            # A conda liblammps is a different build and is not that loader.
            lammps_dep = cppc.find_library('lammps', required: false)
            lammps_exe = get_option('lammps_exe')
            lammps_inc = lammps_src.startswith(project_root + '/') ? fs.relative_to(
                lammps_src,
                engines_here,
            ) : lammps_src
            # A serial lmp is built against src/STUBS/mpi.h and its library
            # exports those MPI functions. The plugin's gather and scatter
            # must use the same header, or they bind to the system MPI.
            lammps_stubs = false
            if lammps_exe != ''
                lammps_probe_env = environment()
                lammps_lib = fs.parent(lammps_exe) / 'liblammps.so'
                if fs.exists(lammps_lib)
                    lammps_probe_env.set('LD_PRELOAD', lammps_lib)
                endif
                lammps_help = run_command(
                    lammps_exe,
                    '-h',
                    check: false,
                    env: lammps_probe_env,
                )
                lammps_stubs = lammps_help.stdout().contains('LAMMPS MPI STUBS')
            endif
            pair_plugin = shared_library(
                'pair_rgpot',
                files('lammps/pair_rgpot.cpp'),
                dependencies: [lj_dep, pot_spec_dep, engines_rgpot] + (
                    lammps_dep.found() ? [lammps_dep] : []
                ),
                include_directories: include_directories(
                    (lammps_stubs ? [lammps_inc / 'STUBS'] : []) + [lammps_inc],
                ),
                # The MPI C++ bindings of Open MPI 4 would leave
                # MPI::Win::Free undefined; LAMMPS skips them too.
                cpp_args: _args + [
                    '-DLAMMPS_SMALLBIG',
                    '-DOMPI_SKIP_MPICXX',
                    '-DMPICH_SKIP_MPICXX',
                ],
                override_options: ['b_lundef=false'],
            )
            if lammps_exe != ''
                lammps_args += [pair_plugin.full_path(), lammps_exe]
                lammps_pytest = run_command(
                    python3_engines,
                    '-c',
                    'import pytest',
                    check: false,
                )
                if lammps_pytest.returncode() == 0
                    pair_env = environment()
                    pair_env.set('RGPOT_PAIR_PLUGIN', pair_plugin.full_path())
                    pair_env.set('RGPOT_LMP', lammps_exe)
                    pair_env.set('RGPOT_EVAL', rgpot_eval.full_path())
                    pair_env.set('RGPOT_LJ_ENGINE', lj_engine.full_path())
                    if has_eam_table
                        pair_env.set(
                            'RGPOT_EAM_ALLOY',
                            project_root / 'CppCore/tests/data/eam_table/CuNi.eam.alloy',
                        )
                    endif
                    lammps_mpirun = find_program(
                        'mpirun',
                        'mpiexec',
                        required: false,
                    )
                    # Each rank of a serial lmp is a world of one, so the
                    # MPI cases skip instead of comparing rank counts.
                    if lammps_mpirun.found() and not lammps_stubs
                        pair_env.set('RGPOT_MPIRUN', lammps_mpirun.full_path())
                    endif
                    test(
                        'lammps_pair_rgpot',
                        python3_engines,
                        args: [
                            '-m',
                            'pytest',
                            '-q',
                            '-p',
                            'no:cacheprovider',
                            '--rootdir',
                            engines_here / 'lammps',
                            files('lammps/test_pair_rgpot.py'),
                        ],
                        env: pair_env,
                        workdir: meson.current_build_dir(),
                        timeout: 900,
                    )
                endif
            endif
        elif lammps_opt.enabled()
            error(
                'with_lammps is enabled but lammps_src is empty, so pair_style rgpot cannot be built',
            )
        endif
        test(
            'lammps_lj',
            python3_engines,
            args: [files('lammps/run_lammps.py')] + lammps_args,
            timeout: 180,
        )
    endif
endif

gromacs_opt = get_option('with_gromacs')
plumed_opt = get_option('with_plumed')
have_plumed = false
if plumed_opt.allowed()
    plumed_dep = dependency('plumed', required: false)
    if not plumed_dep.found() and plumed_opt.enabled()
        error('with_plumed is enabled but pkg-config plumed was not found')
    endif
    if plumed_dep.found()
        plumed_kernel = cppc.find_library('plumedKernel', required: false)
        plumed_link = [plumed_dep]
        if plumed_kernel.found()
            plumed_link += [plumed_kernel]
        endif
        have_plumed = true
        plumed_lib = shared_library(
            'RgpotPlumed',
            files('plumed/RgpotColvar.cpp'),
            dependencies: plumed_link + [pot_spec_dep, engines_rgpot],
            include_directories: [engines_inc, rgpot_inc],
            cpp_args: _args,
            name_prefix: 'lib',
            install: not meson.is_subproject(),
        )
        plumed_exe = find_program('plumed', required: false)
        if python3_engines.found() and plumed_exe.found()
            plumed_script = files('plumed/run_plumed.py')
            plumed_eam = has_eam_table ? [
                project_root / 'CppCore/tests/data/eam_table/CuNi.eam.alloy',
            ] : []
            plumed_python = get_option('with_python') ? [
                meson.project_build_root() / 'python',
            ] : []
            plumed_cases = {
                'plumed_driver': ['dimer', rgpot_lj.full_path()],
                'plumed_debug_forces': ['forces'],
                'plumed_box_derivatives': ['box'],
                'plumed_units': ['units'],
                'plumed_api': ['api'],
            }
            foreach name, head : plumed_cases
                tail = [plumed_lib.full_path()]
                if name in ['plumed_debug_forces', 'plumed_box_derivatives']
                    tail += plumed_eam
                elif name == 'plumed_api'
                    tail += [rgpot.full_path()] + plumed_python
                endif
                test(
                    name,
                    python3_engines,
                    args: [plumed_script] + head + tail,
                    workdir: meson.current_build_dir(),
                    timeout: 600,
                )
            endforeach
        else
            message(
                'PLUMED tests skipped: needs python3 and the plumed program on PATH',
            )
        endif
    else
        message('PLUMED tests skipped: pkg-config plumed was not found')
    endif
endif

if gromacs_opt.allowed()
    gmx = find_program('gmx', required: gromacs_opt.enabled())
    if gmx.found() and have_plumed and python3_engines.found()
        test(
            'gromacs_mdrun',
            python3_engines,
            args: [
                files('gromacs/run_gromacs.py'),
                rgpot_lj.full_path(),
                plumed_lib.full_path(),
                meson.current_build_dir() / 'gromacs-work',
                rgpot_eval.full_path(),
            ],
            timeout: 300,
        )
    elif gromacs_opt.enabled()
        error('with_gromacs is enabled but gmx or the PLUMED module is missing')
    else
        message('GROMACS test skipped: needs gmx, the PLUMED module and python3')
    endif
endif

if get_option('with_gromacs_dropin').allowed()
    subdir('gromacs')
endif

# Configures a parent project around this tree, so it needs meson on PATH
# and only runs from the top-level build. The parent takes CC and CXX
# without a cross file, so it is not run for a Windows host.
meson_prog = find_program('meson', required: false)
if (
    get_option('with_tests')
    and host_system != 'windows'
    and not meson.is_subproject()
    and python3_engines.found()
    and meson_prog.found()
)
    nested_meson_env = environment()
    _nested_cc = cc.cmd_array()
    _nested_cxx = cppc.cmd_array()
    nested_meson_env.set('CC', _nested_cc[_nested_cc.length() - 1])
    nested_meson_env.set('CXX', _nested_cxx[_nested_cxx.length() - 1])
    _nested_bins = [
        _nested_cc[0],
        _nested_cc[_nested_cc.length() - 1],
        _nested_cxx[0],
        _nested_cxx[_nested_cxx.length() - 1],
        python3_engines.full_path(),
    ]
    if has_fortran
        _nested_fc = fc.cmd_array()
        nested_meson_env.set('FC', _nested_fc[_nested_fc.length() - 1])
        _nested_bins += [_nested_fc[0], _nested_fc[_nested_fc.length() - 1]]
    endif
    _nested_ninja = find_program('ninja', 'ninja-build', required: false)
    if _nested_ninja.found()
        nested_meson_env.set('NINJA', _nested_ninja.full_path())
        _nested_bins += [_nested_ninja.full_path()]
    endif
    foreach _tool_name : ['pkg-config', 'pkgconf', 'cmake', 'cargo', 'rustc']
        _tool_prog = find_program(_tool_name, required: false)
        if _tool_prog.found()
            _nested_bins += [_tool_prog.full_path()]
        endif
    endforeach
    _nested_dirs = []
    foreach _bin : _nested_bins
        if fs.is_absolute(_bin)
            _dir = fs.parent(_bin)
            if not (_dir in _nested_dirs)
                _nested_dirs += [_dir]
            endif
        endif
    endforeach
    if _nested_dirs.length() > 0
        nested_meson_env.prepend('PATH', _nested_dirs)
    endif
    test(
        'optional_engine_install',
        python3_engines,
        args: [
            files('test_optional_engine_install.py'),
            meson.project_source_root(),
            meson_prog.full_path(),
        ],
        env: nested_meson_env,
        timeout: 300,
    )
    test(
        'eigen_subproject_fallback',
        python3_engines,
        args: [
            files('test_eigen_subproject_fallback.py'),
            meson.project_source_root(),
            meson_prog.full_path(),
        ],
        env: nested_meson_env,
        timeout: 300,
    )
    _consumer_pc_args = [
        '--pc',
        meson.project_build_root() / 'meson-private' / 'rgpot.pc',
    ]
    if get_option('default_library') != 'shared'
        _consumer_pc_args += ['--pc-static']
    endif
    test(
        'consumer_no_openmp',
        python3_engines,
        args: [
            files('test_consumer_no_openmp.py'),
        ] + _consumer_pc_args + [
            meson.project_source_root(),
            meson_prog.full_path(),
        ],
        env: nested_meson_env,
        timeout: 1200,
        is_parallel: false,
    )
    # A static librgpot links the runtime its compiler selects. The parent
    # build's own compiler is covered above, so this runs only when Clang is
    # another one: GCC's libgomp does not satisfy objects Clang compiled.
    _clang_cc = find_program('clang', required: false)
    _clang_cxx = find_program('clang++', required: false)
    if (
        _clang_cc.found()
        and _clang_cxx.found()
        and cppc.get_id() != 'clang'
    )
        test(
            'consumer_no_openmp_clang',
            python3_engines,
            args: [
                files('test_consumer_no_openmp.py'),
                '--compilers',
                _clang_cc.full_path(),
                _clang_cxx.full_path(),
                meson.project_source_root(),
                meson_prog.full_path(),
            ],
            env: nested_meson_env,
            timeout: 1200,
            is_parallel: false,
        )
    endif
    _cmake_prog = find_program('cmake', required: false)
    if _cmake_prog.found() and _nested_ninja.found()
        test(
            'cmake_consumer_openmp',
            python3_engines,
            args: [
                files('test_cmake_consumer_openmp.py'),
                meson.project_source_root(),
                _cmake_prog.full_path(),
            ],
            env: nested_meson_env,
            timeout: 1200,
            is_parallel: false,
        )
    endif
    test(
        'gromacs_include_prefix',
        python3_engines,
        args: [
            files('test_gromacs_include_prefix.py'),
            meson.project_source_root(),
            meson_prog.full_path(),
        ],
        env: nested_meson_env,
        timeout: 300,
    )
    test(
        'gromacs_missing_include',
        python3_engines,
        args: [
            files('test_gromacs_missing_include.py'),
            meson.project_source_root(),
            meson_prog.full_path(),
        ],
        env: nested_meson_env,
        timeout: 300,
    )
    if has_fortran
        test(
            'fortran_sw_module_order',
            python3_engines,
            args: [
                files('test_fortran_sw_module_order.py'),
                meson.project_source_root(),
                meson_prog.full_path(),
            ],
            env: nested_meson_env,
            timeout: 600,
        )
    endif
endif

# abi_symbols reads the C ABI from the library's dynamic symbols, and
# librgpot.dll exports only rgpot_library_version, so on Windows there is
# no C ABI in the export table to compare against the baseline.
if host_system != 'windows'
    abi_script = find_program('../scripts/abi_symbols.sh')
    abi_baselines = files('../scripts/abi_baseline.txt')
    if has_fast_nlist
        abi_baselines += files('../scripts/abi_baseline_fast.txt')
    endif
    test(
        'abi_symbols',
        abi_script,
        args: [rgpot.full_path(), abi_baselines],
        timeout: 30,
    )
    # abi_symbols above reads only the kind of librgpot this build makes. A
    # small fixture built as a shared object, a lib*.a archive and a *.lib
    # archive runs the script's other nm modes in every build. The two
    # should_fail tests keep the check strict: a shared object hides hidden
    # names, and an archive still reports a missing one.
    if get_option('with_tests')
        abi_fixture_src = files('abi_fixture.c')
        abi_fixture_baseline = files('abi_fixture_baseline.txt')
        abi_fixture_so = shared_library('rgpot_abi_fixture', abi_fixture_src)
        abi_fixture_a = static_library('rgpot_abi_fixture_a', abi_fixture_src)
        abi_fixture_lib = static_library(
            'rgpot_abi_fixture_lib',
            abi_fixture_src,
            name_prefix: '',
            name_suffix: 'lib',
        )
        foreach _kind : [
            ['shared', abi_fixture_so],
            ['archive_a', abi_fixture_a],
            ['archive_lib', abi_fixture_lib],
        ]
            test(
                'abi_symbols_fixture_' + _kind[0],
                abi_script,
                args: [_kind[1].full_path(), abi_fixture_baseline],
                depends: _kind[1],
                timeout: 30,
            )
        endforeach
        test(
            'abi_symbols_fixture_shared_hidden',
            abi_script,
            args: [abi_fixture_so.full_path(), files('abi_fixture_hidden.txt')],
            depends: abi_fixture_so,
            timeout: 30,
            kwargs: _rt_kw['abi_symbols_fixture_shared_hidden'],
        )
        test(
            'abi_symbols_fixture_archive_missing',
            abi_script,
            args: [abi_fixture_a.full_path(), abi_baselines],
            depends: abi_fixture_a,
            timeout: 30,
            kwargs: _rt_kw['abi_symbols_fixture_archive_missing'],
        )
    endif
endif

openmm_opt = get_option('with_openmm')
if openmm_opt.allowed() and python3_engines.found()
    # Drop '' and '.'. Meson runs this from engines/, and engines/openmm
    # would otherwise import as the OpenMM module.
    openmm_py = run_command(
        python3_engines,
        '-c',
        'import sys; sys.path = [p for p in sys.path if p not in ("", ".")]; import openmm',
        check: false,
    )
    if openmm_py.returncode() != 0 and openmm_opt.enabled()
        error(
            'with_openmm is enabled but the openmm Python module was not found',
        )
    endif
    if openmm_py.returncode() == 0
        test(
            'openmm_lj',
            python3_engines,
            args: [files('openmm/run_openmm.py'), rgpot_lj.full_path()],
            timeout: 120,
        )
    endif
endif

native_lj_engine = shared_library(
    'native_lj_engine',
    [files('native_lj_engine.cpp'), engine_abi_stamp_source],
    dependencies: [lj_dep, engines_rgpot],
    include_directories: rgpot_inc,
    cpp_args: _args + ['-DRGPOT_ENGINE_BUILD'],
    install: not meson.is_subproject(),
)
standalone_engine_plugins += native_lj_engine
external_pot_test = executable(
    'test_external_potentials',
    files('test_external_potentials.cpp'),
    dependencies: [lj_dep, engines_rgpot],
    include_directories: rgpot_inc,
    cpp_args: _args,
)
external_env = environment()
if openmm_opt.allowed() and python3_engines.found()
    _openmm_where = run_command(
        python3_engines,
        '-c',
        'import openmm.version; print(openmm.version.openmm_library_path)',
        check: false,
    )
    if _openmm_where.returncode() == 0
        openmm_libdir = _openmm_where.stdout().strip()
        openmm_incdir = fs.parent(openmm_libdir) / 'include'
        openmm_inc = include_directories(
            openmm_incdir.startswith(project_root + '/') ? fs.relative_to(
                openmm_incdir,
                engines_here,
            ) : openmm_incdir,
        )
        openmm_lib = cppc.find_library(
            'OpenMM',
            dirs: [openmm_libdir],
            required: openmm_opt.enabled(),
        )
        if openmm_lib.found() and cppc.has_header(
            'openmm/Platform.h',
            include_directories: openmm_inc,
        )
            openmm_session_lib = static_library(
                'openmm_session',
                files('../CppCore/rgpot/ExternalPot/OpenMM.cc'),
                dependencies: [openmm_lib],
                include_directories: [rgpot_inc, openmm_inc],
                cpp_args: _args,
                pic: true,
                install: false,
            )
            openmm_engine = shared_library(
                'openmm_engine',
                [
                    files('openmm/openmm_engine.cpp'),
                    engine_abi_stamp_source,
                ],
                link_with: openmm_session_lib,
                dependencies: [openmm_lib, engines_rgpot],
                include_directories: [rgpot_inc, openmm_inc],
                cpp_args: _args + ['-DRGPOT_ENGINE_BUILD'],
                build_rpath: openmm_libdir,
                install_rpath: openmm_libdir,
                install: not meson.is_subproject(),
                # OpenMM platforms keep CUDA and OpenCL contexts and loaded
                # plugins alive; a dlclose of the engine must not unmap them.
                link_args: cppc.get_supported_link_arguments(
                    ['-Wl,-z,nodelete'],
                ),
            )
            standalone_engine_plugins += openmm_engine
            external_env.set('RGPOT_OPENMM_ENGINE', openmm_engine.full_path())
            external_env.set('OPENMM_PLUGIN_DIR', openmm_libdir / 'plugins')
            openmm_session_test = executable(
                'test_openmm_session',
                files('openmm/test_openmm_session.cpp'),
                link_with: openmm_session_lib,
                dependencies: [openmm_lib, engines_rgpot],
                include_directories: [rgpot_inc, openmm_inc],
                cpp_args: _args,
                build_rpath: openmm_libdir,
            )
            test(
                'openmm_session',
                openmm_session_test,
                env: external_env,
                timeout: 120,
            )
            openmm_session_review_test = executable(
                'test_openmm_session_review',
                files('openmm/test_openmm_session_review.cpp'),
                link_with: openmm_session_lib,
                dependencies: [openmm_lib, engines_rgpot],
                include_directories: [rgpot_inc, openmm_inc],
                cpp_args: _args,
                build_rpath: openmm_libdir,
            )
            openmm_analytic_test = executable(
                'test_openmm_analytic',
                files('openmm/test_openmm_analytic.cpp'),
                dependencies: [openmm_lib, engines_rgpot],
                include_directories: [
                    rgpot_inc,
                    openmm_inc,
                    include_directories('..'),
                ],
                cpp_args: _args,
                build_rpath: openmm_libdir,
            )
            test(
                'openmm_analytic',
                openmm_analytic_test,
                args: [openmm_engine.full_path()],
                env: external_env,
                timeout: 300,
            )
            openmm_abi_test = executable(
                'test_openmm_abi',
                files('openmm/test_openmm_abi.cpp'),
                dependencies: [openmm_lib, engines_rgpot],
                include_directories: [rgpot_inc, openmm_inc],
                cpp_args: _args,
                build_rpath: openmm_libdir,
            )
            test(
                'openmm_abi',
                openmm_abi_test,
                args: [openmm_engine.full_path()],
                env: external_env,
                timeout: 120,
            )
            foreach review_case : [
                'equipartition',
                'mean-force',
                'wall-energy',
                'metric-seed',
                'seed-chunking',
                'wall-multi-bounce',
                'off-plane-force',
                'openmm-constraints',
            ]
                test(
                    'openmm_review_' + review_case,
                    openmm_session_review_test,
                    args: [review_case],
                    env: external_env,
                    timeout: 180,
                )
            endforeach
        endif
    endif
endif
test(
    'external_potentials',
    external_pot_test,
    args: [lj_engine.full_path(), native_lj_engine.full_path()],
    env: external_env,
    timeout: 120,
)
standalone_engine_paths = []
foreach engine : standalone_engine_plugins
    standalone_engine_paths += engine.full_path()
endforeach
test(
    'engine_abi_stamps',
    executable(
        'test_engine_abi_stamps',
        files('test_engine_abi_stamps.cpp'),
        include_directories: rgpot_inc,
        dependencies: dl_dep,
        cpp_args: _args,
    ),
    args: standalone_engine_paths,
    depends: standalone_engine_plugins,
    timeout: 30,
)
# The Rust loader opens the Lennard-Jones plugin built above. native_lj_engine
# is not a candidate: it serves the native-unit GROMACS and OpenMM constructors
# through rgpot_md_native_force and exports no rgpot_engine_force. cargo gets
# its own target directory, apart from the one with_rust_core builds in. A
# build with the address, thread or memory sanitizer leaves the plugin needing
# a runtime the cargo test binary does not carry, so that build skips this test.
# cargo builds the loader for the build machine, so it cannot open a DLL.
engine_test_cargo = find_program('cargo', required: false)
engine_test_sanitized = false
foreach sanitizer : ['address', 'thread', 'memory']
    if sanitizer in get_option('b_sanitize')
        engine_test_sanitized = true
    endif
endforeach
if (
    engine_test_cargo.found()
    and not engine_test_sanitized
    and host_system != 'windows'
)
    test(
        'engine_abi_stamp_rust_loader',
        engine_test_cargo,
        args: [
            'test',
            '--locked',
            '--manifest-path',
            project_root / 'rgpot-core' / 'Cargo.toml',
            '--target-dir',
            meson.project_build_root() / 'cargo-engine-test-target',
            '--features',
            'engine',
            '--test',
            'engine',
            'configured_external_engine_passes_the_abi_check',
            '--',
            '--ignored',
            '--exact',
        ],
        env: {'RGPOT_TEST_ENGINE': lj_engine.full_path()},
        depends: lj_engine,
        timeout: 900,
    )
    if potlibc_dep.found()
        test(
            'potlibc_engine_rust_loader',
            engine_test_cargo,
            args: [
                'test',
                '--locked',
                '--manifest-path',
                project_root / 'rgpot-core' / 'Cargo.toml',
                '--target-dir',
                meson.project_build_root() / 'cargo-engine-test-target',
                '--features',
                'engine',
                '--test',
                'engine',
                '--',
                '--ignored',
                '--exact',
                'potlibc_engine_evaluates_through_the_rust_loader',
                'potlibc_lock_failure_is_a_latched_lock_error_through_the_rust_loader',
            ],
            env: {
                'RGPOT_TEST_POTLIBC_ENGINE': potlibc_engine.full_path(),
                'RGPOT_TEST_POTLIBC_ENGINE_FAIL_LOCK': potlibc_engine_fail_lock.full_path(),
            },
            depends: [potlibc_engine, potlibc_engine_fail_lock],
            timeout: 900,
        )
    endif
endif

# Optional header for gromacs_unit_reference. A conda or pip prefix has an
# include directory. A uv isolated build does not
# (…/uv/builds-v0/…/include), and include_directories() errors on a missing
# directory, so the directory is checked before it is added.
# dependency('gromacs', required: false) is the pkg-config probe. A missing
# GROMACS takes that not-found path and configure continues. Requesting
# GROMACS still errors above when gmx or the PLUMED module is missing.
gromacs_inc = ''
if python3_engines.found()
    _gmx_inc = run_command(
        python3_engines,
        '-c',
        'import pathlib, sys; print(pathlib.Path(sys.prefix) / "include")',
        check: false,
    )
    if _gmx_inc.returncode() == 0
        gromacs_inc = _gmx_inc.stdout().strip()
    endif
endif
gromacs_dep = dependency('gromacs', required: false)
gromacs_unit_inc = [engines_inc, rgpot_inc]
gromacs_unit_deps = []
gromacs_unit_ready = false
# A prefix inside the consumer source tree makes include_directories()
# reject the path before the header check. Skip it; the test is optional.
if (
    gromacs_inc != ''
    and fs.is_dir(gromacs_inc)
    and not gromacs_inc.startswith(meson.global_source_root())
)
    gromacs_headers = include_directories(
        gromacs_inc.startswith(project_root + '/') ? fs.relative_to(
            gromacs_inc,
            engines_here,
        ) : gromacs_inc,
    )
    if cppc.has_header(
        'gromacs/math/units.h',
        include_directories: gromacs_headers,
    )
        gromacs_unit_inc += [gromacs_headers]
        gromacs_unit_ready = true
    endif
endif
if (
    not gromacs_unit_ready
    and gromacs_dep.found()
    and cppc.has_header('gromacs/math/units.h', dependencies: gromacs_dep)
)
    gromacs_unit_deps += [gromacs_dep]
    gromacs_unit_ready = true
endif
if gromacs_unit_ready
    test(
        'gromacs_unit_reference',
        executable(
            'test_gromacs_units',
            files('test_gromacs_units.cpp'),
            include_directories: gromacs_unit_inc,
            dependencies: gromacs_unit_deps,
            cpp_args: _args,
        ),
        timeout: 30,
    )
endif
