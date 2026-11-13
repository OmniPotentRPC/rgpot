# Engine drivers. Each one is built when its library is found.
# with_* = enabled and a missing library is an error.

fs = import('fs')
engines_inc = include_directories('.')
rgpot_inc = include_directories('../CppCore', '../include')
dl_dep = cppc.find_library('dl', required: false)
thread_dep = dependency('threads')
lj_lib = static_library(
    'rgpot_lj_eval',
    files('lj_eval.cpp'),
    include_directories: [engines_inc, rgpot_inc],
    dependencies: dl_dep,
    cpp_args: _args,
    pic: true,
)
lj_dep = declare_dependency(
    link_with: lj_lib,
    include_directories: [engines_inc, rgpot_inc],
)

rgpot_lj = executable(
    'rgpot-lj',
    files('rgpot_lj.cpp'),
    dependencies: lj_dep,
    link_with: rgpot,
    include_directories: rgpot_inc,
    cpp_args: _args,
    install: true,
)
test('lj_eval', executable(
    'test_lj_eval',
    files('test_lj_eval.cpp'),
    dependencies: lj_dep,
    link_with: rgpot,
    cpp_args: _args,
), timeout: 60)

lj_engine = shared_library(
    'lj_engine',
    files('lj_engine.cpp'),
    dependencies: lj_dep,
    link_with: rgpot,
    include_directories: rgpot_inc,
    cpp_args: _args + ['-DRGPOT_ENGINE_BUILD'],
    install: true,
)
test(
    'lj_engine',
    executable(
        'test_engine',
        files('test_engine.cpp'),
        dependencies: lj_dep,
        link_with: rgpot,
        cpp_args: _args,
    ),
    args: [lj_engine.full_path()],
    timeout: 60,
)

capi_threads = executable(
    'capi_threads',
    files('capi_threads.c'),
    link_with: rgpot,
    include_directories: rgpot_inc,
    dependencies: thread_dep,
    link_language: 'cpp',
)
test('capi_threads', capi_threads, timeout: 60)
if cppc.has_argument('-fsanitize=thread') and 'thread' not in get_option('b_sanitize')
    capi_tsan = executable(
        'capi_threads_tsan',
        files('capi_threads.c'),
        link_with: rgpot,
        include_directories: rgpot_inc,
        c_args: ['-fsanitize=thread'],
        link_args: ['-fsanitize=thread'],
        dependencies: thread_dep,
        link_language: 'cpp',
    )
    test('capi_threads_tsan', capi_tsan, timeout: 120)
endif

python3_engines = find_program('python3', required: false)

lammps_opt = get_option('with_lammps')
if lammps_opt.allowed() and python3_engines.found()
    lammps_py = run_command(python3_engines, '-c', 'import lammps', check: false)
    if lammps_py.returncode() != 0 and lammps_opt.enabled()
        error('with_lammps is enabled but the lammps Python module was not found')
    endif
    if lammps_py.returncode() == 0
        lammps_args = [rgpot_lj.full_path()]
        lammps_src = get_option('lammps_src')
        if lammps_src != ''
            # -DLAMMPS_SMALLBIG matches the default LAMMPS integer layout.
            # The plugin is loaded by an lmp built from the same headers.
            # A conda liblammps is a different build and is not that loader.
            lammps_dep = cppc.find_library('lammps', required: false)
            pair_plugin = shared_library(
                'pair_rgpot',
                files('lammps/pair_rgpot.cpp'),
                dependencies: [lj_dep] + (lammps_dep.found() ? [lammps_dep] : []),
                link_with: rgpot,
                include_directories: include_directories(lammps_src),
                cpp_args: _args + ['-DLAMMPS_SMALLBIG'],
                override_options: ['b_lundef=false'],
            )
            lammps_exe = get_option('lammps_exe')
            if lammps_exe != ''
                lammps_args += [pair_plugin.full_path(), lammps_exe]
            endif
        elif lammps_opt.enabled()
            error('with_lammps is enabled but lammps_src is empty, so pair_style rgpot cannot be built')
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
            dependencies: plumed_link + [lj_dep],
            link_with: rgpot,
            include_directories: [engines_inc, rgpot_inc],
            cpp_args: _args,
            name_prefix: 'lib',
            install: true,
        )
        if python3_engines.found() and find_program('plumed', required: false).found()
            test(
                'plumed_driver',
                python3_engines,
                args: [files('plumed/run_plumed.py'), rgpot_lj.full_path(), plumed_lib.full_path()],
                timeout: 180,
            )
        endif
    endif
endif

if gromacs_opt.allowed()
    gmx = find_program('gmx', required: gromacs_opt.enabled())
    if gmx.found() and have_plumed and python3_engines.found()
        test(
            'gromacs_mdrun',
            python3_engines,
            args: [
                files('gromacs/run_gromacs.py'), rgpot_lj.full_path(),
                plumed_lib.full_path(), meson.current_build_dir() / 'gromacs-work',
            ],
            timeout: 180,
        )
    elif gromacs_opt.enabled()
        error('with_gromacs is enabled but gmx or the PLUMED module is missing')
    endif
endif

if python3_engines.found()
    test(
        'c_symbols',
        python3_engines,
        args: [files('../docs/check_c_symbols.py')],
        timeout: 30,
    )
endif

abi_script = find_program('../scripts/abi_symbols.sh')
test(
    'abi_symbols',
    abi_script,
    args: [rgpot.full_path(), files('../scripts/abi_baseline.txt')],
    timeout: 30,
)

openmm_opt = get_option('with_openmm')
if openmm_opt.allowed() and python3_engines.found()
    openmm_py = run_command(python3_engines, '-c', 'import openmm', check: false)
    if openmm_py.returncode() != 0 and openmm_opt.enabled()
        error('with_openmm is enabled but the openmm Python module was not found')
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
