project(
    'rgpot',
    'cpp',
    version: '0.1.0',
    default_options: ['warning_level=2', 'cpp_std=c++20'],
)
# IMPORTANT!! warning_level=3 passes -fimplicit-none
# Many of the older Fortran codes need implicit typing

host_system = host_machine.system()

# Add C++ compiler options
_cppc_early = meson.get_compiler('cpp')
if _cppc_early.get_id() == 'msvc' or _cppc_early.get_id() == 'clang-cl'
    cpp_args = ['/W3']
else
    cpp_args = ['-Wall', '-Wextra', '-Wpedantic']
endif
_args = [cpp_args]  # Extra arguments
_deps = []  # Dependencies
_linkto = []  # All the sub-libraries (potentials)
_incdirs = []  # All the includes
_rgpot_srcs = []

# Add conditionals
if host_system == 'darwin'
    _args += ['-DOSX=TRUE']
    cpp_args += ['-faligned-allocation']
endif

# Languages
add_languages('c', required: true)
if not get_option('with_rpc_client_only')
    # Check for Fortran compiler instead of requiring it globally
    has_fortran = add_languages('fortran', required: false)
else
    has_fortran = false
endif

rpc_enabled = get_option('with_rpc') or get_option('with_rpc_client_only')

if get_option('with_rpc_client_only')
    _args += ['-DRGPOT_RPC_CLIENT_ONLY=TRUE']
endif

cc = meson.get_compiler('c')
cppc = meson.get_compiler('cpp')

# Dependencies
# libm for Unix systems
m_dep = cppc.find_library('m', required: false)
_deps += m_dep
# For clang on Unix (MSVC links the C++ runtime automatically)
if host_system != 'windows'
    _deps += [declare_dependency(link_args: '-lstdc++')]
endif

if get_option('with_examples') and not get_option('with_rpc_client_only')
    # Eigen
    if not get_option('with_eigen')
        eigen_dep = dependency('eigen3')
        if not eigen_dep.found()
            eigen_dep = dependency(
                'eigen3',
                subproject: subproject('eigen'),
                required: true,
            )
        endif
        _deps += eigen_dep
    endif
    # fmt
    fmt_dep = dependency('fmt', required: true)
    _deps += fmt_dep
    if not get_option('with_xtensor')
        # These are added if not done later
        _deps += dependency('xtensor')
        _deps += dependency('xtensor-blas')
        _args += ['-DWITH_XTENSOR=TRUE']
    endif
endif

if get_option('with_xtensor')
    _deps += dependency('xtensor')
    _deps += dependency('xtensor-blas')
    _args += ['-DWITH_XTENSOR=TRUE']
endif

if get_option('with_eigen')
    eigen_dep = dependency('eigen3')
    if not eigen_dep.found()
        eigen_dep = dependency('eigen3', subproject: subproject('eigen'))
    endif
    _deps += eigen_dep
endif


if not get_option('pure_lib')
    _args += ['-DNOT_PURE_LIB=TRUE']
    subproject('fmt')
    fmt_dep = dependency('fmt', required: true)
    _deps += fmt_dep
endif

# --------------------- Subprojects
if has_fortran
    fortcuh2_proj = subproject('fortcuh2')
    fortcuh2_dep = fortcuh2_proj.get_variable('fortcuh2_dep')
    _deps += [fortcuh2_dep]
    _args += ['-DRGPOT_HAS_FORTRAN=TRUE']
endif

# --------------------- Rust Core (rgpot-core)
#
# Build the Rust crate via Cargo and link the resulting static library.
# The crate also generates include/rgpot.h via cbindgen.

if get_option('with_rust_core')
    cargo = find_program('cargo', required: true)

    # Determine Cargo build profile and library name
    if get_option('buildtype') == 'release'
        _cargo_profile_dir = 'release'
        _cargo_args = ['build', '--release']
    else
        _cargo_profile_dir = 'debug'
        _cargo_args = ['build']
    endif

    # Add feature flags for the Rust crate
    _cargo_features = []
    if rpc_enabled
        _cargo_features += ['rpc']
        _args += ['-DRGPOT_HAS_RPC=TRUE']
    endif
    if get_option('with_cache')
        _cargo_features += ['cache']
    endif

    if _cargo_features.length() > 0
        _cargo_args += ['--features', ','.join(_cargo_features)]
    endif

    _rgpot_core_dir = meson.current_source_dir() / 'rgpot-core'

    # Build the Rust crate; produces librgpot_core.a and include/rgpot.h
    rgpot_rust_target = custom_target(
        'rgpot_rust_core',
        command: [
            cargo,
            _cargo_args,
            '--manifest-path',
            _rgpot_core_dir / 'Cargo.toml',
            '--target-dir',
            meson.current_build_dir() / 'cargo-target',
        ],
        output: ['librgpot_core.a'],
        build_by_default: true,
        console: true,
    )

    # Link the Rust static library
    _rust_target_dir = meson.current_build_dir() / 'cargo-target' / _cargo_profile_dir
    rgpot_rust_lib = cppc.find_library(
        'rgpot_core',
        dirs: [_rust_target_dir],
        required: true,
        static: true,
    )

    # Thread and dl libraries needed by the Rust runtime
    thread_dep = dependency('threads')
    dl_dep = cppc.find_library('dl', required: false)

    _deps += [rgpot_rust_lib, thread_dep, dl_dep]

    # Include paths: generated C header + C++ RAII wrappers
    _incdirs += [
        include_directories('rgpot-core' / 'include'),  # rgpot.h (cbindgen)
        include_directories('include'),  # rgpot/*.hpp (C++ RAII)
    ]
endif

# --------------------- Projects
_incdirs += [include_directories('CppCore')]
subdir('CppCore')
rgpot_dep = declare_dependency(
    include_directories: _incdirs,
    link_with: _linkto,
    compile_args: _args,
    dependencies: _deps,
)
