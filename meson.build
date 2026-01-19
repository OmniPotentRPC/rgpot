project(
    'pot',
    'cpp',
    version: '0.1.0',
    default_options: ['warning_level=2', 'cpp_std=c++20'],
)
# IMPORTANT!! warning_level=3 passes -fimplicit-none
# Many of the older Fortran codes need implicit typing

host_system = host_machine.system()

# Add C++ compiler options
cpp_args = ['-Wall', '-Wextra', '-Wpedantic']
_args = [cpp_args]  # Extra arguments
_deps = []  # Dependencies
_linkto = []  # All the sub-libraries (potentials)
_incdirs = []  # All the includes

# Add conditionals
if host_system == 'darwin'
    _args += ['-DOSX=TRUE']
    cpp_args += ['-faligned-allocation']
endif

# Languages
add_languages('c', required: true)
add_languages('fortran', required: true)
cc = meson.get_compiler('c')
cppc = meson.get_compiler('cpp')

# Dependencies
# libm for Unix systems
m_dep = cppc.find_library('m', required: false)
_deps += m_dep
# For clang
_deps += [declare_dependency(link_args: '-lstdc++')]

if get_option('with_examples')
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
fortcuh2_proj = subproject('fortcuh2')
fortcuh2_dep = fortcuh2_proj.get_variable('fortcuh2_dep')
_deps += [fortcuh2_dep]

# --------------------- Projects
_incdirs += [include_directories('CppCore')]
subdir('CppCore')
potlib_dep = declare_dependency(
    include_directories: _incdirs,
    link_with: _linkto,
    compile_args: _args,
    dependencies: _deps,
)
