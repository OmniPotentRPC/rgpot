# Fortran-backed potentials, ported to Fortran 2018.
#
# Sources are listed in dependency order for compilers without automatic
# module ordering; meson's Fortran scanner reorders as needed.
#
# Only the bind(c) entry points need to leave the enclosing librgpot, and
# C++ callers reach them from inside the same shared object, so the whole
# static library builds with hidden visibility: no bare Fortran symbol
# (or legacy COMMON block name) ever reaches the dynamic table.

_fpot_args = fc.get_supported_arguments(
    [
        '-std=f2018',
        '-fimplicit-none',
        '-Wall',
        '-Wno-unused-dummy-argument',
        '-fvisibility=hidden',
    ],
)

fortran_pots_sources = files(
    'rgpot_cuh2.f90',
    'rgpot_cuh2_capi.f90',
    'rgpot_eam_al.f90',
    'rgpot_eam_al_capi.f90',
    'rgpot_edip.f90',
    'rgpot_edip_capi.f90',
    'rgpot_fehe.f90',
    'rgpot_fehe_capi.f90',
    'rgpot_ferror.f90',
    'rgpot_kinds.f90',
    'rgpot_lenosky.f90',
    'rgpot_lenosky_capi.f90',
    'rgpot_neighbors.f90',
    'rgpot_sw.f90',
    'rgpot_sw_capi.f90',
    'rgpot_tersoff.f90',
    'rgpot_tersoff_capi.f90',
    'rgpot_water_h.f90',
    'rgpot_water_h_capi.f90',
)

# C++ faces over the bind(c) entries.
fortran_pots_cpp_sources = files('FortranPots.cc')

# Kernels only. gfortran ignores -fvisibility for module symbols, so the
# archive is kept separate and hidden at link time (see the umbrella
# library's --exclude-libs).
fortran_kernels = static_library(
    'rgpot_fortran_kernels',
    fortran_pots_sources,
    dependencies: [vesin_f_dep],
    fortran_args: _fpot_args,
    pic: true,
    install: false,
)

# The C++ faces stay visible: consumers construct these classes.
fortran_pots = static_library(
    'rgpot_fortran_pots',
    fortran_pots_cpp_sources,
    dependencies: _deps,
    cpp_args: _args,
    include_directories: _incdirs,
    link_with: fortran_kernels,
    pic: true,
    install: false,
)

fortran_pots_dep = declare_dependency(
    link_with: [fortran_pots, fortran_kernels],
    dependencies: [vesin_f_dep],
)

# Fortran-only test programs link the kernels directly. These are leaf
# executables, so they take the vesin objects themselves rather than the
# headers-only view the convenience archives use.
fortran_kernels_dep = declare_dependency(
    link_with: fortran_kernels,
    dependencies: [vesin_f_dep, vesin_dep],
)

# Plugin anneal dlopens as RGPOT_CUH2_LIBRARY. The umbrella librgpot hides
# Fortran symbols; this .so exports only the bind(c) entry.
_cuh2_plugin_args = fc.get_supported_arguments(
    [
        '-std=f2018',
        '-fimplicit-none',
        '-Wall',
        '-Wno-unused-dummy-argument',
    ],
)
rgpot_cuh2 = library(
    'rgpot_cuh2',
    [
        'rgpot_kinds.f90',
        'rgpot_ferror.f90',
        'rgpot_neighbors.f90',
        'rgpot_cuh2.f90',
        'rgpot_cuh2_capi.f90',
    ],
    dependencies: [vesin_f_dep, vesin_dep],
    fortran_args: _cuh2_plugin_args,
    install: not meson.is_subproject(),
    gnu_symbol_visibility: 'default',
)

# Physics checks that run without the C++ stack: a failure here points at
# the kernel, not the bindings.
if get_option('with_tests')
    test_sw_gather = executable(
        'test_sw_gather',
        'test_sw_gather.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranSWGather', test_sw_gather, timeout: 120)

    test_cuh2_gather = executable(
        'test_cuh2_gather',
        'test_cuh2.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranCuH2Gather', test_cuh2_gather, timeout: 120)

    test_eam_al = executable(
        'test_eam_al',
        'test_eam_al.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranEAMAlGather', test_eam_al, timeout: 120)

    test_water_h = executable(
        'test_water_h',
        'test_water_h.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranWaterHGather', test_water_h, timeout: 120)

    test_fehe = executable(
        'test_fehe',
        'test_fehe.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranFeHeGather', test_fehe, timeout: 120)

    test_tersoff = executable(
        'test_tersoff',
        'test_tersoff.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranTersoffGather', test_tersoff, timeout: 180)

    test_edip = executable(
        'test_edip',
        'test_edip.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranEDIPGather', test_edip, timeout: 180)

    test_lenosky = executable(
        'test_lenosky',
        'test_lenosky.f90',
        dependencies: [fortran_kernels_dep],
        fortran_args: _fpot_args,
        install: false,
    )
    test('FortranLenoskyGather', test_lenosky, timeout: 180)
endif
