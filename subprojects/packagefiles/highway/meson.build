project(
    'highway',
    'cpp',
    version: '1.2.0',
    default_options: ['cpp_std=c++17', 'warning_level=0'],
)

# Targets compiled into libhwy and into every rgpot kernel TU. The numeric
# mask matches CppCore/rgpot/simd/hwy_disabled.h (Highway 1.2 bit positions).
# AVX3_DL, AVX3_SPR and the reserved AVX10.2 bit are not in the set.
cpp = meson.get_compiler('cpp')
fp = cpp.get_supported_arguments(['-ffp-contract=off', '/fp:precise'])
disabled = '-DHWY_DISABLED_TARGETS=0xC0F800024041090'

libhwy = static_library(
    'hwy',
    files(
        'hwy/abort.cc',
        'hwy/aligned_allocator.cc',
        'hwy/nanobenchmark.cc',
        'hwy/per_target.cc',
        'hwy/print.cc',
        'hwy/targets.cc',
        'hwy/timer.cc',
    ),
    include_directories: include_directories('.'),
    cpp_args: fp + [disabled],
    pic: true,
)

libhwy_dep = declare_dependency(
    link_with: libhwy,
    include_directories: include_directories('.'),
    compile_args: fp + [disabled],
)
