_deps += dependency('rocksdb')
# Wrap installed
_deps += subproject('xxhash').get_variable('xxhash_dep')

pot_core = static_library(
    'pot_core',
    sources: [files('rgpot/PotHelpers.cc', 'rgpot/PotentialCache.cc')],
    dependencies: _deps,
    include_directories: _incdirs,
    install: false,
)

_deps += declare_dependency(link_with: pot_core)

# Implemented potentials
subdir('rgpot/LennardJones')  # 'lennard_jones'
subdir('rgpot/CuH2')  # 'cuh2'

potentials = [lennard_jones, cuh2]

_linkto += potentials

# Create library
potlib = library(
    'pot',
    dependencies: _deps,
    cpp_args: _args,
    include_directories: _incdirs,
    link_with: _linkto,
    install: true,
)

# ------------------------ Examples

if get_option('with_examples')
    example_array = [  #
        ['EigenCallLJPot', 'ljpot_call_eig', 'eigen_call_ljpot.cc', ''],
        ['EigenCallCuH2Pot', 'cuh2_call_eig', 'eigen_call_cuh2.cc', ''],
        ['XtCallCuH2Pot', 'cuh2_call_xt', 'xt_call_cuh2.cc', ''],
        ['CallCuH2Pot', 'cuh2_call', 'call_cuh2.cc', ''],
    ]
    # Not really a test but the best way to make tiny programs
    foreach example : example_array
        test(
            example.get(0),
            executable(
                example.get(1),
                sources: ['examples/' + example.get(2)],
                dependencies: _deps,
                link_with: _linkto,
                cpp_args: _args,
            ),
            workdir: meson.source_root() + example.get(3),
        )
    endforeach
endif

# ------------------------ Tests

if get_option('with_tests')
    test_deps = _deps
    test_deps += dependency(
        'Catch2',
        method: 'cmake',
        modules: ['Catch2::Catch2WithMain'],
    )
    test_args = _args
    test_args += ['-DPOTTEST']
    test_array = [  #
        ['CuH2test', 'cuh2_test', 'CuH2PotTest.cc', ''],
        ['CacheTest', 'cache_test', 'CacheTest.cc', ''],
        # ['CuH2 Utils Test', 'cuh2_util_test', 'cuh2UtilsTest.cc', ''],
    ]
    foreach test : test_array
        test(
            test.get(0),
            executable(
                test.get(1),
                sources: ['tests/' + test.get(2)],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
            ),
            workdir: meson.source_root() + test.get(3),
        )
    endforeach
endif
