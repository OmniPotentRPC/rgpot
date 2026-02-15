if get_option('with_cache') and not get_option('with_rpc_client_only')
    _deps += dependency('rocksdb')
    # Wrap installed
    _deps += subproject('xxhash', default_options: ['cli=false']).get_variable(
        'xxhash_dep',
    )
    _args += ['-DRGPOT_HAS_CACHE=TRUE']
    _rgpot_srcs += files('rgpot/PotentialCache.cc')
endif

_rgpot_srcs += files('rgpot/PotHelpers.cc')

# ------------------------ Main library
if not get_option('with_rpc_client_only')
    rgpot_core = static_library(
        'rgpot_core',
        sources: _rgpot_srcs,
        dependencies: _deps,
        include_directories: _incdirs,
        install: false,
    )

    _deps += declare_dependency(link_with: rgpot_core)

    # Implemented rgpotentials
    subdir('rgpot/LennardJones')  # 'lennard_jones'
    rgpotentials = [lennard_jones]

    if has_fortran
        subdir('rgpot/CuH2')  # 'cuh2'
        rgpotentials += [cuh2]
    endif

    _linkto += rgpotentials

    # Create library
    rgpot = library(
        'rgpot',
        dependencies: _deps,
        cpp_args: _args,
        include_directories: _incdirs,
        link_with: _linkto,
        install: not meson.is_subproject(),
    )
endif

# ------------------------ Server
if rpc_enabled
    subdir('rgpot/rpc')
endif

# ------------------------ Examples

if get_option('with_examples') and not get_option('with_rpc_client_only')
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
    test_args += ['-DRGPOTTEST']
    test_array = [  #
        # ['CuH2 Utils Test', 'cuh2_util_test', 'cuh2UtilsTest.cc', ''],
]
    if has_fortran
        test_array += [  #
            ['CuH2test', 'cuh2_test', 'CuH2PotTest.cc', ''],
        ]
    endif
    if get_option('with_cache')
        test_array += [  #
            ['CacheTest', 'cache_test', 'CacheTest.cc', ''],
            ['InvarianceTest', 'invariance_test', 'InvarianceTest.cc', ''],
        ]
    endif
    if rpc_enabled
        test_deps += dependency('capnp-rpc')
        test_deps += ptlrpc_dep
        test_array += [['RPCTest', 'test_rpc', 'CapnpAdapterTest.cc', '']]
        if get_option('with_rpc_client_only')
            test_deps += rgpot_bridge_dep
            test_array += [
                ['RPCClientTest', 'test_rpc_client', 'BridgeStressTest.cc', ''],
            ]
        endif
    endif
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
