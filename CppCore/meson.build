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
_rgpot_srcs += files('rgpot/units.cc')
_rgpot_srcs += files('rgpot/BatchEvaluator.cc')

if has_plugins
    _incdirs += [include_directories('../include')]  # rgpot/plugin.h
    subdir('rgpot/plugin')
endif

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

    if has_xtb
        subdir('rgpot/XTBPot')  # 'xtbpot'
        rgpotentials += [xtbpot]
    endif

    if has_tblite
        subdir('rgpot/TBLitePot')  # 'tblitepot'
        rgpotentials += [tblitepot]
    endif

    if has_metatomic
        subdir('rgpot/MetatomicPot')  # 'metatomicpot'
        rgpotentials += [metatomicpot]
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
# Each example only builds when its required dependencies are present.

if get_option('with_examples') and not get_option('with_rpc_client_only')
    example_array = []
    if has_fortran
        example_array += [
            ['CallCuH2Pot', 'cuh2_call', 'call_cuh2.cc', ''],
        ]
    endif
    if has_eigen
        example_array += [
            ['EigenCallLJPot', 'ljpot_call_eig', 'eigen_call_ljpot.cc', ''],
        ]
        if has_fortran
            example_array += [
                ['EigenCallCuH2Pot', 'cuh2_call_eig', 'eigen_call_cuh2.cc', ''],
            ]
        endif
    endif
    if has_xtensor and has_fortran
        example_array += [
            ['XtCallCuH2Pot', 'cuh2_call_xt', 'xt_call_cuh2.cc', ''],
        ]
    endif
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

# ------------------------ External plugin .so targets (for testing)

if has_plugins and get_option('with_tests')
    plugin_inc = include_directories('../include')

    # C: null plugin
    rgpot_null_plugin = shared_library(
        'rgpot_null',
        'examples/plugins/null_plugin.c',
        include_directories: plugin_inc,
        install: false,
    )

    # C: Morse plugin
    rgpot_morse_plugin = shared_library(
        'rgpot_morse',
        'examples/plugins/morse_plugin.c',
        include_directories: plugin_inc,
        dependencies: m_dep,
        install: false,
    )

    # Fortran + C: Coulomb plugin
    if has_fortran
        rgpot_coulomb_plugin = shared_library(
            'rgpot_coulomb',
            ['examples/plugins/coulomb_plugin.f90',
             'examples/plugins/coulomb_plugin_wrapper.c'],
            include_directories: plugin_inc,
            install: false,
        )
    endif
endif

# ------------------------ Plugin self-tests (Python, Julia, Rust)
# These run the plugin examples as standalone scripts to verify they work.

if has_plugins and get_option('with_tests')
    python3 = find_program('python3', required: false)
    julia = find_program('julia', required: false)

    _plugin_examples_dir = meson.current_source_dir() / 'examples' / 'plugins'

    if python3.found()
        test(
            'PythonHarmonicPlugin',
            python3,
            args: [_plugin_examples_dir / 'harmonic_plugin.py'],
            timeout: 30,
        )
        test(
            'PythonJaxLJPlugin',
            python3,
            args: [_plugin_examples_dir / 'jax_lj_plugin.py'],
            timeout: 120,
        )
        test(
            'PythonTorchNNPlugin',
            python3,
            args: [_plugin_examples_dir / 'torch_nn_plugin.py'],
            timeout: 120,
        )
    endif

    if julia.found()
        test(
            'JuliaSpringPlugin',
            julia,
            args: [_plugin_examples_dir / 'spring_plugin.jl'],
            timeout: 120,
        )
    endif

    # Rust Morse plugin: build with cargo, test by loading .so
    cargo = find_program('cargo', required: false)
    if cargo.found()
        _morse_rust_dir = meson.current_source_dir() / 'examples' / 'plugins' / 'morse_rust'
        _morse_rust_target = meson.current_build_dir() / 'morse_rust_target'
        _morse_rust_build = custom_target(
            'morse_rust_plugin',
            command: [
                cargo, 'build', '--release',
                '--manifest-path', _morse_rust_dir / 'Cargo.toml',
                '--target-dir', _morse_rust_target,
            ],
            output: ['libmorse_plugin.so'],
            build_by_default: true,
            console: true,
        )
    endif
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
    test_array = [
        ['UnitsTest', 'units_test', 'UnitsTest.cc', ''],
        ['BatchTest', 'batch_test', 'BatchTest.cc', ''],
    ]
    if has_fortran
        test_array += [
            ['CuH2test', 'cuh2_test', 'CuH2PotTest.cc', ''],
        ]
        if has_xtensor
            test_array += [
                ['CuH2 Utils Test', 'cuh2_util_test', 'cuh2UtilsTest.cc', ''],
            ]
        endif
    endif
    if get_option('with_cache')
        test_array += [  #
            ['CacheTest', 'cache_test', 'CacheTest.cc', ''],
            ['InvarianceTest', 'invariance_test', 'InvarianceTest.cc', ''],
        ]
    endif
    if has_plugins
        test_array += [
            ['PluginTest', 'plugin_test', 'PluginTest.cc', ''],
        ]
        # External plugin integration tests (need the .so files built above)
        _ext_plugin_test = executable(
            'external_plugin_test',
            sources: ['tests/ExternalPluginTest.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
        )
        _ext_plugin_env = environment()
        _ext_plugin_env.set(
            'RGPOT_TEST_PLUGIN_DIR',
            meson.current_build_dir(),
        )
        _ext_test_depends = [rgpot_null_plugin, rgpot_morse_plugin]
        if has_fortran
            _ext_test_depends += [rgpot_coulomb_plugin]
        endif
        test(
            'ExternalPluginTest',
            _ext_plugin_test,
            env: _ext_plugin_env,
            depends: _ext_test_depends,
        )
    endif
    if has_xtb
        test_array += [
            ['XTBtest', 'xtb_test', 'XTBPotTest.cc', ''],
        ]
    endif
    if has_tblite
        test_array += [
            ['TBLitetest', 'tblite_test', 'TBLitePotTest.cc', ''],
        ]
    endif
    if has_metatomic
        test_array += [
            ['MetatomicTest', 'metatomic_test', 'MetatomicPotTest.cc', '/CppCore/tests'],
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
