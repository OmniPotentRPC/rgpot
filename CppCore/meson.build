if get_option('with_cache') and not get_option('with_rpc_client_only')
    _deps += dependency('rocksdb')
    _deps += subproject('xxhash', default_options: ['cli=false']).get_variable(
        'xxhash_dep',
    )
    _args += ['-DRGPOT_HAS_CACHE=TRUE']
    _rgpot_srcs += files('rgpot/PotentialCache.cc')
endif

_rgpot_srcs += files('rgpot/PotHelpers.cc')
_rgpot_srcs += files('rgpot/units.cc')

# Cap'n Proto schema + NWChem/CPMD frontends need libcapnp (ptlrpc). Only pull
# them in when RPC is on (server or client-only); lean -Dwith_rpc=false builds
# (xtb/tblite/metatomic CI) must not require pkg-config 'capnp-rpc'.
_rpc_stack = rpc_enabled or get_option('with_rpc_client_only')
if _rpc_stack
    subdir('rgpot/rpc')
    # Pure dlopen consumers of split nwchemc / cpmdc; need generated schema only.
    subdir('rgpot/NWChemPot')  # 'nwchempot' frontend + no-op ABI stub
    subdir('rgpot/CPMDPot')    # 'cpmdpot' frontend + no-op ABI stub
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

    if _rpc_stack
        rgpotentials += [nwchempot]
        rgpotentials += [cpmdpot]
    endif

    _linkto += rgpotentials

    # Create library
    rgpot_deps = _deps
    if _rpc_stack
        rgpot_deps += [ptlrpc_dep]
    endif
    rgpot = library(
        'rgpot',
        dependencies: rgpot_deps,
        cpp_args: _args,
        include_directories: _incdirs,
        link_with: _linkto,
        install: not meson.is_subproject(),
    )
endif

# ------------------------ Server
if rpc_enabled and not get_option('with_rpc_client_only')
    server = executable(
        'potserv',
        ['rgpot/rpc/server.cpp', gen_rpc],
        link_with: [ptlrpc, _linkto],
        dependencies: _deps + [capnp_rpc_dep],
        cpp_args: _args,
        include_directories: _incdirs,
        link_language: 'cpp',
        install: not meson.is_subproject(),
    )
    # Multi-rank host for NWChemPot: host owns MPI; every rank calls force.
    # Optional when MPI C/C++ is available (not required for potserv).
    mpi_host_dep = dependency('mpi', language: 'cpp', required: false)
    if not mpi_host_dep.found()
        mpi_host_dep = dependency('mpi', language: 'c', required: false)
    endif
    if mpi_host_dep.found()
        nwchem_mpi_force_host = executable(
            'nwchem_mpi_force_host',
            'tools/nwchem_mpi_force_host.cc',
            link_with: [ptlrpc, _linkto],
            dependencies: _deps + [capnp_rpc_dep, mpi_host_dep],
            cpp_args: _args,
            include_directories: _incdirs,
            link_language: 'cpp',
            install: false,
        )
    endif
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
            [
                'MetatomicTest',
                'metatomic_test',
                'MetatomicPotTest.cc',
                '/CppCore/tests',
            ],
        ]
    endif
    # NWChem/CPMD frontend tests (need ptlrpc + frontends from _rpc_stack)
    if _rpc_stack and not get_option('with_rpc_client_only')
        nwchemc_fake_engine = shared_library(
            'nwchemc_fake_engine',
            sources: ['tests/nwchemc_fake_engine.c'],
            include_directories: include_directories('rgpot/NWChemPot'),
            install: false,
        )
        nwchemc_fake_env = environment()
        nwchemc_fake_env.set(
            'RGPOT_NWCHEM_ENGINE',
            nwchemc_fake_engine.full_path(),
        )
        test(
            'NWChemPotTest',
            executable(
                'nwchem_pot_test',
                sources: ['tests/NWChemPotTest.cc'],
                dependencies: test_deps + [ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            env: nwchemc_fake_env,
            depends: nwchemc_fake_engine,
            workdir: meson.source_root(),
            suite: 'nwchem',
        )
        test(
            'NWChemPotMessageAbiTest',
            executable(
                'nwchem_pot_message_abi_test',
                sources: ['tests/NWChemPotMessageAbiTest.cc'],
                dependencies: test_deps + [ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            env: nwchemc_fake_env,
            depends: nwchemc_fake_engine,
            workdir: meson.source_root(),
            suite: 'nwchem',
        )
        test(
            'test_nwchem_abi',
            executable(
                'test_nwchem_abi',
                sources: ['tests/test_nwchem_abi_main.c'],
                dependencies: [nwchem_abi_stub_dep],
                include_directories: include_directories('rgpot/NWChemPot'),
            ),
            suite: 'nwchem',
        )
        cpmdc_fake_engine = shared_library(
            'cpmdc_fake_engine',
            sources: ['tests/cpmdc_fake_engine.cc'],
            dependencies: [ptlrpc_dep],
            include_directories: _incdirs + ['.'] + include_directories(
                'rgpot/CPMDPot',
            ),
            install: false,
        )
        cpmdc_fake_env = environment()
        cpmdc_fake_env.set(
            'RGPOT_CPMD_ENGINE',
            cpmdc_fake_engine.full_path(),
        )
        test(
            'CPMDPotMessageAbiTest',
            executable(
                'cpmd_pot_message_abi_test',
                sources: ['tests/CPMDPotMessageAbiTest.cc'],
                dependencies: test_deps + [ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            env: cpmdc_fake_env,
            depends: cpmdc_fake_engine,
            workdir: meson.source_root(),
            suite: 'cpmd',
        )
        test(
            'ProfileLoaderTest',
            executable(
                'profile_loader_test',
                sources: ['tests/ProfileLoaderTest.cc'],
                dependencies: test_deps + [ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            env: cpmdc_fake_env,
            depends: cpmdc_fake_engine,
            workdir: meson.source_root(),
            suite: 'abi',
        )
        test(
            'test_cpmd_abi',
            executable(
                'test_cpmd_abi',
                sources: ['tests/test_cpmd_abi_main.c'],
                dependencies: [cpmd_abi_stub_dep],
                include_directories: include_directories('rgpot/CPMDPot'),
            ),
            suite: 'cpmd',
        )
        test(
            'CPMDDlopenContract',
            find_program('python3'),
            args: [
                meson.current_source_dir() / 'tests/test_cpmd_dlopen_contract.py',
                meson.source_root(),
            ],
            suite: 'cpmd',
        )
        test(
            'NWChemDlopenContract',
            find_program('python3'),
            args: [
                meson.current_source_dir() / 'tests/test_nwchem_dlopen_contract.py',
                meson.source_root(),
            ],
            suite: 'nwchem',
        )
        test(
            'RPCSchemaSync',
            find_program('python3'),
            args: [
                meson.current_source_dir() / 'tests/test_rpc_schema_sync.py',
                meson.source_root(),
            ],
            suite: 'rpc',
        )
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
                link_language: 'cpp',
            ),
            workdir: meson.source_root() + test.get(3),
        )
    endforeach
endif
