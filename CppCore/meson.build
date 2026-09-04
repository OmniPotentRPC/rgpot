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
        pic: true,
        install: false,
    )

    # Snapshot before the core link-dep: the umbrella library link_wholes
    # rgpot_core explicitly and must not also pull it through link_with.
    _deps_sans_core = _deps
    _deps += declare_dependency(link_with: rgpot_core)

    # Implemented rgpotentials
    subdir('rgpot/LennardJones')  # 'lennard_jones': LJPot + LJClusterPot
    rgpotentials = [lennard_jones]

    subdir('rgpot/Morse')  # 'morse'
    rgpotentials += [morse]

    subdir('rgpot/ZBL')  # 'zbl'
    rgpotentials += [zbl]

    subdir('rgpot/CuH2')  # 'cuh2': slab-geometry helpers
    rgpotentials += [cuh2]

    if has_fortran
        subdir('rgpot/fortran')  # Fortran 2018 pot kernels + C entry points
        rgpotentials += [fortran_pots, fortran_kernels]
        # The kernel archive's symbols (module symbols and the bind(c)
        # entries) stay local to librgpot: only the C++ faces above are
        # part of the library's interface.
        # vesin's Fortran interface is an implementation detail of the
        # kernels and is hidden with them.
        _fortran_hide_args = cppc.get_supported_link_arguments(
            [
                '-Wl,--exclude-libs,librgpot_fortran_kernels.a',
                '-Wl,--exclude-libs,libvesin_fortran.a',
            ],
        )
    endif

    if has_xtb
        subdir('rgpot/XTBPot')  # 'xtbpot' + engine + dlopen frontend
        rgpotentials += [xtbpot]
        _linkto += [xtb_dlopen]
    endif

    if has_xckernel
        subdir('rgpot/XcKernel')  # in-process XcKernel, not a Potential
        rgpotentials += [xckernel]
    endif

    if has_tblite
        subdir('rgpot/TBLitePot')  # 'tblitepot'
        rgpotentials += [tblitepot]
    endif

    if has_dftd3
        subdir('rgpot/D3Pot')  # 'd3pot'
        rgpotentials += [d3pot]
    endif

    if has_dftd4
        subdir('rgpot/D4Pot')  # 'd4pot'
        rgpotentials += [d4pot]
    endif

    # MetatomicDlopen frontend always (no torch). Engine/fat pot when has_metatomic.
    subdir('rgpot/MetatomicPot')
    _linkto += [metatomic_dlopen]
    # Fat MetatomicPot NEEDs torch. Folding it into the umbrella makes
    # `import rgpot` (LJ / dlopen) require libtorch — the pip contract is
    # numpy-only until `rgpot[metatomic]`. The wheel still *builds* the
    # engine plugin (has_metatomic); it is dlopen'd, not linked.
    if has_metatomic and not get_option('with_python')
        rgpotentials += [metatomicpot]
    endif

    if _rpc_stack
        rgpotentials += [nwchempot]
        rgpotentials += [cpmdpot]
    endif

    _linkto += rgpotentials

    # Create the umbrella library: every pot and the core ship inside one
    # versioned SONAME. link_whole keeps all symbols even where nothing in
    # librgpot itself references them (consumers do).
    # vesin objects and the Fortran runtime enter here, once, rather than
    # through the convenience archives (see the vesin_hdr_dep note).
    rgpot_deps = _deps_sans_core + [vesin_dep] + fortran_rt_deps
    _rpc_whole = []
    if _rpc_stack
        # link_whole, not link_with: the RPC entry points are called by
        # consumers rather than by librgpot itself.
        rgpot_deps += [ptlrpc_hdr_dep]
        _rpc_whole += [ptlrpc]
    endif
    _rgpot_link_args = []
    if has_fortran
        _rgpot_link_args += _fortran_hide_args
    endif
    _rgpot_srcs = []
    _rgpot_cpp_args = _args
    if host_system == 'windows'
        # MSVC omits the import library when a DLL exports no symbols.
        _rgpot_srcs += files('rgpot/win_export.cc')
        _rgpot_cpp_args += [
            '-DMESON_PROJECT_VERSION="@0@"'.format(meson.project_version()),
        ]
    endif
    rgpot = library(
        'rgpot',
        _rgpot_srcs,
        dependencies: rgpot_deps,
        cpp_args: _rgpot_cpp_args,
        link_args: _rgpot_link_args,
        include_directories: _incdirs,
        link_whole: [rgpot_core] + _linkto + _rpc_whole,
        version: meson.project_version(),
        soversion: '3',
        install: not meson.is_subproject(),
    )

    # Subproject consumers (eOn's wrap fallback) take the umbrella through
    # this variable; pkg-config consumers get the same surface from
    # rgpot.pc.
    rgpot_dep = declare_dependency(
        link_with: rgpot,
        dependencies: rgpot_deps,
        include_directories: _incdirs,
    )

    # The Fortran kernels must stay invisible outside librgpot: an
    # exported legacy name is how two potentials used to bind to the same
    # symbol.
    if has_fortran and host_system == 'linux' and get_option('with_tests')
        test(
            'FortranSymbolAudit',
            find_program('../scripts/audit_fortran_symbols.sh'),
            args: [rgpot.full_path()],
            depends: [rgpot],
        )
    endif
endif

# Metatomic engine + dlopen frontend for client-only (eOn with_rgpot) consumers.
# Thin host links metatomic_dlopen_dep; optional libmetatomic_engine.so holds torch.
if get_option('with_rpc_client_only') and has_metatomic
    subdir('rgpot/MetatomicPot')
endif

# ------------------------ Server
if rpc_enabled and not get_option('with_rpc_client_only')
    server = executable(
        'potserv',
        ['rgpot/rpc/server.cpp'],
        link_with: [ptlrpc, _linkto],
        dependencies: _deps + [capnp_rpc_dep, ptlrpc_hdr_dep, vesin_dep] + fortran_rt_deps,
        cpp_args: _args,
        include_directories: _incdirs,
        link_language: 'cpp',
        install: not meson.is_subproject(),
    )
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
                dependencies: _deps + [vesin_dep] + fortran_rt_deps,
                link_with: _linkto,
                cpp_args: _args,
            ),
            workdir: meson.source_root() + example.get(3),
        )
    endforeach
endif

# ------------------------ Tests

if get_option('with_tests')
    test_deps = _deps + [vesin_dep] + fortran_rt_deps
    test_deps += dependency(
        'Catch2',
        method: 'cmake',
        modules: ['Catch2::Catch2WithMain'],
    )
    test_args = _args
    test_args += ['-DRGPOTTEST']
    # Client-only builds skip rgpot_core; UnitsTest still needs units.cc.
    if get_option('with_rpc_client_only')
        units_for_tests = static_library(
            'units_for_tests',
            sources: ['rgpot/units.cc'],
            include_directories: _incdirs,
            cpp_args: _args,
            install: false,
        )
        test_deps += declare_dependency(link_with: units_for_tests)
    endif
    test_array = [
        ['UnitsTest', 'units_test', 'UnitsTest.cc', ''],
    ]
    # Classical pots live in the rgpotentials static libs, which client-only
    # builds never produce.
    if not get_option('with_rpc_client_only') and has_fortran
        test_array += [
            ['FortranPotsTest', 'fortran_pots_test', 'FortranPotsTest.cc', ''],
        ]
    endif
    if not get_option('with_rpc_client_only')
        test_array += [
            ['MorsePotTest', 'morse_pot_test', 'MorsePotTest.cc', ''],
            [
                'LJClusterPotTest',
                'ljcluster_pot_test',
                'LJClusterPotTest.cc',
                '',
            ],
            ['ZBLPotTest', 'zbl_pot_test', 'ZBLPotTest.cc', ''],
        ]
    endif
    if has_fortran
        test_array += [
            ['CuH2test', 'cuh2_test', 'CuH2PotTest.cc', ''],
        ]
    endif
    # Slab helpers are pure C++; only their xtensor faces are conditional.
    if has_xtensor and not get_option('with_rpc_client_only')
        test_array += [
            ['CuH2 Utils Test', 'cuh2_util_test', 'cuh2UtilsTest.cc', ''],
        ]
    endif
    if get_option('with_cache')
        test_array += [  #
            ['CacheTest', 'cache_test', 'CacheTest.cc', ''],
            ['InvarianceTest', 'invariance_test', 'InvarianceTest.cc', ''],
        ]
    endif
    # XTB tests registered below with RGPOT_XTB_ENGINE for dlopen parity
    if has_tblite
        test_array += [
            ['TBLitetest', 'tblite_test', 'TBLitePotTest.cc', ''],
        ]
    endif
    if has_dftd3
        test_array += [
            ['D3test', 'd3_test', 'D3PotTest.cc', ''],
        ]
    endif
    if has_dftd4
        test_array += [
            ['D4test', 'd4_test', 'D4PotTest.cc', ''],
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
            test_deps += pot_bridge_dep
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

    if has_dftd3 or has_dftd4
        test(
            'DftdGoldens',
            executable(
                'dftd_golden_test',
                sources: ['tests/DftdGoldenTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            workdir: meson.source_root(),
            suite: 'dftd',
        )
        executable(
            'dump_dftd_goldens',
            sources: ['tests/dump_dftd_goldens.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            install: false,
        )
    endif

    if has_xckernel
        test(
            'XcKernelGoldens',
            executable(
                'xckernel_test',
                sources: ['tests/XcKernelTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            workdir: meson.source_root(),
            suite: 'xckernel',
        )
        executable(
            'dump_xckernel_goldens',
            sources: ['tests/dump_xckernel_goldens.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            install: false,
        )
    endif

    if has_xtb
        xtb_test_env = environment()
        xtb_test_env.set('RGPOT_XTB_ENGINE', xtb_engine.full_path())
        test(
            'XTBtest',
            executable(
                'xtb_test',
                sources: ['tests/XTBPotTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            env: xtb_test_env,
            depends: [xtb_engine],
            workdir: meson.source_root(),
            suite: 'xtb',
        )
        # Timing microbench: linked vs dlopen (same process OK for xTB).
        xtb_backend_bench = executable(
            'xtb_backend_bench',
            sources: ['tests/xtb_backend_bench.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            install: false,
        )
    endif

    # Pot microbenchmarks. Own binary, outside the default suite: the cases
    # carry the hidden Catch2 tag pair "[.][benchmark]" and only run when the
    # tag is requested, which `meson test --benchmark` does through args.
    # LJPot lives in rgpot_core, absent from client-only builds.
    if not get_option('with_rpc_client_only')
        pot_bench_args = test_args
        if has_fortran
            pot_bench_args += ['-DRGPOT_BENCH_CUH2']
        endif
        pot_bench = executable(
            'pot_bench',
            sources: ['tests/PotBench.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: pot_bench_args,
            link_with: _linkto,
            link_language: 'cpp',
            install: false,
        )
        benchmark(
            'pot_bench',
            pot_bench,
            args: ['[benchmark]'],
            workdir: meson.source_root(),
            timeout: 300,
        )
    endif
endif

# ------------------------ Install / pkg-config for eOn and other consumers
# eOn prefers dependency('rgpot') over subproject wrap. Packaging profile
# matching eOn wrap defaults (no torch/xtb at link time; engines dlopen):
#   -Dwith_rpc_client_only=true -Dpure_lib=true
#   -Dwith_tests=false -Dwith_examples=false
#   -Dwith_xtb=false -Dwith_metatomic=false
# See docs/eon_pkgconfig.md.
if not meson.is_subproject()
    # Headers: #include "rgpot/NWChemPot/..." with -I${prefix}/include
    install_subdir(
        'rgpot',
        install_dir: get_option('includedir'),
        exclude_directories: ['__pycache__'],
        exclude_files: [
            '*.o',
            '*.a',
            '*.so',
            '*.cc',
            '*.c',
            '*.cpp',
            '*.f90',
            '*.F',
            '*.py',
        ],
    )
    if _rpc_stack
        # Schema source next to generated header (rpc meson installs .h).
        install_data(
            'rgpot/rpc/Potentials.capnp',
            install_dir: get_option('includedir') / 'rgpot' / 'rpc',
        )
    endif

    # The .pc always ships: consumers discover whatever this build enabled
    # through the RGPOT_HAS_* cflags rather than by the file's absence.
    _pc_libs = []
    _pc_requires = []
    _pc_cflags = []
    if not get_option('with_rpc_client_only')
        _pc_libs += [rgpot]
        _pc_cflags += ['-DRGPOT_HAS_POTLIB=TRUE']
        if has_fortran
            _pc_cflags += ['-DRGPOT_HAS_FORTRAN_POTS=TRUE']
        endif
        if has_metatomic
            _pc_cflags += ['-DRGPOT_HAS_METATOMIC_POT=TRUE']
        endif
        if has_xtb
            _pc_cflags += ['-DRGPOT_HAS_XTB_POT=TRUE']
        endif
        if has_tblite
            _pc_cflags += ['-DRGPOT_HAS_TBLITE_POT=TRUE']
        endif
        if has_dftd3
            _pc_cflags += ['-DRGPOT_HAS_DFTD3_POT=TRUE']
        endif
        if has_dftd4
            _pc_cflags += ['-DRGPOT_HAS_DFTD4_POT=TRUE']
        endif
    endif
    if _rpc_stack
        _pc_libs += [nwchempot, cpmdpot]
        _pc_requires += ['capnp-rpc']
        _pc_cflags += [
            '-DRGPOT_HAS_NWCHEM_FRONTEND=TRUE',
            '-DRGPOT_HAS_CPMD_FRONTEND=TRUE',
        ]
    endif

    pkg = import('pkgconfig')
    pkg.generate(
        name: 'rgpot',
        description: 'rgpot potentials (librgpot umbrella) + QM frontends',
        url: 'https://github.com/OmniPotentRPC/rgpot',
        version: meson.project_version(),
        libraries: _pc_libs,
        requires: _pc_requires,
        subdirs: ['.'],
        extra_cflags: _pc_cflags,
        filebase: 'rgpot',
    )
endif
