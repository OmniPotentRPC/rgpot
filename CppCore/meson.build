if get_option('with_cache') and not get_option('with_rpc_client_only')
    _deps += dependency('rocksdb')
    _deps += subproject('xxhash', default_options: ['cli=false']).get_variable(
        'xxhash_dep',
    )
    _args += ['-DRGPOT_HAS_CACHE=TRUE']
    _rgpot_srcs += files('rgpot/PotentialCache.cc')
endif

# Calculator groups split MPI_COMM_WORLD and broadcast results between
# groups. librgpot itself never links MPI: the MPI side is librgpot_mpi,
# which CalculatorGroup.cc loads with dlopen when a host asks for groups
# (calculator_mpi_abi.h). Without it a process is one group of one rank.
_mpi_dep = dependency('mpi', language: 'cpp', required: get_option('with_mpi'))
_dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
_deps += _dl_dep
_rgpot_srcs += files('rgpot/CalculatorGroup.cc')
rgpot_mpi_lib = disabler()
rgpot_mpi_dep = disabler()
if _mpi_dep.found()
    rgpot_mpi_lib = shared_library(
        'rgpot_mpi',
        files('rgpot/CalculatorGroupMpi.cc'),
        dependencies: [_mpi_dep],
        include_directories: include_directories('.'),
        gnu_symbol_visibility: 'hidden',
        version: meson.project_version(),
        soversion: '3',
        install: not meson.is_subproject(),
    )
    # Hosts never need to link this: librgpot finds it next to itself.
    # Tests and hosts that want it loaded up front link it.
    rgpot_mpi_dep = declare_dependency(
        link_with: rgpot_mpi_lib,
        dependencies: [_mpi_dep],
        compile_args: ['-DRGPOT_HAS_MPI'],
    )
endif
_rgpot_srcs += files('rgpot/PotHelpers.cc')
_rgpot_srcs += files('rgpot/units.cc')
_rgpot_srcs += files('rgpot/abi/CxxLayout.cc')

# Cap'n Proto schema + NWChem/CPMD frontends need libcapnp (ptlrpc). Only pull
# them in when RPC is on (server or client-only); lean -Dwith_rpc=false builds
# (xtb/tblite/metatomic CI) must not require pkg-config 'capnp-rpc'.
_rpc_stack = rpc_enabled or get_option('with_rpc_client_only')
_metatomic_build_rpath = ''
if _rpc_stack
    subdir('rgpot/rpc')
    # Pure dlopen consumers of split nwchemc / cpmdc; need generated schema only.
    subdir('rgpot/NWChemPot')  # 'nwchempot' frontend + no-op ABI stub
    subdir('rgpot/CPMDPot')    # 'cpmdpot' frontend + no-op ABI stub
endif

# ------------------------ Main library
if not get_option('with_rpc_client_only')
    _deps += dependency('threads')
    _rgpot_srcs += files(
        'rgpot/plugin/PluginBridge.cc',
        'rgpot/plugin/PluginRegistry.cc',
        'rgpot/plugin/builtins/lj_plugin.cpp',
    )
    rgpot_core = static_library(
        'rgpot_core',
        sources: _rgpot_srcs,
        dependencies: _deps,
        cpp_args: _args,
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

    # Pure dlopen consumer of split mopacc (no Cap'n Proto, no OpenMOPAC embed).
    subdir('rgpot/MOPACPot')  # 'mopacpot'
    rgpotentials += [mopacpot]

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

    # XTBDlopen frontend always (no libxtb). Linked pot + engine when has_xtb.
    subdir('rgpot/XTBPot')
    _linkto += [xtb_dlopen]
    if has_xtb
        rgpotentials += [xtbpot]
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

    if has_expr
        subdir('rgpot/ExprPot')  # 'exprpot'
        rgpotentials += [exprpot]
    endif

    # MetatomicDlopen frontend always (no torch). Engine/fat pot when has_metatomic.
    subdir('rgpot/MetatomicPot')
    if has_metatomic
        _metatomic_build_rpath = _engine_build_rpath
    endif
    _linkto += [metatomic_dlopen]
    # Fat MetatomicPot NEEDs torch. Folding it into the umbrella makes
    # `import rgpot` (LJ / dlopen) require libtorch — the pip contract is
    # numpy-only until `rgpot[metatomic]`. The wheel still *builds* the
    # engine plugin (has_metatomic); it is dlopen'd, not linked.
    if has_metatomic and not get_option('with_python')
        rgpotentials += [metatomicpot]
        subdir('rgpot/UmaPot')
        rgpotentials += [uma]
    endif

    if _rpc_stack
        rgpotentials += [nwchempot]
        rgpotentials += [cpmdpot]
        if has_skala
            subdir('rgpot/SkalaPot')
            rgpotentials += [skala]
        endif
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
        build_rpath: _metatomic_build_rpath,
        include_directories: _incdirs,
        link_whole: [rgpot_core] + _linkto + _rpc_whole,
        version: meson.project_version(),
        soversion: '3',
        install: not meson.is_subproject(),
    )

    # Generic rgpot_engine_* plugins (engine_c_abi.h) for the backends that
    # have no tensor runtime: one shared table over PotentialBase
    # (rgpot/engine/EngineTable.cc) and a small backend file each. The UMA,
    # Metatomic and XTB engines add the same table to their own libraries.
    if _rpc_stack and host_system != 'windows'
        _engine_common = files('rgpot/engine/EngineTable.cc')
        _engine_args = _args + [
            '-DRGPOT_ENGINE_BUILD',
            '-DRGPOT_ENGINE_PROJECT_VERSION="@0@"'.format(meson.project_version()),
        ]
        _engine_deps = _deps_sans_core + [
            vesin_dep,
            capnp_rpc_dep,
            ptlrpc_dep,
        ] + fortran_rt_deps
        _classical_whole = [lennard_jones, morse, zbl]
        if has_fortran
            _classical_whole += [cuh2, fortran_pots, fortran_kernels]
        endif
        classical_engine = shared_library(
            'classical_engine',
            [_engine_common, files('rgpot/engine/ClassicalEngine.cc')],
            dependencies: _engine_deps,
            link_with: [rgpot_core],
            link_whole: _classical_whole,
            cpp_args: _engine_args,
            include_directories: _incdirs,
            gnu_symbol_visibility: 'default',
            install: not meson.is_subproject(),
        )
        mopac_engine = shared_library(
            'mopac_engine',
            [_engine_common, files('rgpot/engine/MopacEngine.cc')],
            dependencies: _engine_deps + [mopac_dl_dep],
            link_with: [rgpot_core],
            link_whole: [mopacpot],
            cpp_args: _engine_args,
            include_directories: _incdirs,
            gnu_symbol_visibility: 'default',
            install: not meson.is_subproject(),
        )
        nwchem_engine = shared_library(
            'nwchem_engine',
            [_engine_common, files('rgpot/engine/NWChemEngine.cc')],
            dependencies: _engine_deps + [nwchem_dl_dep],
            link_with: [rgpot_core],
            link_whole: [nwchempot],
            cpp_args: _engine_args,
            include_directories: _incdirs,
            gnu_symbol_visibility: 'default',
            install: not meson.is_subproject(),
        )
        if has_skala
            skala_engine = shared_library(
                'skala_engine',
                [_engine_common, files('rgpot/engine/SkalaEngine.cc')],
                dependencies: _engine_deps + [nwchem_dl_dep],
                link_with: [rgpot_core],
                link_whole: [nwchempot, skala],
                cpp_args: _engine_args,
                include_directories: _incdirs,
                gnu_symbol_visibility: 'default',
                install: not meson.is_subproject(),
            )
        endif
    endif

    # Subproject consumers (eOn's wrap fallback) take the umbrella through
    # this variable; pkg-config consumers get the same surface from
    # rgpot.pc.
    rgpot_dep = declare_dependency(
        link_with: rgpot,
        dependencies: rgpot_deps,
        include_directories: _incdirs,
        compile_args: _args,
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
    # librgpot must not pull libmpi into a host: the MPI side is dlopen'd.
    if get_option('with_tests') and host_system == 'linux'
        test(
            'CoreLinksNoMpi',
            find_program('../scripts/check_no_mpi_needed.sh'),
            args: [rgpot.full_path()],
            depends: [rgpot],
        )
    endif
    # nm/readelf ELF dump; same host gate as FortranSymbolAudit.
    if get_option('with_tests') and host_system != 'windows'
        test(
            'ExprLeptonSymbols',
            find_program('../scripts/check_lepton_symbols.sh'),
            args: [rgpot.full_path(), has_expr ? 'present' : 'absent'],
            depends: [rgpot],
            suite: 'expr',
        )
    endif
endif

# Client-only (eOn with_rgpot): thin dlopen frontends without fat link deps.
# XTB always; metatomic only when torch/metatomic are available to build the
# engine, whose optional libmetatomic_engine.so holds torch.
if get_option('with_rpc_client_only')
    subdir('rgpot/XTBPot')
    if has_metatomic
        subdir('rgpot/MetatomicPot')
        _metatomic_build_rpath = _engine_build_rpath
    endif
endif

# ------------------------ Server
if rpc_enabled and not get_option('with_rpc_client_only')
    server = executable(
        'potserv',
        ['rgpot/rpc/server.cpp'],
        link_with: [ptlrpc, _linkto],
        dependencies: _deps + [capnp_rpc_dep, ptlrpc_hdr_dep, vesin_dep] + _mta_engine_deps + fortran_rt_deps,
        cpp_args: _args,
        build_rpath: _metatomic_build_rpath,
        include_directories: _incdirs,
        link_language: 'cpp',
        install: not meson.is_subproject(),
    )
    if _mpi_dep.found()
        mpi_server = executable(
            'potserv_mpi',
            ['rgpot/rpc/server.cpp'],
            link_with: [ptlrpc, _linkto],
            dependencies: _deps + [
                capnp_rpc_dep,
                ptlrpc_hdr_dep,
                vesin_dep,
                _mpi_dep,
            ] + _mta_engine_deps + fortran_rt_deps,
            cpp_args: _args + ['-DRGPOT_POTSERV_MPI'],
            build_rpath: _metatomic_build_rpath,
            include_directories: _incdirs,
            link_language: 'cpp',
            install: not meson.is_subproject(),
        )
        nwchem_mpi_force_host = executable(
            'nwchem_mpi_force_host',
            ['tools/nwchem_mpi_force_host.cc'],
            link_with: [ptlrpc, _linkto],
            dependencies: _deps + [ptlrpc_hdr_dep, vesin_dep, _mpi_dep] + fortran_rt_deps,
            cpp_args: _args,
            build_rpath: _metatomic_build_rpath,
            include_directories: _incdirs,
            install: not meson.is_subproject(),
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
    if has_expr
        example_array += [
            ['CallExprPot', 'exprpot_call', 'call_exprpot.cc', ''],
        ]
    endif
    # Not a meson test: scripts/time_lj_rgpot_vs_pyeonclient.py drives this.
    executable(
        'time_lj_force',
        sources: ['examples/time_lj_force.cc'],
        dependencies: _deps + [vesin_dep] + fortran_rt_deps,
        link_with: _linkto,
        cpp_args: _args,
    )
    # Not a meson test: scripts/bench_pair_scaling.sh drives this.
    executable(
        'time_pair_scaling',
        sources: ['examples/time_pair_scaling.cc'],
        dependencies: _deps + [vesin_dep] + fortran_rt_deps,
        link_with: _linkto,
        cpp_args: _args,
    )
    if rpc_enabled
        # Not a meson test: needs a running potserv (see the file header).
        executable(
            'time_rpc_call',
            sources: ['examples/time_rpc_call.cc'],
            link_with: [pot_bridge],
            include_directories: _incdirs,
            cpp_args: _args,
        )
    endif
    if _mpi_dep.found()
        # Not a meson test: run under mpirun (see the file header).
        executable(
            'time_calculator_share',
            sources: ['examples/time_calculator_share.cc'],
            dependencies: _deps + [vesin_dep, rgpot_mpi_dep] + fortran_rt_deps,
            link_with: _linkto,
            cpp_args: _args,
        )
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
                build_rpath: _metatomic_build_rpath,
            ),
            workdir: meson.source_root() + example.get(3),
        )
    endforeach
endif

# ------------------------ Tests

if get_option('with_tests')
    test(
        'AotiPublicationTest',
        executable(
            'aoti_publication_test',
            'tests/AotiPublicationTest.cc',
            include_directories: ['.'],
            dependencies: dependency('threads'),
            override_options: ['cpp_std=c++20'],
        ),
    )
    test(
        'MolecularFrameTest',
        executable(
            'molecular_frame_test',
            'tests/MolecularFrameTest.cc',
            include_directories: ['.'],
            override_options: ['cpp_std=c++20'],
        ),
    )
    test_deps = _deps + [vesin_dep] + fortran_rt_deps
    if has_metatomic
        test_deps += _mta_engine_deps
    endif
    test_deps += dependency(
        'Catch2',
        method: 'cmake',
        modules: ['Catch2::Catch2WithMain'],
    )
    test_args = _args
    test_args += ['-DRGPOTTEST']
    if _mpi_dep.found()
        # Tests expect librgpot_mpi next to them (it builds in this
        # directory), whether or not they link MPI themselves.
        test_args += ['-DRGPOT_HAS_MPI']
    endif
    if not get_option('with_rpc_client_only')
        subdir('tests/plugin')
    endif
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
            ['PairListTest', 'pair_list_test', 'PairListTest.cc', ''],
            [
                'BatchEvaluatorTest',
                'batch_evaluator_test',
                'BatchEvaluatorTest.cc',
                '',
            ],
            [
                'CalculatorGroupTest',
                'calculator_group_test',
                'CalculatorGroupTest.cc',
                '',
            ],
            [
                'CalculatorAbortFlagTest',
                'calculator_abort_flag_test',
                'CalculatorAbortFlagTest.cc',
                '',
            ],
            [
                'LJClusterPotTest',
                'ljcluster_pot_test',
                'LJClusterPotTest.cc',
                '',
            ],
            ['ZBLPotTest', 'zbl_pot_test', 'ZBLPotTest.cc', ''],
            # The dlopen frontend is in librgpot whether or not with_xtb is.
            ['XTBDlopenTest', 'xtb_dlopen_test', 'XTBDlopenTest.cc', ''],
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
    if has_expr and not get_option('with_rpc_client_only')
        test(
            'ExprPotTest',
            executable(
                'expr_pot_test',
                sources: ['tests/ExprPotTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            workdir: meson.source_root(),
            suite: 'expr',
        )
        test(
            'ExprGoldens',
            executable(
                'expr_golden_test',
                sources: ['tests/ExprGoldenTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            workdir: meson.source_root(),
            suite: 'expr',
        )
        executable(
            'dump_expr_goldens',
            sources: ['tests/dump_expr_goldens.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            install: false,
        )
        if get_option('with_rust_core')
            test(
                'ExprEindirTest',
                executable(
                    'expr_eindir_test',
                    sources: ['tests/ExprEindirTest.cc'],
                    dependencies: test_deps,
                    include_directories: _incdirs + ['.'],
                    cpp_args: test_args,
                    link_with: _linkto,
                    link_language: 'cpp',
                ),
                workdir: meson.source_root(),
                suite: 'expr',
            )
        endif
    endif
    if has_metatomic
        test_array += [
            [
                'MetatomicTest',
                'metatomic_test',
                'MetatomicPotTest.cc',
                '/CppCore/tests',
            ],
            [
                'UmaPotTest',
                'uma_pot_test',
                'UmaPotTest.cc',
                '/CppCore/tests',
            ],
        ]
        executable(
            'uma_ase_parity',
            sources: ['tests/uma_ase_parity.cc'],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
        )
        if _rpc_stack
            uma_engine_env = environment()
            uma_engine_env.set('RGPOT_UMA_ENGINE', uma_engine.full_path())
            uma_engine_test = executable(
                'uma_engine_test',
                sources: ['tests/UmaEngineTest.cc'],
                dependencies: test_deps + [capnp_rpc_dep, ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            )
            test(
                'UmaEngineTest',
                uma_engine_test,
                env: uma_engine_env,
                depends: [uma_engine],
                workdir: meson.source_root(),
            )
        endif
        if _rpc_stack
            executable(
                'uma_engine_bench',
                sources: ['tests/uma_engine_bench.cc'],
                dependencies: test_deps + [capnp_rpc_dep, ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            )
        endif
    endif
    # NWChem/CPMD frontend tests (need ptlrpc + frontends from _rpc_stack)
    if _rpc_stack and not get_option('with_rpc_client_only')
        nwchemc_fake_engine = shared_library(
            'nwchemc_fake_engine',
            sources: ['tests/nwchemc_fake_engine.c'],
            include_directories: include_directories('rgpot/NWChemPot'),
            install: false,
        )
        nwchemc_lifetime_engine = shared_library(
            'nwchemc_lifetime_engine',
            sources: ['tests/nwchemc_lifetime_engine.cc'],
            include_directories: _incdirs,
            install: false,
        )
        nwchem_lifetime_test = executable(
            'nwchem_lifetime_test',
            sources: ['tests/NWChemPotLifetimeTest.cc'],
            dependencies: test_deps + [ptlrpc_dep],
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
        )
        foreach lifetime_mode : ['ordinary', 'probe']
            test(
                'NWChemPotLifetime-' + lifetime_mode,
                find_program('python3'),
                args: [
                    meson.current_source_dir() / 'tests/test_nwchem_lifetime.py',
                    nwchem_lifetime_test.full_path(),
                    nwchemc_lifetime_engine.full_path(),
                    lifetime_mode,
                ],
                depends: [nwchem_lifetime_test, nwchemc_lifetime_engine],
                suite: 'nwchem',
            )
        endforeach
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
                build_rpath: _metatomic_build_rpath,
            ),
            env: nwchemc_fake_env,
            depends: nwchemc_fake_engine,
            workdir: meson.source_root(),
            suite: 'nwchem',
        )
        if has_skala
            test(
                'SkalaPotTest',
                executable(
                    'skala_pot_test',
                    sources: ['tests/SkalaPotTest.cc'],
                    dependencies: test_deps + [ptlrpc_dep],
                    include_directories: _incdirs + ['.'],
                    cpp_args: test_args,
                    link_with: _linkto,
                    link_language: 'cpp',
                ),
                env: nwchemc_fake_env,
                depends: nwchemc_fake_engine,
                workdir: meson.source_root(),
                suite: 'skala',
            )
        endif
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
                build_rpath: _metatomic_build_rpath,
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
        cpmdc_gradient_engine = shared_library(
            'cpmdc_gradient_engine',
            sources: ['tests/cpmdc_gradient_engine.cc'],
            include_directories: _incdirs + ['.'] + include_directories(
                'rgpot/CPMDPot',
            ),
            install: false,
        )
        cpmdc_gradient_env = environment()
        cpmdc_gradient_env.set(
            'RGPOT_CPMD_ENGINE',
            cpmdc_gradient_engine.full_path(),
        )
        test(
            'CPMDPotGradientStressTest',
            executable(
                'cpmd_pot_gradient_stress_test',
                sources: ['tests/CPMDPotGradientStressTest.cc'],
                dependencies: test_deps + [ptlrpc_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            ),
            env: cpmdc_gradient_env,
            depends: cpmdc_gradient_engine,
            workdir: meson.source_root(),
            suite: 'cpmd',
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
                build_rpath: _metatomic_build_rpath,
            ),
            env: cpmdc_fake_env,
            depends: cpmdc_fake_engine,
            workdir: meson.source_root(),
            suite: 'cpmd',
        )
        mpiexec = find_program('mpirun', 'mpiexec', required: false)
        if _mpi_dep.found() and mpiexec.found()
            mpi_host_contract = executable(
                'mpi_host_contract',
                'tests/MpiHostContractTest.cc',
                dependencies: [_mpi_dep],
                include_directories: _incdirs,
            )
            collective_engine = shared_library(
                'collective_engine',
                'tests/nwchemc_collective_engine.cc',
                dependencies: [ptlrpc_dep, _mpi_dep],
                include_directories: _incdirs,
                install: false,
            )
            mpi_rpc_client = executable(
                'mpi_rpc_contract_client',
                'tests/MpiRpcContractClient.cc',
                dependencies: _deps + [ptlrpc_dep],
                include_directories: _incdirs,
                link_with: _linkto,
                cpp_args: test_args,
                build_rpath: _metatomic_build_rpath,
            )
            mpi_host_script = meson.current_source_dir() / 'tests/test_mpi_hosts.py'
            foreach mode : ['owned', 'borrowed']
                test(
                    'MpiHost-' + mode,
                    find_program('python3'),
                    args: [
                        mpi_host_script,
                        mode,
                        mpiexec.full_path(),
                        mpi_host_contract.full_path(),
                    ],
                    depends: mpi_host_contract,
                    timeout: 60,
                    suite: 'mpi-host',
                )
            endforeach
            test(
                'NWChemMpiForceHost',
                find_program('python3'),
                args: [
                    mpi_host_script,
                    'force',
                    mpiexec.full_path(),
                    nwchem_mpi_force_host.full_path(),
                    collective_engine.full_path(),
                ],
                depends: [nwchem_mpi_force_host, collective_engine],
                timeout: 60,
                suite: 'mpi-host',
            )
            foreach backend : ['NWChem', 'CPMD']
                foreach layout : ['serial', 'mpi']
                    rpc_server = layout == 'mpi' ? mpi_server : server
                    test(
                        'CollectiveRPC-' + backend + '-' + layout,
                        find_program('python3'),
                        args: [
                            mpi_host_script,
                            'rpc',
                            mpiexec.full_path(),
                            rpc_server.full_path(),
                            mpi_rpc_client.full_path(),
                            collective_engine.full_path(),
                            backend,
                            layout,
                        ],
                        depends: [rpc_server, mpi_rpc_client, collective_engine],
                        timeout: 60,
                        suite: 'mpi-host',
                    )
                endforeach
            endforeach
            rank_exception = executable(
                'rank_exception_abort',
                sources: ['tests/RankExceptionAbort.cc'],
                dependencies: test_deps + [ptlrpc_dep, rgpot_mpi_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            )
            rank_exception_sh = find_program(
                meson.current_source_dir() / 'tests/rank_exception_abort.sh',
            )
            test(
                'RankExceptionAbort',
                rank_exception_sh,
                args: [rank_exception.full_path(), 'abort'],
                env: cpmdc_fake_env,
                depends: [cpmdc_fake_engine, rank_exception],
                timeout: 60,
                is_parallel: false,
                suite: 'cpmd',
            )
            test(
                'CalculatorUnevenBatch',
                rank_exception_sh,
                args: [rank_exception.full_path(), 'uneven'],
                env: cpmdc_fake_env,
                depends: [cpmdc_fake_engine, rank_exception],
                timeout: 60,
                is_parallel: false,
                suite: 'cpmd',
            )
            test(
                'RankExceptionFinalizeOwner',
                rank_exception_sh,
                args: [rank_exception.full_path(), 'finalize-owner'],
                env: cpmdc_fake_env,
                depends: [rank_exception],
                timeout: 60,
                is_parallel: false,
                suite: 'cpmd',
            )
            test(
                'RankExceptionFinalizeGuest',
                rank_exception_sh,
                args: [rank_exception.full_path(), 'finalize-guest'],
                env: cpmdc_fake_env,
                depends: [rank_exception],
                timeout: 60,
                is_parallel: false,
                suite: 'cpmd',
            )
            test(
                'ShareBroadcastMismatch',
                rank_exception_sh,
                args: [rank_exception.full_path(), 'share-bad'],
                env: cpmdc_fake_env,
                depends: [rank_exception],
                timeout: 60,
                is_parallel: false,
                suite: 'cpmd',
            )
            test(
                'ShareBroadcastMatch',
                rank_exception_sh,
                args: [rank_exception.full_path(), 'share-ok'],
                env: cpmdc_fake_env,
                depends: [rank_exception],
                timeout: 60,
                is_parallel: false,
                suite: 'cpmd',
            )
            refused_split = executable(
                'refused_split',
                sources: ['tests/RefusedSplit.cc'],
                dependencies: test_deps + [ptlrpc_dep, rgpot_mpi_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            )
            refused_split_sh = find_program(
                meson.current_source_dir() / 'tests/refused_split.sh',
            )
            test(
                'RefusedSplit',
                refused_split_sh,
                args: [refused_split.full_path()],
                depends: [refused_split],
                timeout: 60,
                is_parallel: false,
                suite: 'cpmd',
            )
            cpmdc_adopt_engine = shared_library(
                'cpmdc_adopt_engine',
                sources: ['tests/cpmdc_adopt_engine.cc'],
                dependencies: [_mpi_dep],
                include_directories: _incdirs + ['.'] + include_directories(
                    'rgpot/CPMDPot',
                ),
                install: false,
            )
            mpi_split_profile = shared_library(
                'mpi_split_profile',
                sources: ['tests/mpi_split_profile.c'],
                dependencies: [dependency('mpi', language: 'c')],
                install: false,
            )
            adopt_comm = executable(
                'adopt_comm',
                sources: ['tests/AdoptComm.cc'],
                dependencies: test_deps + [ptlrpc_dep, rgpot_mpi_dep, _dl_dep],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
                build_rpath: _metatomic_build_rpath,
            )
            adopt_comm_sh = find_program(
                meson.current_source_dir() / 'tests/adopt_comm.sh',
            )
            foreach entry : [
                ['AdoptCommMissing', 'missing', cpmdc_gradient_engine],
                ['AdoptCommSplit', 'split', cpmdc_adopt_engine],
                ['AdoptCommRefusedHook', 'refused-hook', cpmdc_adopt_engine],
                ['AdoptCommProbe', 'probe', cpmdc_adopt_engine],
                ['AdoptCommInstanceFirst', 'instance-first', cpmdc_adopt_engine],
                ['AdoptCommBoundStatic', 'bound-static', cpmdc_adopt_engine],
                ['AdoptCommBoundInstance', 'bound-instance', cpmdc_adopt_engine],
                ['AdoptCommRefusedForce', 'refused-force', cpmdc_adopt_engine],
            ]
                test(
                    entry[0],
                    adopt_comm_sh,
                    args: [
                        entry[1],
                        adopt_comm.full_path(),
                        entry[2].full_path(),
                        mpi_split_profile.full_path(),
                    ],
                    depends: [adopt_comm, entry[2], mpi_split_profile],
                    timeout: 60,
                    is_parallel: false,
                    suite: 'cpmd',
                )
            endforeach
            if get_option('with_cache')
                collective_cache = executable(
                    'collective_cache_test',
                    sources: ['tests/CollectiveCacheTest.cc'],
                    dependencies: test_deps + [rgpot_mpi_dep],
                    include_directories: _incdirs + ['.'],
                    cpp_args: test_args,
                    link_with: _linkto,
                    link_language: 'cpp',
                    build_rpath: _metatomic_build_rpath,
                )
                test(
                    'CollectiveCache',
                    find_program(
                        meson.current_source_dir() / 'tests/collective_cache.sh',
                    ),
                    args: [collective_cache.full_path()],
                    depends: [collective_cache],
                    timeout: 60,
                    is_parallel: false,
                    suite: 'cpmd',
                )
            endif
        endif
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
                build_rpath: _metatomic_build_rpath,
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
    if not get_option('with_rpc_client_only')
        mopacc_fake_engine = shared_library(
            'mopacc_fake_engine',
            sources: ['tests/mopacc_fake_engine.c'],
            include_directories: include_directories('rgpot/MOPACPot'),
            install: false,
        )
        mopacc_fake_env = environment()
        mopacc_fake_env.set(
            'RGPOT_MOPACC_ENGINE',
            mopacc_fake_engine.full_path(),
        )
        test(
            'MOPACPotTest',
            executable(
                'mopac_pot_test',
                sources: ['tests/MOPACPotTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            env: mopacc_fake_env,
            depends: mopacc_fake_engine,
            workdir: meson.source_root(),
            suite: 'mopac',
        )
        test(
            'test_mopac_abi',
            executable(
                'test_mopac_abi',
                sources: ['tests/test_mopac_abi_main.c'],
                dependencies: [mopac_abi_stub_dep],
                include_directories: include_directories('rgpot/MOPACPot'),
            ),
            suite: 'mopac',
        )
        test(
            'MOPACDlopenContract',
            find_program('python3'),
            args: [
                meson.current_source_dir() / 'tests/test_mopac_dlopen_contract.py',
                meson.source_root(),
            ],
            suite: 'mopac',
        )
    endif

    calculator_group_test_exe = disabler()
    foreach test : test_array
        _test_exe = executable(
            test.get(1),
            sources: ['tests/' + test.get(2)],
            dependencies: test_deps,
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            link_with: _linkto,
            link_language: 'cpp',
            build_rpath: _metatomic_build_rpath,
        )
        test(
            test.get(0),
            _test_exe,
            workdir: meson.source_root() + test.get(3),
        )
        if test.get(1) == 'calculator_group_test'
            calculator_group_test_exe = _test_exe
        endif
    endforeach
    if not get_option('with_rpc_client_only')
        layout_fixture = shared_library(
            'rgpot_layout_fixture',
            sources: ['tests/layout_fixture.cc'],
            include_directories: _incdirs + ['.'],
            cpp_args: test_args,
            dependencies: test_deps,
            install: false,
        )
        test(
            'CxxLayoutTest',
            executable(
                'cxx_layout_test',
                sources: ['tests/CxxLayoutTest.cc'],
                dependencies: test_deps,
                include_directories: _incdirs + ['.'],
                cpp_args: test_args,
                link_with: _linkto,
                link_language: 'cpp',
            ),
            env: {'RGPOT_LAYOUT_FIXTURE': layout_fixture.full_path()},
            depends: [layout_fixture],
            workdir: meson.source_root(),
        )
        if host_system != 'windows'
            # The generic engine table over a stand-in backend: no rpc stack,
            # no torch.
            engine_stub = shared_library(
                'engine_stub',
                ['tests/engine_stub.cc', 'rgpot/engine/EngineTable.cc'],
                include_directories: _incdirs + ['.'],
                cpp_args: test_args + [
                    '-DRGPOT_ENGINE_PROJECT_VERSION="@0@"'.format(meson.project_version()),
                ],
                dependencies: test_deps,
                install: false,
            )
            engine_stub_env = environment()
            engine_stub_env.set('RGPOT_ENGINE_STUB', engine_stub.full_path())
            engine_stub_env.set(
                'RGPOT_ENGINE_STUB_MODEL',
                meson.current_build_dir() / 'engine_stub_model.bin',
            )
            test(
                'EngineAdapterTest',
                executable(
                    'engine_adapter_test',
                    sources: ['tests/EngineAdapterTest.cc'],
                    dependencies: test_deps,
                    include_directories: _incdirs + ['.'],
                    cpp_args: test_args,
                    link_language: 'cpp',
                ),
                env: engine_stub_env,
                depends: [engine_stub],
                workdir: meson.source_root(),
            )
        endif
        if _rpc_stack and host_system != 'windows'
            engine_table_env = environment()
            engine_table_expect = ['CLASSICAL', 'MOPAC', 'NWCHEM']
            engine_table_depends = [classical_engine, mopac_engine, nwchem_engine]
            engine_table_env.set('RGPOT_ENGINE_CLASSICAL', classical_engine.full_path())
            engine_table_env.set('RGPOT_ENGINE_MOPAC', mopac_engine.full_path())
            engine_table_env.set('RGPOT_ENGINE_NWCHEM', nwchem_engine.full_path())
            if has_skala
                engine_table_expect += ['SKALA']
                engine_table_depends += [skala_engine]
                engine_table_env.set('RGPOT_ENGINE_SKALA', skala_engine.full_path())
            endif
            if has_xtb
                engine_table_expect += ['XTB']
                engine_table_depends += [xtb_engine]
                engine_table_env.set('RGPOT_ENGINE_XTB', xtb_engine.full_path())
            endif
            if has_metatomic
                engine_table_expect += ['METATOMIC']
                engine_table_depends += [metatomic_engine]
                engine_table_env.set('RGPOT_ENGINE_METATOMIC', metatomic_engine.full_path())
            endif
            engine_table_env.set('RGPOT_ENGINE_EXPECT', ','.join(engine_table_expect))
            test(
                'EngineTableTest',
                executable(
                    'engine_table_test',
                    sources: ['tests/EngineTableTest.cc'],
                    dependencies: test_deps + [capnp_rpc_dep, ptlrpc_dep],
                    include_directories: _incdirs + ['.'],
                    cpp_args: test_args,
                    link_with: _linkto,
                    link_language: 'cpp',
                    build_rpath: _metatomic_build_rpath,
                ),
                env: engine_table_env,
                depends: engine_table_depends,
                workdir: meson.source_root(),
            )
        endif
        python3_prog = find_program('python3', required: false)
        if python3_prog.found()
            test(
                'CxxLayoutVirtualCounts',
                python3_prog,
                args: [files('tests/check_virtual_counts.py')],
            )
        endif
    endif
    if _mpi_dep.found() and not is_disabler(calculator_group_test_exe)
        _calc_mpiexec = find_program('mpirun', 'mpiexec', required: false)
        if _calc_mpiexec.found()
            calculator_group_mpi_sh = find_program(
                meson.current_source_dir() / 'tests/calculator_group_mpi.sh',
            )
            test(
                'CalculatorGroupShare',
                calculator_group_mpi_sh,
                args: [calculator_group_test_exe.full_path(), 'share'],
                depends: [calculator_group_test_exe],
                timeout: 60,
                is_parallel: false,
                suite: 'cpmd',
            )
            test(
                'CalculatorGroupBadOwner',
                calculator_group_mpi_sh,
                args: [calculator_group_test_exe.full_path(), 'bad-owner'],
                depends: [calculator_group_test_exe],
                timeout: 60,
                is_parallel: false,
                suite: 'cpmd',
            )
        endif
    endif

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
                build_rpath: _metatomic_build_rpath,
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
            build_rpath: _metatomic_build_rpath,
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
            build_rpath: _metatomic_build_rpath,
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
# XTBDlopen frontend still builds; set RGPOT_XTB_ENGINE for libxtb_engine.so.
# See docs/eon_pkgconfig.md.
if not meson.is_subproject()
    install_headers('../include/rgpot/plugin.h', subdir: 'rgpot')
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
            _pc_cflags += ['-DRGPOT_HAS_UMA=TRUE']
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
        if has_expr
            _pc_cflags += ['-DRGPOT_HAS_EXPR=TRUE']
        endif
        if has_skala
            _pc_cflags += ['-DRGPOT_HAS_SKALA=TRUE']
        endif
        if get_option('with_cache')
            # PotentialCache.hpp includes rocksdb/db.h.
            _pc_cflags += ['-DRGPOT_HAS_CACHE=TRUE']
            _pc_requires += ['rocksdb']
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
    if _mpi_dep.found()
        # The MPI side of the calculator groups. librgpot loads it with
        # dlopen; linking it is for hosts that want MPI mapped up front.
        pkg.generate(
            rgpot_mpi_lib,
            name: 'rgpot-mpi',
            description: 'rgpot calculator groups over MPI (dlopen side of librgpot)',
            url: 'https://github.com/OmniPotentRPC/rgpot',
            version: meson.project_version(),
            subdirs: ['.'],
            extra_cflags: ['-DRGPOT_HAS_MPI'],
            filebase: 'rgpot-mpi',
        )
    endif
endif

if _mpi_dep.found()
    # Subproject consumers: dependency('rgpot-mpi') resolves to this build.
    meson.override_dependency('rgpot-mpi', rgpot_mpi_dep)
endif
