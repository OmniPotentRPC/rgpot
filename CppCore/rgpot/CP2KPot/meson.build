# CP2KPot: dlopen consumer of the libcp2k shim (libcp2kc_rgpot).

cp2k_inc = include_directories('.')

cp2k_frontend_cpp_args = ['-DRGPOT_HAS_CP2K_FRONTEND=TRUE']

cp2k_dl_dep = dependency('dl', required: false)
if not cp2k_dl_dep.found()
    cp2k_dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
endif

cp2kpot = static_library(
    'cp2kpot',
    'CP2KPot.cc',
    'CP2KDeck.cc',
    dependencies: _deps + [cp2k_dl_dep, ptlrpc_hdr_dep],
    cpp_args: _args + cp2k_frontend_cpp_args,
    include_directories: _incdirs + [cp2k_inc],
    install: not meson.is_subproject(),
)

cp2kpot_dep = declare_dependency(
    link_with: [cp2kpot],
    dependencies: [cp2k_dl_dep, ptlrpc_dep],
    include_directories: _incdirs + [cp2k_inc],
    compile_args: cp2k_frontend_cpp_args,
)

# The shim is the only target that links libcp2k and MPI. It is built only
# with -Dwith_cp2k=enabled (or auto with libcp2k on the pkg-config path).
cp2k_lib_dep = dependency('libcp2k', required: get_option('with_cp2k'))
cp2kc_shim = disabler()
if cp2k_lib_dep.found()
    cp2k_mpi_c_dep = dependency('mpi', language: 'c', required: true)
    _cp2k_libdir = cp2k_lib_dep.get_variable(
        pkgconfig: 'libdir',
        default_value: '',
    )
    cp2kc_shim = shared_library(
        'cp2kc_rgpot',
        'cp2kc_shim.c',
        dependencies: [cp2k_lib_dep, cp2k_mpi_c_dep],
        include_directories: [cp2k_inc],
        build_rpath: _cp2k_libdir,
        install_rpath: _cp2k_libdir,
        install: not meson.is_subproject(),
    )
endif
