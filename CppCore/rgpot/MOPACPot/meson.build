# MOPACPot: pure consumer of the split mopacc engine.
#
# The frontend always builds and resolves libmopacc.so by dlopen. rgpot
# carries no OpenMOPAC embed. The no-op ABI stub is for the conformance test.

mopac_inc = include_directories('.')

mopac_abi_stub = static_library(
    'mopac_abi_stub',
    'mopac_c_abi_stub.c',
    include_directories: mopac_inc,
    install: false,
)

mopac_frontend_cpp_args = ['-DRGPOT_HAS_MOPAC_FRONTEND=TRUE']

mopac_dl_dep = dependency('dl', required: false)
if not mopac_dl_dep.found()
    mopac_dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
endif

mopacpot = static_library(
    'mopacpot',
    'MOPACPot.cc',
    dependencies: _deps + [mopac_dl_dep],
    cpp_args: _args + mopac_frontend_cpp_args,
    include_directories: _incdirs + [mopac_inc],
    pic: true,
    # Convenience archive only. librgpot link_wholes this next to rgpot_core.
    # An installed static library makes Meson copy the uninstalled core into
    # this archive, and the umbrella link then fails with duplicate symbols.
    install: false,
)

mopacpot_dep = declare_dependency(
    link_with: [mopacpot],
    dependencies: [mopac_dl_dep],
    include_directories: _incdirs + [mopac_inc],
    compile_args: mopac_frontend_cpp_args,
)

mopac_abi_stub_dep = declare_dependency(
    link_with: mopac_abi_stub,
    include_directories: mopac_inc,
)
