# NWChemPot: pure consumer of the split nwchemc engine.
#
# The frontend (NWChemPot.cc) always builds and resolves libnwchemc.so by
# dlopen at runtime -- rgpot carries no in-process NWChem embed of its own.
# The no-op ABI stub is built for the C ABI conformance test only.

nwchem_inc = include_directories('.')

nwchem_abi_stub = static_library(
    'nwchem_abi_stub',
    'nwchem_c_abi_stub.c',
    include_directories: nwchem_inc,
    install: false,
)

nwchem_frontend_cpp_args = ['-DRGPOT_HAS_NWCHEM_FRONTEND=TRUE']

nwchem_dl_dep = dependency('dl', required: false)
if not nwchem_dl_dep.found()
    nwchem_dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
endif

nwchempot = static_library(
    'nwchempot',
    'NWChemPot.cc',
    dependencies: _deps + [nwchem_dl_dep, ptlrpc_hdr_dep],
    cpp_args: _args + nwchem_frontend_cpp_args,
    include_directories: _incdirs + [nwchem_inc],
    pic: true,
    # Convenience archive. librgpot link_wholes this next to rgpot_core.
    # Installing it makes Meson promote link_with of the uninstalled core
    # to link_whole, which copies PotHelpers.cc and units.cc into this
    # archive. The umbrella link then fails with duplicate symbols. That
    # stays true if this subdir moves to after the core dep is added.
    # Client-only builds have no umbrella and no core archive, so the
    # static library is the installed product.
    install: get_option('with_rpc_client_only') and not meson.is_subproject(),
)

nwchempot_dep = declare_dependency(
    link_with: [nwchempot],
    dependencies: [nwchem_dl_dep, ptlrpc_dep],
    include_directories: _incdirs + [nwchem_inc],
    compile_args: nwchem_frontend_cpp_args,
)

nwchem_abi_stub_dep = declare_dependency(
    link_with: nwchem_abi_stub,
    include_directories: nwchem_inc,
)
