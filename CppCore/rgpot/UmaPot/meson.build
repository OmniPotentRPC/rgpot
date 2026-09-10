# UMA / OMol: vesin neighbor list + AOTInductor .pt2 (needs torch from _deps).
uma = static_library(
    'uma',
    'UmaPot.cc',
    dependencies: _deps + [vesin_dep],
    cpp_args: _args,
    include_directories: ['../../'],
    pic: true,
    install: false,
)

# Portable engine plugin behind the generic rgpot engine C ABI
# (engine_c_abi.h): dlopen'd by hosts (eonclient), never linked, so the
# host carries no torch NEEDED entries. Config travels as a capnp
# UmaParams message, so the plugin folds in the generated schema TU
# (ptlrpc) and needs the rpc stack at build time. RUNPATH mirrors the
# metatomic engine; both reuse variables MetatomicPot/meson.build minted.
if _rpc_stack
    uma_engine = library(
        'uma_engine',
        [
            'UmaPot.cc',
            'UmaEngineAbi.cc',
        ],
        # ptlrpc_dep carries the generated Potentials.capnp.h include
        # view plus the schema TU objects (final-binary rule: a plugin
        # is a leaf, no double-compile risk).
        dependencies: _deps + [dl_dep, vesin_dep, capnp_rpc_dep, ptlrpc_dep],
        cpp_args: _args + ['-DRGPOT_ENGINE_BUILD'],
        include_directories: ['../../'],
        install: not meson.is_subproject(),
        build_rpath: _engine_build_rpath,
        install_rpath: _engine_install_rpath,
        gnu_symbol_visibility: 'default',
    )

    uma_engine_dep = declare_dependency(
        link_with: [uma_engine],
        include_directories: ['../../'],
    )
else
    message('uma_engine skipped: capnp config needs with_rpc or with_rpc_client_only')
endif
