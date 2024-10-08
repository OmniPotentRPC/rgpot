capnp_rpc_dep = dependency('capnp-rpc')
_mbproot = meson.project_build_root()
capnpc_ = find_program('capnpc')
gen_rpc = custom_target(
    'gen_rpc',
    input: ['Potentials.capnp'],
    output: ['Potentials.capnp.c++', 'Potentials.capnp.h'],
    # TODO(rg): Can this be cleaner?
    command: [
        capnpc_,
        f'-oc++:@_mbproot@/CppCore/rgpot/rpc',
        '@INPUT@',
        '--verbose',
        '--src-prefix=../CppCore/rgpot/rpc',
    ],
)
ptlrpc = library(
    'ptlrpc',
    gen_rpc,
    dependencies: _deps + capnp_rpc_dep,
    cpp_args: _args,
    include_directories: _incdirs,
    install: true,
)
server = executable(
    'potserv',
    ['server.cpp', gen_rpc],
    link_with: [ptlrpc, _linkto],
    dependencies: _deps + capnp_rpc_dep,
    cpp_args: _args,
    include_directories: _incdirs,
    install: true,
)
