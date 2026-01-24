# Find dependencies and tools
capnp_rpc_dep = dependency('capnp-rpc')
capnpc = find_program('capnpc')

gen_rpc = custom_target(
    'gen_rpc',
    input: 'Potentials.capnp',
    output: ['Potentials.capnp.c++', 'Potentials.capnp.h'],
    command: [
        capnpc,
        '-o',
        'c++:@OUTDIR@',
        '--src-prefix=' + meson.current_source_dir(),
        '@INPUT@',
    ],
)
# Use the target object itself for includes to ensure proper dependency tracking
_incdirs += include_directories('.')

ptlrpc = library(
    'ptlrpc',
    gen_rpc,
    dependencies: _deps + capnp_rpc_dep,
    cpp_args: _args,
    include_directories: _incdirs,
    install: true,
)

ptlrpc_dep = declare_dependency(
    link_with: ptlrpc,
    sources: gen_rpc,
    include_directories: _incdirs,
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
