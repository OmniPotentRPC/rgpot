# Find dependencies and tools
capnp_rpc_dep = dependency('capnp-rpc')
capnpc = find_program('capnpc')

output_dir = meson.current_source_dir() / 'gen'

gen_rpc = custom_target(
    'gen_rpc',
    input: 'Potentials.capnp',
    output: ['Potentials.capnp.c++', 'Potentials.capnp.h'],
    command: [
        'sh',
        '-c',
        'mkdir -p @0@ && @1@ -o c++:@0@ --src-prefix=@2@ @3@'.format(
            output_dir,
            capnpc.path(),
            meson.current_source_dir(),
            '@INPUT@',
        ),
    ],
)

gen_inc = include_directories('gen')
_incdirs += gen_inc

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
