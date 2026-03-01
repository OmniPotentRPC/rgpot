# Find dependencies and tools
capnp_rpc_dep = dependency('capnp-rpc')
capnp = find_program('capnp')
python = find_program('python3', 'python')

gen_rpc = custom_target(
    'gen_rpc',
    input: 'Potentials.capnp',
    output: ['Potentials.capnp.cpp', 'Potentials.capnp.h'],
    command: [
        python,
        meson.current_source_dir() / 'capnp_compile.py',
        capnp,
        '@OUTDIR@',
        meson.current_source_dir(),
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
    install: not meson.is_subproject(),
)

_ptlrpc_extra_deps = [capnp_rpc_dep]
if host_machine.system() == 'windows'
    _ptlrpc_extra_deps += cppc.find_library('ws2_32')
endif

ptlrpc_dep = declare_dependency(
    link_with: ptlrpc,
    sources: gen_rpc,
    include_directories: _incdirs,
    dependencies: _ptlrpc_extra_deps,
)

pot_bridge = library(
    'pot_client_bridge',
    'pot_bridge.cc',
    link_with: ptlrpc,
    dependencies: [capnp_rpc_dep],
    cpp_args: _args,
    include_directories: _incdirs,
    install: not meson.is_subproject(),
)

pot_bridge_dep = declare_dependency(
    link_with: pot_bridge,
)

if not get_option('with_rpc_client_only')
    server = executable(
        'potserv',
        ['server.cpp', gen_rpc],
        link_with: [ptlrpc, _linkto],
        dependencies: _deps + capnp_rpc_dep,
        cpp_args: _args,
        include_directories: _incdirs,
        install: not meson.is_subproject(),
    )
endif
