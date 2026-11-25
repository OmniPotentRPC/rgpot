# Find dependencies and tools
capnp_rpc_dep = dependency('capnp-rpc')
capnp = find_program('capnp')
python = find_program('python3', 'python')

# The schema is rgconf's potential.capnp at the commit pinned in
# schema.pin, vendored byte for byte with the common.capnp it imports. The
# build compiles, installs and serves the vendored files as they are;
# schema-ordinal-ledger holds both copies (here and rgpot-core/schema) to
# Potentials.ordinals. The comparison with the pinned rgconf commit itself
# needs the network and runs in CI (schema_ledger.py check --fetch).
schema_ledger = files('schema_ledger.py')
if not meson.is_subproject()
    install_data(
        rgpot_schema_files,
        install_dir: get_option('includedir') / 'rgpot' / 'rpc',
    )
endif
test(
    'schema-ordinal-ledger',
    python,
    args: [schema_ledger, 'check', '--root', meson.project_source_root()],
    suite: 'schema',
)

# Potentials.capnp imports common.capnp, so the generated
# Potentials.capnp.h includes common.capnp.h and the translation units of
# both belong in ptlrpc.
gen_rpc = custom_target(
    'gen_rpc',
    input: rgpot_schema_files,
    output: [
        'Potentials.capnp.cpp',
        'Potentials.capnp.h',
        'common.capnp.cpp',
        'common.capnp.h',
    ],
    command: [
        python,
        meson.current_source_dir() / 'capnp_compile.py',
        capnp,
        '@OUTDIR@',
        meson.current_source_dir(),
        '@INPUT@',
    ],
    # Install the generated headers next to the schema for pkg-config consumers.
    install: not meson.is_subproject(),
    install_dir: [
        false,
        get_option('includedir') / 'rgpot' / 'rpc',
        false,
        get_option('includedir') / 'rgpot' / 'rpc',
    ],
)
# Use the target object itself for includes to ensure proper dependency tracking
_incdirs += include_directories('.')

# Convenience archive, not a shipped shared object: the generated capnp
# translation unit belongs inside the one librgpot SONAME, and a separate
# libptlrpc.so would become a runtime dependency of every consumer that
# links a QM frontend.
ptlrpc = static_library(
    'ptlrpc',
    gen_rpc,
    dependencies: _deps + capnp_rpc_dep + [vesin_dep],
    cpp_args: _args,
    include_directories: _incdirs,
    pic: true,
    install: false,
)

_ptlrpc_extra_deps = [capnp_rpc_dep]
if host_machine.system() == 'windows'
    _ptlrpc_extra_deps += cppc.find_library('ws2_32')
endif

# Header only: listing the generated .cpp here would compile a second copy
# of the capnp translation unit into every consumer, and the convenience
# archives all link_whole into one librgpot. The archives take this view;
# only final binaries take the objects through ptlrpc_dep below.
ptlrpc_hdr_dep = declare_dependency(
    sources: [gen_rpc[1], gen_rpc[3]],
    include_directories: _incdirs,
    dependencies: _ptlrpc_extra_deps,
)

ptlrpc_dep = declare_dependency(
    link_with: ptlrpc,
    dependencies: [ptlrpc_hdr_dep],
)

pot_bridge = library(
    'pot_client_bridge',
    ['pot_bridge.cc'],
    link_with: ptlrpc,
    dependencies: [capnp_rpc_dep, ptlrpc_hdr_dep],
    cpp_args: _args + ['-DPOT_BRIDGE_BUILD'],
    # gen_rpc outputs live in the build dir; include it explicitly so
    # Potentials.capnp.h is found even when only pot_bridge.cc is compiled first.
    include_directories: _incdirs + [include_directories('.')],
    install: not meson.is_subproject(),
)

pot_bridge_dep = declare_dependency(
    link_with: pot_bridge,
)
