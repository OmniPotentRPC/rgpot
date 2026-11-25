# Copyright 2023--present rgpot developers
dxtb_engine = library(
    'dxtb_engine',
    [
        'DxtbPot.cc',
        '../engine/DxtbEngine.cc',
        '../engine/EngineTable.cc',
    ],
    dependencies: _deps_sans_core + [
        capnp_rpc_dep,
        ptlrpc_dep,
    ],
    link_with: [rgpot_core],
    cpp_args: _args + [
        '-DRGPOT_ENGINE_BUILD',
        '-DRGPOT_ENGINE_PROJECT_VERSION="@0@"'.format(
            meson.project_version(),
        ),
    ],
    include_directories: _incdirs,
    install: not meson.is_subproject(),
    gnu_symbol_visibility: 'default',
)
