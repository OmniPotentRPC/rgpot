# Copyright 2023--present rgpot developers
dxtb_engine = library(
    'dxtb_engine',
    [
        'DxtbPot.cc',
        '../engine/DxtbEngine.cc',
        '../engine/EngineTable.cc',
        engine_abi_stamp_source,
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
    # A live handle owns a worker process and its socket; a dlclose of the
    # engine must not unmap the code that shuts the worker down and reaps it.
    link_args: meson.get_compiler('cpp').get_supported_link_arguments(
        ['-Wl,-z,nodelete'],
    ),
)
