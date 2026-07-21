# XTB: thin dlopen frontend (always) + linked pot + engine plugin (with_xtb)
#
# Match Metatomic: hosts link XTBDlopen without pkg-config xtb / NEEDED libxtb.
# Linked XTBPot and libxtb_engine.so still require -Dwith_xtb=true.

xtb_inc = include_directories('.')

dl_dep = dependency('dl', required: false)
if not dl_dep.found()
    dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
endif

# Thin frontend — no libxtb; safe for with_xtb=false hosts (eOn, gprd paper).
xtb_dlopen = static_library(
    'xtb_dlopen',
    'XTBDlopen.cc',
    dependencies: _deps + [dl_dep],
    cpp_args: _args,
    include_directories: _incdirs + [xtb_inc],
    install: false,
)

xtb_dlopen_dep = declare_dependency(
    link_with: [xtb_dlopen],
    dependencies: [dl_dep],
    include_directories: _incdirs + [xtb_inc],
)

# Fat linked pot + portable engine plugin only when libxtb is available.
if has_xtb
    xtbpot = library(
        'xtbpot',
        'XTBPot.cc',
        dependencies: _deps,
        cpp_args: _args,
        include_directories: _incdirs + [xtb_inc],
        install: not meson.is_subproject(),
    )

    # Portable plugin: links libxtb + XTBPot implementation via XTBEngineAbi.
    xtb_engine = library(
        'xtb_engine',
        [
            'XTBPot.cc',
            'XTBEngineAbi.cc',
        ],
        dependencies: _deps,
        cpp_args: _args + ['-DRGPOT_XTB_ENGINE_BUILD'],
        include_directories: _incdirs + [xtb_inc],
        install: not meson.is_subproject(),
        gnu_symbol_visibility: 'default',
    )

    xtb_engine_dep = declare_dependency(
        link_with: [xtb_engine],
        include_directories: _incdirs + [xtb_inc],
    )
endif
