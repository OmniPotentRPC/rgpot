# Linked XTBPot (NEEDED libxtb) + optional plugin engine for dlopen frontend.

dl_dep = dependency('dl', required: false)
if not dl_dep.found()
    dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
endif

xtb_inc = include_directories('.')

# Static convenience lib; ships inside the librgpot umbrella (link_whole).
xtbpot = static_library(
    'xtbpot',
    'XTBPot.cc',
    dependencies: _deps,
    cpp_args: _args,
    include_directories: _incdirs + [xtb_inc],
    pic: true,
    install: false,
)

# Thin dlopen frontend — no NEEDED on libxtb (engine carries it).
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

# Portable plugin: links libxtb + XTBPot implementation via XTBEngineAbi.
xtb_engine = library(
    'xtb_engine',
    [
        'XTBPot.cc',
        'XTBEngineAbi.cc',
    ],
    dependencies: _deps + [vesin_dep],
    cpp_args: _args + ['-DRGPOT_XTB_ENGINE_BUILD'],
    include_directories: _incdirs + [xtb_inc],
    install: not meson.is_subproject(),
    gnu_symbol_visibility: 'default',
)

xtb_engine_dep = declare_dependency(
    link_with: [xtb_engine],
    include_directories: _incdirs + [xtb_inc],
)

# Shared library directory. Tests put this on the process-start loader path
# so a bad explicit engine_path cannot be rescued by the bare soname.
xtb_engine_dir = meson.current_build_dir()
