# MOPACPot: pure consumer of the split mopacc engine.
#
# The frontend (MOPACPot.cc) always builds and resolves libmopacc.so by
# dlopen at runtime -- rgpot carries no in-process OpenMOPAC embed of its
# own. Params are a packed MopacCParams blob (no Cap'n Proto).

mopac_inc = include_directories('.')

mopac_frontend_cpp_args = ['-DRGPOT_HAS_MOPAC_FRONTEND=TRUE']

mopac_dl_dep = dependency('dl', required: false)
if not mopac_dl_dep.found()
    mopac_dl_dep = meson.get_compiler('cpp').find_library('dl', required: false)
endif

mopacpot = static_library(
    'mopacpot',
    'MOPACPot.cc',
    dependencies: _deps + [mopac_dl_dep],
    cpp_args: _args + mopac_frontend_cpp_args,
    include_directories: _incdirs + [mopac_inc],
    install: not meson.is_subproject(),
)

mopacpot_dep = declare_dependency(
    link_with: [mopacpot],
    dependencies: [mopac_dl_dep],
    include_directories: _incdirs + [mopac_inc],
    compile_args: mopac_frontend_cpp_args,
)
