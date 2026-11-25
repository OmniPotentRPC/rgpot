# Public LAMMPS / GROMACS / OpenMM constructors. No engine library is
# linked here: LAMMPS dlopens its plugin, and the other two dlopen a
# native-unit library only when the caller passes a path.

external_pot_dl = dependency('dl', required: false)
if not external_pot_dl.found()
    external_pot_dl = meson.get_compiler('cpp').find_library(
        'dl',
        required: false,
    )
endif

# The lammpc path of LammpsPot reads the generated Potentials.capnp.h.
# ptlrpc_hdr_dep lists that header as a source, which orders its generation
# before ExternalPot.cc in a parallel build.
external_pot_deps = _deps_sans_core + [external_pot_dl]
external_pot_args = _args
if _rpc_stack
    external_pot_deps += [ptlrpc_hdr_dep]
    external_pot_args += ['-DRGPOT_LAMMPC_PROFILE=1']
endif

external_pot = static_library(
    'external_pot',
    'ExternalPot.cc',
    dependencies: external_pot_deps,
    cpp_args: external_pot_args,
    include_directories: _incdirs,
    pic: true,
    install: false,
)

external_pot_dep = declare_dependency(
    link_with: external_pot,
    dependencies: [external_pot_dl],
    include_directories: _incdirs,
)
